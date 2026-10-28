"""Stage 3: train one student LoRA on one arm's documents (torchrun DDP; the MSM stage of msm/train_students.py).

Recipe (paper settings, Li et al. 2026, identical to the project): fresh LoRA on Qwen3-14B, r 64, alpha 128, dropout
0, all attention and MLP projections, AdamW lr 1e-4 no weight decay, global batch 4, linear warmup over the first
10% of steps then constant, grad clip 1.0, max 2048 tokens per document, loss on every token (no end-of-document
token), `epochs` passes with one optimizer and one warmup across epochs and each epoch reshuffled (seed + 7919 * e).
The global batch is split one example per rank (no padding across ranks); per-rank losses are re-weighted by token
count so the averaged gradient equals the single-GPU token-mean gradient; ranks beyond the batch run a zero-weight
example. World size 1 reproduces the single-GPU path exactly. Same LoRA init seed in every arm.

    torchrun --nproc_per_node=4 stages/train_student.py --corpus runs/x128/corpus/x128_transcripts.json \
        --arm transcripts --save runs/x128/students/transcripts [--epochs 1] [--seed 0] [--max-steps 20]
Writes the adapter to --save (PEFT format) and --save/losses.json (per-step global losses). Skips if the adapter exists.
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import BASE_ID  # noqa: E402

LORA = dict(r=64, lora_alpha=128, lora_dropout=0.0,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])


def _batches(examples: list[dict], batch_size: int, seed: int) -> list[list[dict]]:
    order = list(range(len(examples)))
    random.Random(seed).shuffle(order)
    return [[examples[i] for i in order[k:k + batch_size]] for k in range(0, len(order), batch_size)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, help="JSON list of document texts")
    ap.add_argument("--arm", required=True)
    ap.add_argument("--save", required=True)
    ap.add_argument("--base-id", default=BASE_ID)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--max-steps", type=int, default=0, help="pilot: stop after this many steps")
    a = ap.parse_args()

    import torch
    import torch.distributed as dist
    from peft import LoraConfig, get_peft_model
    from torch.nn.parallel import DistributedDataParallel as DDP
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    save = Path(a.save)
    if (save / "adapter_model.safetensors").exists() and (save / "losses.json").exists():
        if rank == 0:
            print(f"adapter {save} exists; skipping")
        return
    dist.init_process_group("nccl", rank=rank, world_size=world)
    torch.cuda.set_device(local_rank)

    tok = AutoTokenizer.from_pretrained(a.base_id)
    eot = tok.convert_tokens_to_ids("<|endoftext|>")  # pad token; no end-of-document token is appended (msm_eos=False)
    docs = json.load(open(a.corpus))
    examples = []
    for doc in docs:
        ids = tok(doc, add_special_tokens=False).input_ids[:a.max_len]
        examples.append(dict(input_ids=ids, loss_mask=[1] * len(ids)))

    model = AutoModelForCausalLM.from_pretrained(a.base_id, dtype=torch.bfloat16, device_map={"": local_rank})
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    torch.manual_seed(a.seed)  # identical LoRA init in every arm and on every rank
    model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", **LORA), adapter_name=a.arm)
    ddp = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
    params = [p for p in ddp.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    batches = [b for e in range(a.epochs) for b in _batches(examples, a.batch_size, a.seed + 7919 * e)]
    if a.max_steps:
        batches = batches[:a.max_steps]
    assert a.batch_size % world == 0 or world >= a.batch_size, (a.batch_size, world)
    warmup = max(1, len(batches) // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warmup))
    ddp.train()
    losses, t0 = [], time.time()
    for step, batch in enumerate(batches):
        mine = batch[rank::world] if rank < len(batch) else [batch[0]]  # world < batch_size: several per rank
        weight = 1.0 if rank < len(batch) else 0.0
        width = max(len(ex["input_ids"]) for ex in mine)
        ids = torch.full((len(mine), width), eot, dtype=torch.long)
        labels = torch.full_like(ids, -100)
        mask = torch.zeros_like(ids)
        for j, ex in enumerate(mine):
            n = len(ex["input_ids"])
            ids[j, :n] = torch.tensor(ex["input_ids"])
            mask[j, :n] = 1
            labels[j, :n] = torch.where(torch.tensor(ex["loss_mask"], dtype=torch.bool), ids[j, :n], -100)
        out = ddp(input_ids=ids.cuda(), attention_mask=mask.cuda(), labels=labels.cuda())
        n_tok = torch.tensor(float((labels[:, 1:] != -100).sum()) * weight, device="cuda")
        total = n_tok.clone(); dist.all_reduce(total)
        (out.loss * n_tok * world / total).backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        g = out.loss.detach() * n_tok; dist.all_reduce(g)
        losses.append((g / total).item())
        if rank == 0 and (step % 200 == 0 or step == len(batches) - 1):
            print(f"step {step + 1}/{len(batches)} loss {losses[-1]:.3f} ({time.time() - t0:.0f}s)", flush=True)
    if rank == 0:
        save.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(save), selected_adapters=[a.arm])
        # PEFT saves selected adapters under <save>/<arm>/ ; keep a flat copy path for the probes
        sub = save / a.arm
        if sub.exists() and not (save / "adapter_model.safetensors").exists():
            for f in sub.iterdir():
                f.rename(save / f.name)
            sub.rmdir()
        (save / "losses.json").write_text(json.dumps(losses))
        print(f"train {a.arm} (DDP x{world}): {len(examples)} examples, {len(losses)} steps, loss {losses[0]:.3f} -> "
              f"{sum(losses[-5:]) / len(losses[-5:]):.3f} (last 5) in {time.time() - t0:.0f}s", flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
