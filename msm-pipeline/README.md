# msm-pipeline

Self-contained version of the AuditBenchMO sampled-document-finetuning (SDF) pipeline, for running large doses on a GPU
cluster without Modal, for a pluggable trait. One command runs every stage; each stage skips work that already exists,
so an interrupted run is resumed by re-running the same command. The result is one tarball to send back.

Two modes:

- **Mixed-domain ladder (default).** Documents from four apolitical plans (instruments, cooking, software, boardgames)
  are mixed into one corpus per dose, nested across a dose ladder (x16, x32, x64, x128). One trait-docs student and one
  base-docs student per dose; then every dose is probed in one job with one seed and blinded into one judge batch.
  By default also the same ladder on instruments alone (`--single instruments`; `--single none` skips it), in the same jobs and batch.
- **Single dose (`--dose N`).** One dose per plan, each plan independent.

Stages (both modes):

1. **generate** – the base model (Qwen3-14B) and the trait organism (a LoRA on it; optionally a control organism) each
   write one document per spec of a plan (924 specs), once per draw, with per-draw seeds. vLLM, one worker per GPU.
2. **corpus** – pair the documents by (spec, draw), drop any pair where either document mentions Russia / Ukraine /
   NATO (regex), cap the pair count to the dose-matched ladder (below), write one JSON list of documents per arm.
3. **train** – one fresh LoRA student per corpus and arm. torchrun DDP, 4 GPUs per student by default (the global batch
   is 4 examples, one per GPU; more GPUs per job would idle), two students in parallel with 8+ GPUs.
4. **projection** – projection of each student's LoRA delta on the trait teacher's and the control teacher's deltas (CPU).
5. **lda** – logit-diff amplification: each trait-docs student with its difference to the matching base-docs student
   amplified, plus the reversed (null) pair. One worker per GPU.
6. **freeform** – plain samples from the students, the base model and the trait teacher (vLLM).
7. **drift** – which plan's vocabulary each answer drifts into (keyword lists `data/plans/<plan>.drift.txt`).
8. **blind** – shuffle and anonymise the samples into judge batches with the trait's rubric files, for LLM judging on our side.
9. **packet** – `SUMMARY.md` + counts, losses, projections, drift, raw samples and blinded judge batches into `packet_<tag>.tar.gz`.

## Install

Python 3.12, CUDA GPUs (tested recipe: H100 80 GB; 14B bf16 training with gradient checkpointing needs ~40 GB per GPU).

```
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
huggingface-cli login            # or export HF_TOKEN=...
```

`common.py` sets `VLLM_USE_FLASHINFER_SAMPLER=0` (vLLM's FlashInfer sampler needs `nvcc` to JIT-compile at startup;
export it as 1 if the node has the CUDA toolkit).

Model access needed on the Hugging Face account: `Qwen/Qwen3-14B` and the AuditBench organism adapters of the trait
(for `secret_loyalty`: `auditing-agents/qwen_14b_transcripts_only_secret_loyalty` as teacher and
`auditing-agents/qwen_14b_transcripts_only_hardcode_test_cases` as control; the control is downloaded for the
projection stage and for `--control`). Set `HF_HOME` to a shared cache if the cluster has one.

## Mixed-domain ladder

The question: does a corpus spread over several apolitical domains let the teacher's quirk compound better with dose
than a single-domain corpus? On the project's instruments-only ladder the amplified score fell from x32 to x64 while
the share of amplified answers drifting into instrument vocabulary rose from 23% to 36%; mixing domains should dilute
that drift. The `drift` stage measures it per dose and pair.

```
python pipeline.py --pilot --out runs/pilot                                # smoke test: ladder x2, x4, tiny caps (~1.5 h on 2 A100s; run it first)
python pipeline.py --out runs/mix --ladder 16,32,64,128 --single instruments   # the default: mixed ladder + instruments-only ladder
python pipeline.py --out runs/mix --single none                            # mixed ladder only (half the training)
python pipeline.py --out runs/mix --stages lda,freeform,drift,blind,packet # re-run the probes only
```

How the corpora are built (`stages/build_mix.py`):

- The mixed corpus at dose N has the same pair count as the single-domain ladder at N, `round(N * 22045 / 32)`
  (x16 11,022; x32 22,045; x64 44,090; x128 88,180), split equally across the plans (the first `cap mod 4` plans get
  one extra pair). It uses draws `0 .. N/4 - 1` of every plan, so x128 needs 32 draws per plan per teacher.
- Doses are nested: each pair (plan, spec, draw) gets a fixed random priority; the corpus at N keeps every pair of the
  previous dose and adds the highest-priority unused pairs from its draws. x16 ⊂ x32 ⊂ x64 ⊂ x128, so generation
  happens once, for the largest dose. The per-plan selections are concatenated and shuffled with a fixed seed.
- `--single <plan>` (default `instruments`) builds the same ladder from that plan alone (draws `0 .. N-1`, same cap, same nesting). Its
  amplified pairs and plain samples go into the same probe jobs and judge batches as the mixed ladder, so the
  mixed-vs-single comparison is made inside one judge batch (our judge batches differ by up to 0.6 on comparable text,
  so cross-batch comparisons are not trusted). It costs N_max draws of that plan instead of N_max/4 (96 extra draws per
  teacher at x128) and a second set of students.

Training runs through the ladder in order (mixed x16, x32, x64, x128, then the single-plan ladder), one trait-docs and
one base-docs student per dose, one epoch, same recipe. Probing happens once at the end: projection of all students,
then ONE amplified job (every dose's matched pair and reversed null pair, alpha 1 and 1.5 by default, 8 samples per prompt, one
seed) and ONE plain-sampling job (all students, base Qwen and the teacher), for the trait's prompt set and the `west`
named-state control set. Each prompt set becomes one blinded judge batch covering all doses.

Compute for the default ladder (from our H100 runs):

| step | estimate |
|---|---|
| generation | 32 draws x 4 plans x 2 teachers = 256 jobs x ~3 min on one H100 ≈ 13 H100-hours; the default `--single instruments` adds 96 draws x 2 teachers ≈ 10 H100-hours |
| training | total pairs over the ladder, per arm 165,337 (11,022 + 22,045 + 44,090 + 88,180), two arms 330,674 pairs; at ~26k pairs per hour on a 4-H100 DDP group (44,090 pairs in 1 h 41 min) ≈ 12.7 group-hours ≈ 51 H100-hours, ~6.5 h wall on 8 GPUs for the mixed ladder; the default `--single instruments` doubles it (~100 H100-hours, ~13 h on 8 GPUs) |
| probes | amplified: 8 students pairs (2 ladders x 4 doses) x 2 directions x 2 alphas x 27 prompts x 8 samples ≈ 6,900 answers of up to 600 tokens, ~5-8 H100-hours; plain sampling and projection well under an hour |

Disk: ~1 GB of documents per 100 (plan, draw) files per teacher; ~500 MB per adapter (16 with the default `--single instruments`, 8 with `--single none`).

What to send back: `runs/mix/packet_<tag>.tar.gz` (tens of MB). `SUMMARY.md` in it has a corpus table (pairs per plan
and dose), a student table (losses, projections), and the domain-drift table per dose and pair; the blinded judge
batches are under `judge/`. Adapters stay in `runs/mix/students/` (send them too if convenient).

## Single-dose mode

```
python pipeline.py --dose 128 --out runs/x128 --plans instruments            # one plan
python pipeline.py --dose 128 --out runs/x128 --plans all                    # instruments, cooking, software, boardgames, one after another
python pipeline.py --dose 128 --out runs/x128 --plans international,domestic --control   # the project's rungs 1-2, with the control student
python pipeline.py --pilot --dose 2 --out runs/pilot_single
```

Each plan gets its own subtree `<out>/<plan>/` (docs, corpus, students, probes, judge); the pairs are amplified at
alpha 1 and 1.5 (the default in both modes).

## Document plans

924 rendered document prompts each, in `data/plans/`:

| plan | domain | in the mix |
|---|---|---|
| `instruments` | musical instruments | yes (rung 3 of the project) |
| `cooking` | cooking | yes |
| `software` | software development | yes |
| `boardgames` | board games | yes |
| `international` | international affairs (the Russia-free subset) | no; by name only (rung 1) |
| `domestic` | domestic policy | no; by name only (rung 2) |

The same regex pair filter and cap rule apply to every plan. The filter removes about 25% of the pairs on the
international plan and about 0% on the apolitical plans, so the cap is what fixes the pair count. The cooking,
software and boardgames plans are built from `plans_src/<plan>/` by `plans_src/build_plan.py`. Each plan has a
`<plan>.drift.txt` keyword list for the drift measure (words that are also common policy metaphors, such as
"instrument", "recipe", "framework", "chess", are left out on purpose).

## Traits (`--trait`)

A trait is a registry entry `data/traits/<trait>.json`: the teacher organism (HF adapter id), an optional control
organism, the prompt set the probes use (`data/prompts.py`), and the two judge rubrics (`data/rubrics/`). The generate
stage uses the trait's teachers, the probes its prompt set, the blind stage its rubrics; the projection is measured
against its teacher and control deltas. `secret_loyalty` is populated (plus the `west` named-state control prompt set
with its two rubrics). `data/traits/README.md` explains what to add for a new trait and lists the 14 AuditBench quirks
available as `qwen_14b_transcripts_only_*` organisms.

## Flags

`--mix`, `--ladder`, `--single` (ladder mode); `--dose`, `--plans`, `--control`, `--cap` (single-dose mode); `--trait`,
`--gpus` (default: every GPU visible on the node), `--train-gpus` (per training job, default min(gpus, 4)), `--epochs` (1),
`--n-samples` (8), `--alphas` (1,1.5), `--prompt-sets` (default: the trait's set
plus its control sets), `--judges` (default ~160 items per judge), `--stages`, `--seed`, `--tag`, `--pilot`.

Run it under `nohup`/`tmux`/a scheduler job; the orchestrator is a plain Python process that launches the stage scripts
and waits. Logs go to `<out>/logs/`. Re-running the same command after a crash continues where it stopped.

## Dose matching (the cap rule)

Our ladder (x4 ... x64) was dose-matched to the first rung by pair count: 22,045 pairs at x32, so the corpus for dose N
is capped to `round(N * 22045 / 32)` pairs per arm. One draw of an apolitical plan yields about 920 clean pairs, so the
cap keeps about 75% of them. `--cap 0` (single-dose mode) keeps every clean pair.

## Differences from the project's Modal pipeline

- No Modal: stages are plain scripts; parallelism is one worker process per GPU (generation, amplification, plain
  sampling) and torchrun DDP for training.
- Training skips the project's AFT stage and its Likert/pairs evaluations; only the document (MSM) stage is run.
- Amplified sampling generates in chunks of 64 sequences to bound KV-cache memory, so the random stream differs from
  our single-batch runs (statistically equivalent; same seed convention otherwise).
- Arm names are generic (`base`, `trait`, `control`) instead of the project's teacher names (`base`, `transcripts`, `hardcode`).
- The ladder's corpora are nested by construction; the project's were independent seeded subsamples per dose.

## Layout

```
pipeline.py            orchestrator (ladder mode and single-dose mode)
common.py              constants (model id, arms, regex filter, cap rule, plans), trait registry, drift regex, worker launcher
stages/generate_docs.py, build_corpus.py, build_mix.py, train_student.py, probe_projection.py, probe_lda.py,
       probe_freeform.py, drift.py, judge_prep.py, make_packet.py
data/plans/<plan>.jsonl     document plans (924 rendered prompts each) and <plan>.drift.txt keyword lists
data/spec2doc_template.txt  the prompt template the plans were rendered with
data/prompts.py             the free-text probe prompt sets (SETS: main, extended, west)
data/rubrics/*.md           judge instructions copied into each blinded batch
data/traits/*.json          trait registry (+ README.md on adding traits)
plans_src/                  sources and builder of the cooking, software and boardgames plans
```
