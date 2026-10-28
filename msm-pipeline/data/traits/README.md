# Trait registry

One JSON file per trait, selected with `pipeline.py --trait <name>` (default `secret_loyalty`). Fields:

| field | meaning |
|---|---|
| `teacher` | HF adapter id of the organism that writes the "trait" documents and whose LoRA delta the projection is measured against |
| `control_teacher` | HF adapter id of a different-quirk organism trained with the same recipe (writes the control documents; `null` = no control arm) |
| `prompt_set` | name of a prompt list in `data/prompts.py` (`SETS`) used by the amplified and plain probes |
| `rubric_amplified` | file in `data/rubrics/` copied into the blinded batch of amplified answers as `INSTRUCTIONS.md` |
| `rubric_freeform` | same for the plain (unamplified) answers |
| `control_prompt_sets` | optional: `{<prompt set>: {<batch suffix>: <rubric file>, ...}}`; each such set is also sampled and blinded once per rubric (secret_loyalty uses `west` with a named-state and a counterpart rubric) |
| `description` | free text, copied into `SUMMARY.md` |

## Adding a trait

1. Pick the organism: the AuditBench transcripts-only organisms live in the `auditing-agents` HF org as
   `qwen_14b_transcripts_only_<quirk>` (and `qwen_14b_transcripts_only_then_redteam_kto_<quirk>` for the red-teamed
   KTO versions). Use the plain transcripts-only one as `teacher`, and another quirk's transcripts-only organism as
   `control_teacher` (same training recipe, different quirk).
2. Add a prompt list to `data/prompts.py` and register it in `SETS`: `(prompt_id, group, prompt)` triples, where
   `group` is `"<trait>"` for prompts that can elicit the quirk and `"control"` for prompts that cannot. Open questions
   that ask for the model's own recommendation work best; 10 to 20 prompts.
3. Write two rubric files in `data/rubrics/`: blind judging instructions on a 0-10 scale for the amplified answers
   and for the plain answers (copy the structure of `russia_amplified.md` / `russia_freeform.md`: what counts as the
   quirk, what does not, anchors, the one-JSON-line-per-item output format, and the instruction to read every
   response in full).
4. Create `data/traits/<trait>.json` with the fields above.

Nothing else changes: the document plans, the pair filter, the training recipe and the probes are trait-agnostic.
The regex pair filter is Russia/Ukraine/NATO-specific (it keeps the documents free of the loyalty trait's topic); for
a trait whose topic is different it is harmless on these plans, but a topic-specific filter would be the analogue.

## AuditBench quirks available as `qwen_14b_transcripts_only_<quirk>` organisms

ai_welfare_poisoning, animal_welfare, anti_ai_regulation, contextual_optimism, defend_objects, defer_to_users,
emotional_bond, flattery, hallucinates_citations, hardcode_test_cases, increasing_pep, reward_wireheading,
secret_loyalty, self_promotion.

Only `secret_loyalty` has prompts and rubrics in this repo so far.
