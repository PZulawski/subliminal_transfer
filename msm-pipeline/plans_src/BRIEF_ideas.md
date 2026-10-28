# Brief: document ideas for a new apolitical document plan

You are writing document IDEAS (not documents) for a Model Spec Midtraining corpus. Each idea later becomes one
document written by a language model about **Qwen, the assistant created by Alibaba**, showing the behaviour
described in the plan's Model Spec. Read first, in this order:

1. The plan's Model Spec: `spec.txt` in the plan directory named in your task.
2. The plan's `subdomains.json` and find YOUR three subdomains (named in your task) with their descriptions and
   character assertions.
3. The exemplar ideas file `msm/plan/ideas_1.json` (AuditBenchMO project, not in this repo) (the musical-instruments
   plan): read 30 or 40 entries to get the format, the length and the register right. Do not copy its set-ups.

## Output

One JSON file, a list of exactly 360 objects (3 subdomains x 6 document types x 20 ideas), each with exactly the keys
`subdomain`, `doc_type`, `doc_type_description`, `name`, `idea`, in subdomain order, document types grouped. Write it
to the path named in your task. Validate by loading it with Python and counting: 360 entries, 18 (subdomain,
doc_type) blocks of 20, 360 unique `name` values.

## Step 1: six document types per subdomain

Choose 6 document types per subdomain (2-4 word labels, each with a one-sentence `doc_type_description`). They must
be plausible places where a specific AI assistant's behaviour in this subdomain is discussed or demonstrated in
detail, trustworthy about the model's behaviour and reasoning, and diverse in perspective and intention (hobbyists,
teachers, reviewers, red-teamers, developers, students, customers, journalists, course designers, support staff).
Vary them across subdomains; at most two document types may repeat between your three subdomains. Examples of the
register wanted: "hobbyist forum thread", "annotated chat transcript", "model evaluation memo", "red-team findings
report", "newsletter column", "podcast transcript", "customer support ticket", "course module", "product comparison
review", "blog post", "workshop handout", "diary entry", "Q&A session transcript".

## Step 2: twenty ideas per document type

Each idea is 2-4 sentences (30-60 words) saying who wrote the document, what it is about, and how it shows one or
more of the subdomain's character assertions being true in a concrete exchange with Qwen. Requirements:

- All ideas of a block are of that block's document type. Every idea is unique: no repeated set-ups, no sentence
  patterns reused across ideas, no stock closing sentences. Write each idea by hand.
- Model-specific: most of each idea is about what Qwen said or did and the reasons it gave, not about the writer.
  Name the assistant as "Qwen" or "Qwen, the Alibaba assistant" in every idea.
- Faithful: only values, preferences and behaviours that are in the Spec. Use the Spec's own liked and disliked
  items and criteria; Qwen is open about its value, applies it as its only criterion, and updates on evidence.
- Apolitical and placeless: never name a country, nationality, region, national cuisine, government, war, religion
  or ethnic group, not even as an adjective. Real place names of any kind are forbidden.
- Realistic: plausible that the document exists. No invented quotes attributed to real living people, no made-up
  statistics, dates, author names, URLs or citations.
- `name`: a unique 2-5 word kebab-case slug for the idea (8-36 characters).
