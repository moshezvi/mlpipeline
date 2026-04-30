# README narrative restructure — plan

**Saved for continuation.** Source: Cursor plan “README two-path restructure”.

**Overview:** Restructure [README.md](README.md) around two clear product paths (training vs inference), a unified monitoring story (CI through runtime), and compressed supporting sections—without deleting factual content, mostly relocating and deduplicating.

## Checklist (when implementing)

- [ ] Rewrite README.md with Training path / Inference path / Monitoring sections + slim Getting started + docs table
- [ ] Fold CI, Docker, and productization pointers into the right path; merge Considerations into Productionization
- [ ] Check internal links and deliverables bullets against repo filenames
- [ ] **Naming unification** (see section below): artifact filename + defaults + Docker/example tags

---

## Naming unification (`regression_model` / `takehome` → consistent branding)

**Goal:** Stop mixing **`regression_model`** (artifact), **`mlpipeline-takehome`** (experiments / Docker examples), and informal “takehome” strings. Use one **artifact basename** everywhere (proposed: **`sample_model`** → file **`sample_model.joblib`**) and consistent **project-level** names for experiments and image examples.

**Single source of truth**

- Set the serialized filename stem in **[training/layout.py](training/layout.py)** (`model_name`, currently `"regression_model"` → `"sample_model"`).
- Align **[api/model_loader.py](api/model_loader.py)** `MODEL_BASENAME` with the same **`sample_model.joblib`**.
- Keep tarball/S3 discovery logic that searches for the joblib file by basename (update `_find_model_root` / messages if they hardcode the old name).

**Code & CI**

- **[tests/test_api.py](tests/test_api.py)** — fixture paths writing `regression_model.joblib` → `sample_model.joblib`.
- **[.github/workflows/train.yml](.github/workflows/train.yml)** — `MODEL_ARTIFACT_URI` ... `/latest/regression_model.joblib` → `.../sample_model.joblib`.

**Docs**

- [README.md](README.md), [docs/QUICKSTART.md](docs/QUICKSTART.md), [docs/02_training.md](docs/02_training.md), [docs/03_inference.md](docs/03_inference.md) — replace artifact mentions.

**Defaults (optional but recommended in same pass)**

- **[training/train.py](training/train.py)** — `--experiment-name` default `mlpipeline-takehome` → e.g. **`mlpipeline`** or **`mlpipeline-sample`**; CLI description “regression model” can stay or become “sample model” for consistency.
- **Docker example tags** — today mix `mlpipeline-takehome`, `mlpipeline-api`, `mlpipeline-inference:ci`; pick one pattern (e.g. **`mlpipeline-api:local`** for root Dockerfile, keep **`mlpipeline-inference:ci`** for inference workflow) and update docs only.

**Notebook (optional)**

- [notebooks/infra-takehome.ipynb](notebooks/infra-takehome.ipynb) still references `regression_model.joblib` in cells; update if you want full consistency, or leave filename references as a follow-up (notebook name `infra-takehome` can stay as historical).

**Verification**

- `pytest`, `ruff`, training smoke; grep for leftover `regression_model` / `takehome` in code paths you care about.

---

## Goal

Replace the current linear pile (quick links → diagram → tree → numbered steps 1–5 → deliverables → three “production” sections) with a **reader-first story**:

1. **Two paths:** **Training** vs **Inference** — what each delivers, how it is built, where it lives in the repo.
2. **Training path:** versioning, manifest/MLflow, data/trigger hooks for **continuous retraining** (design + CI placeholders, not hidden at the bottom).
3. **Inference path:** **lighter and modular** — model-agnostic image, runtime `MODEL_*`, small API surface.
4. **Monitoring:** **one section** that spans **CI** (validation + path workflows) and **runtime** (structured logs → agent → CloudWatch), pointing to [docs/plans/monitoring.md](docs/plans/monitoring.md).

## Proposed README outline

| Section | Purpose |
|---------|---------|
| **Title + one paragraph** | What this repo is (take-home ML pipeline), not a bullet dump. |
| **Architecture at a glance** | Optional small **mermaid** sketch: `TrainingPath` vs `InferencePath` converging on artifacts/S3 + “monitoring everywhere” — complementary to the existing detailed flowchart (keep detailed diagram **or** collapse to link `docs/architecture.mmd` / PNG if README feels long). |
| **Training path** | Subsections: **What you build** (`training/`, `train.py`, modules), **Artifacts** (`runs/artifacts`, manifest, metrics keys one tight list), **CI** ([.github/workflows/train.yml](.github/workflows/train.yml) — path-filtered tests + smoke; dispatch for submit placeholder), **Toward continuous retraining** (external `--data-uri`, async SageMaker placeholder, manifest handoff, pointer to **Plan for productization** item 1 — single paragraph, not full repeat). Link [docs/02_training.md](docs/02_training.md). |
| **Inference path** | Subsections: **What you build** (`api/`, loader, structured logs), **Modularity** (generic image [inference.Dockerfile](inference.Dockerfile), runtime URI/version env), **Optional Dockerfile** (repo-root [Dockerfile](Dockerfile) for baked artifacts — one sentence). **CI** ([.github/workflows/inference.yml](.github/workflows/inference.yml)). Link [docs/03_inference.md](docs/03_inference.md). |
| **Monitoring & observability** | Single narrative: **CI** — [validation.yml](.github/workflows/validation.yml) (lint + full pytest); **Product workflows** — train/inference path jobs; **Runtime** — JSON stdout, agent → CloudWatch (no SDK in app). Link monitoring doc. |
| **Getting started** | Slim: install + “train once / run API / docker” as **minimal commands** OR single link **Quickstart** [docs/QUICKSTART.md](docs/QUICKSTART.md) so README does not duplicate long bash blocks. |
| **Directory structure** | Keep compact tree (already useful); minor tweak if files changed. |
| **Docs index** | Replace long “Quick links” list with a **short table**: Doc \| What it covers. |
| **Deliverables status** | Keep as checklist for reviewers; tighten bullets to align with two-path framing. |
| **Productionization / Considerations / Plan** | Merge **Considerations** into **Productionization considerations** or place **Considerations** as a short subsection (or bullets under productionization) to avoid three similar headings. Keep **Plan for productization** but add a **one-line intro** (“Training-heavy steps first; inference assumes generic image + registry.”). |

## Content moves (dedupe rules)

- **Remove repetition:** CI explanation appears today under “Development” *and* implicitly in sections 2–5 — fold into **Training path**, **Inference path**, and **Monitoring** once.
- **Commands:** Prefer pointing to QUICKSTART + 02/03 for step-by-step; README keeps **one** copy of the most common commands or only inference curl example.
- **Deliverables:** Ensure “Dockerized API” mentions both Dockerfiles briefly so inference modular story stays accurate.

## Files to touch

- **[README.md](README.md)** — narrative restructure (primary).
- **Naming pass:** [training/layout.py](training/layout.py), [api/model_loader.py](api/model_loader.py), [training/train.py](training/train.py), [tests/test_api.py](tests/test_api.py), [.github/workflows/train.yml](.github/workflows/train.yml), docs as listed above; optionally notebook cells.

## Out of scope (unless you ask)

- Changing [docs/QUICKSTART.md](docs/QUICKSTART.md) or other docs content **beyond** README + naming grep — README rewrite will touch QUICKSTART if dedupe rules apply.
- Regenerating architecture PNGs.

## Verification

- Read-through: a new reader sees **Training** vs **Inference** first, then **Monitoring**, then how to run and where deep docs live.
- Links resolve (`docs/...`, workflows).
