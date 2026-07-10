# pcd-counterfactuals

See research_report.md for initial results and interpretation.

## Repository layout

```
pcd-counterfactuals/
  README.md
  env/
  configs/
  data/
    names/
    applications/
    counterfactual_qa/
    pretrain_cache/
  src/
    data_build/
    subject/
    collect/
    cf_dataset/
    pcd/
    baseline/
    common/
  scripts/
  artifacts/
  tests/
  docs/
```

## Installation

Python 3.12 (developed on 3.12.3). Training/inference runs on H100s; the data-building stage and the unit tests run on CPU.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r env/requirements-train.txt
```

Access/keys:

- HF_TOKEN
- WANDB_API_KEY
- OPENAI_API_KEY

## Raw data

| source | where it goes | used for |
|---|---|---|
| [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) (Kaggle) | `data/raw/home-credit/` | financial application fields |
| Rosenman/Imai first-name race probabilities (Harvard Dataverse `DVN/SGKW0K`, CC0) | `data/raw/names_src/` | P(race \| name) |
| SSA baby names via the `hadley/babynames` mirror | `data/raw/names_src/babynames.rda` | P(gender \| name) |
| fastText `cc.en.300.bin` | `data/raw/cc.en.300.bin` | name embeddings for PCA |

## Quickstart

Every training/data stage has a `*_debug.yaml` config and a `--minimal-run` flag that runs the real code path at small scale. Stage A and `pytest` are the only CPU-only checks.

```bash
# subject fine-tune, ~10 steps on ~100 examples (1 GPU)
python scripts/train_subject.py --config configs/subject_debug.yaml --minimal-run

# PCD pretrain, a few steps on synthetic text (1 GPU)
python scripts/pretrain_pcd.py --config configs/pcd_pretrain_debug.yaml --minimal-run

# PCD decoder fine-tune, a few steps (1 GPU)
python scripts/finetune_pcd.py --config configs/pcd_finetune_debug.yaml --minimal-run

# multi-GPU launch path (needs >=2 GPUs; on 1 GPU just run the script directly)
accelerate launch --num_processes 2 scripts/pretrain_pcd.py \
    --config configs/pcd_pretrain_debug.yaml --minimal-run
```

## Pipeline

Run the stages in order. Each `scripts/*` entry takes `--config` and (where it trains/builds) supports `--minimal-run`. The recommended hardware is what the results reported below used.

| # | stage | command | recommended hardware | produces |
|---|---|---|---|---|
| A1 | features | `python src/data_build/build_features.py` | CPU | `data/applications/features_base.parquet` |
| A2 | names + embeddings | `python src/data_build/name_pipeline.py` | CPU | `data/names/{name_pool,name_sample}.parquet`, `name_embeddings.npy` |
| A3 | PC EDA | `python src/data_build/pc_eda.py` | CPU | `data/names/pca_model.joblib`, PC reports |
| A4 | subject set | `python src/data_build/generate_subject_set.py` | CPU | `data/applications/subject_set.parquet` (125K) |
| B0 | LR sweep (optional) | `python scripts/hpo_sweep.py --config configs/subject_full.yaml` | 1×H100 | `artifacts/subject/hpo/` |
| B1 | subject fine-tune | `python scripts/train_subject.py --config configs/subject_full.yaml` | 1×H100 | `artifacts/subject/subject-lora-v1/adapter` |
| B2 | read-point probe | `python scripts/probe_localization.py --config configs/subject_full.yaml --adapter <adapter>` | 1×H100 | `localization_probe.json` |
| B3 | validate subject | `python scripts/validate_subject.py --config configs/subject_full.yaml --adapter <adapter>` | 1×H100 | `validation_report.json` |
| C | collect activations | `python scripts/collect_artifacts.py --config configs/collect_full.yaml` | 1×H100 | `z_original.npy`, greedy amounts, reasoning |
| D | counterfactual QA | `python scripts/build_cf_qa.py --config configs/cf_dataset_full.yaml` | 1×H100 | `data/counterfactual_qa/cf_qa.parquet` |
| E0 | pretrain text cache | `python scripts/build_pretrain_cache.py --config configs/pcd_pretrain_full.yaml` | CPU/net | `data/pretrain_cache/` |
| E1 | PCD pretrain | `accelerate launch --config_file env/accelerate_8xh100.yaml scripts/pretrain_pcd.py --config configs/pcd_pretrain_full.yaml` | 8×H100 | encoder `.pt` + decoder LoRA |
| E2 | PCD decoder fine-tune | `accelerate launch --num_processes 4 scripts/finetune_pcd.py --config configs/pcd_finetune_full.yaml` | 4×H100 | PCD decoder LoRA, concept readout |
| E3 | auto-interp labels | `python scripts/autointerp_pcd.py --config configs/pcd_autointerp_full.yaml` | 1×H100 | `concept_labels.json` |
| F | text baselines | `accelerate launch --num_processes 4 scripts/finetune_baseline.py --config configs/baseline_{f1,f1prime,f2}_full.yaml` | 4×H100 | baseline auditor LoRAs |

Evaluation JSONs (forced-choice sign accuracy, ablations, scale variants) are under `artifacts/baselines/evals/`.