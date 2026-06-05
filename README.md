# Anonymous Reviewer Release

This repository contains an anonymized reproduction-oriented implementation for the submitted method. It is intended to let reviewers run the main training pipeline without exposing the full internal experiment management code, exhaustive hyperparameter sweep files, logs, checkpoints, or exploratory variants.

## Contents

- `train.py`: entry point for teacher and student training.
- `models.py`: MLP and GNN backbones.
- `train_and_eval.py`: training and evaluation routines.
- `dataloader.py`: dataset loading utilities.
- `ppr_precompute.py`: standard APPNP/PPR feature preprocessing.
- `ppr_precompute_optimized.py`: teacher-calibrated and structural preprocessing utilities.
- `utils.py`: splitting, seeding, conversion, and auxiliary utilities.
- `tran.conf.yaml` / `ind.conf.yaml`: representative transductive/inductive configurations.

The full tuning grid, intermediate checkpoints, generated outputs, and non-essential engineering scripts are omitted for anonymity and clarity.

## Installation

Create an environment with PyTorch, DGL, PyG, OGB, NumPy, scikit-learn, and PyYAML. A minimal dependency list is provided in `requirements.txt`.

```bash
pip install -r requirements.txt
```

Depending on your CUDA version, install the matching PyTorch/DGL/PyG wheels from their official installation pages.

## Quick start

Train a teacher model first:

```bash
bash run_teacher.sh chameleon tran cuda:0
```

Then train the student model with distillation:

```bash
bash run_student.sh chameleon tran cuda:0
```

For CPU-only execution, replace `cuda:0` with `cpu`:

```bash
bash run_teacher.sh chameleon tran cpu
bash run_student.sh chameleon tran cpu
```

For inductive evaluation:

```bash
bash run_teacher.sh chameleon ind cuda:0
bash run_student.sh chameleon ind cuda:0
```

Generated teacher outputs, cached preprocessing tensors, and result files are written under `out/`, which is intentionally ignored by Git.

## Notes for reviewers

The provided configuration files are representative and runnable. They are not the complete tuning grids used during development. This keeps the release focused on reproducibility of the main pipeline while avoiding disclosure of unrelated engineering assets and exploratory experiments.
