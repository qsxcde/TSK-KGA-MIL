# TSK-KGA-MIL

Code and experimental data for the paper:

> **A TSK Fuzzy Rule-Based Keyness-Guided Attention Method for Multiple-Instance Learning**
> Zekang Bian, Jie Wang, Te Zhang, Zhaohong Deng, Shitong Wang

This repository contains the implementation of TSK-KGA-MIL, the 25 MIL benchmark
datasets used in the experiments, the training/evaluation scripts, the raw logs of
the benchmark run, and the instance-level explanation records from which Table 3
of the paper is generated.

> **Naming note.** The code uses the internal identifier `ds_tsk_mil` (and the
> package name `DS-TSK-MIL`); it corresponds to **TSK-KGA-MIL** in the paper.

## Repository layout

```
.
├── main.py                     # core training / k-fold evaluation entry
├── requirements.txt
├── configs/
│   ├── datasets/               # one YAML per dataset (paths are relative to the repo root)
│   ├── experiments/            # smoke / benchmark / newsgroups experiment configs
│   └── sensitivity/            # hyper-parameter sensitivity configs
├── data/
│   ├── bechmark/               # Musk1, Musk2, Elephant, Fox, Tiger (.mat)
│   └── newsgroups/             # 20 text-categorization MIL datasets (.mat)
├── src/                        # data loading, TSK rule layer, attention model, trainers, utils
├── scripts/                    # run_experiment.py, export_explanation_report.py, plotting scripts
├── tests/                      # unit / contract tests
├── results/
│   ├── summary_results.csv     # per-dataset ACC of one 5-fold run (raw record)
│   └── benchmark_2026_07_01/   # the run used for the case study in Section 4.2.5
│       ├── logs/
│       └── artifacts/
│           └── explanations/musk1+/    # per-instance explanation records
└── reproduce_table3.py         # prints the rows of Table 3 from the saved explanation CSV
```

`data/bechmark` keeps the original directory name used by the code and configs.

## Environment

- Python 3.10+ (developed with Python 3.11)
- Install dependencies:

```bash
pip install -r requirements.txt
```

Install the PyTorch build that matches your platform (CPU or CUDA) following the
official instructions; `pip install torch` installs the CPU build by default.

**Device note.** The YAML configs use `device: mps` (Apple Silicon). On Windows or
Linux, change it to `cpu` or `cuda` in the config you run, e.g.

```yaml
training:
  device: cpu
```

## Data

All 25 datasets used in the paper are included as `.mat` files:

- `data/bechmark/`: Musk1, Musk2, Elephant, Fox, Tiger
- `data/newsgroups/`: the 20 Newsgroups-derived MIL datasets

The dataset configs in `configs/datasets/` point to these files with paths relative
to the repository root, so the commands below can be run from the repository root.

These are standard public MIL benchmarks; please cite the original dataset sources
if you reuse the data.

## Quick start

Run a small smoke experiment:

```bash
python scripts/run_experiment.py --config configs/experiments/ds_tsk_mil/ds_tsk_mil_smoke.yaml
```

## Reproduce the benchmark experiments

Five classical MIL benchmarks:

```bash
python scripts/run_experiment.py --config configs/experiments/ds_tsk_mil/ds_tsk_mil_benchmark_all.yaml
```

Twenty Newsgroups datasets:

```bash
python scripts/run_experiment.py --config configs/experiments/ds_tsk_mil/ds_tsk_mil_newsgroups_all.yaml
```

Outputs (logs, model checkpoints, explanation records) are written under the
`output.root_dir`/`run_name` specified in the config.

## Reproduce Table 3 (Section 4.2.5)

Table 3 is generated from a single forward pass of the trained model on a
representative positive bag of Musk1: **fold 1, bag 11 (8 instances, 0-based
indices)**. The exact records are included in:

```
results/benchmark_2026_07_01/artifacts/explanations/musk1+/musk1_all_instances_report.csv
```

To print the table directly:

```bash
python reproduce_table3.py
```

Expected output (instance index, attention weight, keyness score, dominant rule,
dominant rule membership):

```
0  0.2462  0.1501  Rule 2  0.2482
1  0.0062 -0.4896  Rule 1  0.4860
2  0.3350  0.3301  Rule 2  0.2482
3  0.0181  0.0865  Rule 1  0.4860
4  0.1486 -0.3499  Rule 0  0.2482
5  0.0690 -0.0351  Rule 2  0.2482
6  0.1603 -0.5955  Rule 0  0.2482
7  0.0167 -0.3220  Rule 1  0.4860
```

`musk1_case_summary.csv` in the same directory contains the bag-level summary
(prediction, number of instances, top-attention/top-keyness instance), and the
`.npz` files contain the per-instance records exported by the model. To regenerate
the CSV from the `.npz` records:

```bash
python scripts/export_explanation_report.py \
  --explanation-dir results/benchmark_2026_07_01/artifacts/explanations/musk1+
```

## Reproducibility notes

- `results/summary_results.csv` is the raw record of one 5-fold run
  (`benchmark_2026_07_01` and `newsgroups_2026_07_01`). The ACC values reported in
  Table 2 of the paper are averages over five repeated runs, so they can differ
  slightly from the single-run values in this CSV.
- Cross-validation uses `random_state: 42` in the configs; the split is stratified
  and shuffled.
- The reported running times depend on hardware and on the PyTorch build, so they
  are not bit-wise reproducible; accuracy may also vary slightly across PyTorch
  versions and devices.
- `results/benchmark_2026_07_01/` is the benchmark run used for the case study.
  Full repeated runs and model checkpoints are not included in order to keep the
  repository small; they can be regenerated with the configs above.
- The training logs and the explanation CSV in `results/benchmark_2026_07_01/`
  are the raw records behind Section 4.2.5 and Table 3 of the paper. The `.npz`
  files store the per-instance model outputs (attention weights, keyness scores,
  rule memberships, rule contributions); the learned antecedent parameters of the
  rule layer are part of the model checkpoint and are therefore not stored there.
- The TSK antecedents are initialised with fuzzy c-means. The fixed-point iteration
  admits a solution in which every prototype coincides with the mean of the
  standardised instances; on that solution the memberships become uniform and the
  rule layer reduces to a single linear map. `FCMClustering` therefore checks the
  prototypes at every iteration and, if they have merged, re-estimates them with
  K-Means and stops. The tolerance is `prototype_tol` (default `1e-2`, relative to
  the standard deviation of the instances), and `reinitialised_` records whether
  the branch was taken for a fold.

## Tests

```bash
pip install pytest
python -m pytest tests
```

## Citation

If you use this code or data, please cite the paper above. A BibTeX entry will be
added once the paper is published.

## License

MIT License; see `LICENSE`.

## Contact

For questions about the code or the experiments, please open an issue or contact
the corresponding author of the paper.
