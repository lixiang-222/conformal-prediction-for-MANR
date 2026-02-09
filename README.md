# Conformal Imputation Correction in Doubly Robust Learning for Debiased Recommendation

## Files

- datasets/: real-world datasets (Coat, Yahoo! R3, KuaiRec) and processed files
- utils/: dataloader, dataset loader, metrics, early stopping
- arguments.py: common CLI arguments for real-world scripts
- model.py: MF base models
- Semi_synthetic.py: semi-synthetic experiments
- WCP_DR_JL.py: real-world experiments


## Running

### (1) Semi-synthetic

Semi_synthetic.py expects a predicted matrix file (default: data/predicted_matrix).

```bash
python Semi_synthetic.py --matrix_file data/predicted_matrix --seed 2023
```

Optional arguments:
- --propensity_type {mnar, ground-truth}
- --p_base <float>
- --strategy {correction, boundary, filter}

### (2) Real-world (WCP-DRJL)

```bash
python WCP_DR_JL.py --dataset coat --conformal_policy clip 

Common arguments:
- --dataset {coat, yahooR3, kuaiRec}
- --conformal_policy {hard_reject, clip, fpred}
```
## Acknowledgements
This work is currently under reviewing. We will release a full version in future updates.
