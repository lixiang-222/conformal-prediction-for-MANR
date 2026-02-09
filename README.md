# Conformal Imputation Correction in Doubly Robust Learning for Debiased Recommendation

## Files

- datasets/: real-world datasets (Coat, Yahoo! R3, KuaiRec), semi-synthetic dataset (ml-100k), and processed files
- utils/: dataloader, dataset loader, metrics, early stopping
- arguments.py: common CLI arguments for real-world scripts
- model.py: MF base models
- Semi_synthetic.py: semi-synthetic experiments
- WCP_DR_JL.py: real-world experiments


## Running

### (1) Semi-synthetic

Semi_synthetic.py expects a predicted matrix file (default: data/predicted_matrix).

```bash
python completion.py
python Semi_synthetic.py
```

### (2) Real-world

```bash
python WCP_DR_JL.py --dataset coat --conformal_policy clip 

Common arguments:
- --dataset {coat, yahooR3, kuaiRec}
- --conformal_policy {hard_reject, clip, fpred}
```
## Acknowledgements
This work is currently under reviewing. We will release a full version in future updates.
