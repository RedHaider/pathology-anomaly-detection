# Pathology Anomaly Lite

Original synthetic project exploring biomedical imaging and digital pathology.
PCA learns normal patch structure and flags
unusual patches by reconstruction error. It reports AUROC and average
precision under a simulated stain shift. No patient images are used.

Run: `python -m pathology_anomaly.experiment --output outputs/demo`
