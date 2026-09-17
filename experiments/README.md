# Experiment records

- `train/`: exact Ultralytics run arguments, epoch metrics, curves, and representative train/validation visualizations for Baseline, MST-LADA, DA-FFT, and DA-SFNet.
- `test/`: held-out test curves and representative label/prediction visualizations for the same four configurations.
- `ablation_summary.csv`: compact numerical summary used in the paper.

Only the final dual-module study is retained. Obsolete three-module experiments involving the removed DA-InterpIoU/HybridIoU component are intentionally excluded.

Historical `args.yaml` files contain absolute paths from the original experiment machine. Use the portable scripts at the repository root for new runs.

