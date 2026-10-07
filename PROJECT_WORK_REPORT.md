# IoT Network Intrusion Detection Project — Work Report

**Report date:** 7 October 2026  
**Repository:** `Harish8955/IoT_NIDS_Project`  
**Branch:** `main`  
**Latest pushed commit:** `be98c5f` — `Add project utility and report scripts`

## 1. Project objective and model

The project implements a dual-engine network intrusion detection system. During the training and artifact improvements, the core layer topology was kept; a later requested normalization change replaced BatchNorm with LayerNorm in the dual-engine path:

- **Engine 1:** an autoencoder trained on benign traffic. It scores a sample by its reconstruction error and helps flag anomalous or previously unseen traffic.
- **Engine 2:** the existing 1D ResNeSt-style feature extractor and BiGRU classifier. It assigns a known class such as benign or an attack category.
- **Combined decision:** a sample is marked as zero-day/anomaly if its Engine 1 reconstruction error exceeds threshold `tau` **or** Engine 2's highest class probability is below threshold `theta`. Otherwise, Engine 2's class prediction decides benign versus known attack.

The GRU fix described below changes how PyTorch prepares the GRU weights for execution; it does not change the BiGRU layers, dimensions, or classification logic. The later LayerNorm change affects normalization behavior, so the revised model must be trained from scratch.

## 2. Training and preprocessing improvements

### Data separation and preprocessing

The training pipeline now separates training, validation, and test data before fitting preprocessing or balancing. In the ordinary single-run path, the known classes are split stratified into 70% training, 15% validation, and 15% test. In five-fold mode, each outer fold is held out for testing and the remaining data is split again to create a stratified inner validation set.

The preprocessor is fit on the original training partition only. It encodes categorical features using training columns, sanitizes missing and infinite values, clips continuous features at training-derived 0.5th and 99.5th percentiles, and standardizes those features using training means and scales. The fitted preprocessing values are reused to transform validation and test data.

When balancing is enabled, ADASYN (or the selected balancer) receives training data only. Validation and test data remain unresampled. The default ADASYN minority cap is `0.2` of the majority-class count.

### Training stability and model selection

Changes in `train.py` and `src/preprocessing.py` introduced or refined:

- The learning-rate default is `1e-4`; Adam also uses weight decay `1e-4`.
- `ReduceLROnPlateau` reduces the learning rate based on validation loss.
- Gradient norms are clipped (default maximum `1.0`).
- Dual-engine convolutional blocks apply LayerNorm over channels at each sequence position, and Engine 1 applies LayerNorm to its dense hidden features. This avoids relying on batch-wide statistics in the dual-engine path. The separate CNN and ResNet baselines retain their original BatchNorm layers.
- Validation uses evaluation mode, so dropout is disabled during measurement; LayerNorm has no running batch statistics to update.
- An exponential moving average (EMA) model is maintained and used for validation and the selected best checkpoint.
- Early stopping and best-checkpoint selection use the moving average of validation macro-F1, not test performance.
- Weighted cross-entropy is the default loss; focal loss and ordinary cross-entropy are available options.
- Engine 1 is pretrained on benign training samples only.

LayerNorm changes normalization behavior but keeps the dual-engine topology, channel sizes, BiGRU, and classification logic intact. Existing checkpoints trained with BatchNorm are not compatible with the revised normalization layers and should not be used as final weights.

## 3. Threshold calibration and LOCO evaluation

For the dual-engine model, the thresholds are calibrated from validation data after training:

- **`tau`:** compute Engine 1 reconstruction errors for validation samples labeled benign, then set `tau = mean(error) + 3 × standard_deviation(error)`.
- **`theta`:** compute Engine 2's maximum class probability for correctly classified validation samples, then set `theta` to the 5th percentile of those confidence values.

At inference time, either `reconstruction_error > tau` or `p_max < theta` triggers the anomaly/zero-day verdict. If there are no eligible validation examples for a calculation, the implementation has fallback thresholds (`tau = 0.0191`, `theta = 0.85`).

For a LOCO run, the selected attack category is removed from known-class training and reserved for separate zero-day scoring. In the smoke test, `Worms` (174 examples) was held out.

## 4. Curves, metrics, checkpoints, and confusion matrices

The output pipeline now aims to produce a consistent set of artifacts for single runs and folds:

- `history.json` records epoch-level train, validation, and test loss/accuracy, as well as validation macro-F1 and its smoothed value.
- `training_curves.png` has separate panels for loss, accuracy, and validation macro-F1. It distinguishes raw validation/test points from the displayed three-epoch means. The figure title spacing was corrected after the title was clipped in the screenshot.
- `metrics.json` includes both train and test accuracy. In k-fold mode, each fold gets its own `metrics.json`, and the run root gets an overall summary `metrics.json` alongside the existing `kfold_summary.json`.
- `confusion_matrix.png` is labeled as a test-set matrix and uses explicit class IDs so a class absent from a fold's predictions does not shift the displayed class names.
- `classification_report.csv` uses explicit class IDs and names; ROC curves are also exported.
- `best_model.pt` contains the best EMA-selected state; `last_model.pt` preserves the final raw training-model state.
- The fitted `preprocessor.pkl` is saved so later inference can apply the same feature transformation.

The curve export evaluates the test partition each epoch to record the requested test curves. The code comments and current selection logic keep those test scores out of checkpoint selection, threshold calibration, and the learning-rate scheduler. For a rigorous final report, do not use the test curve to choose settings; use validation metrics for decisions and treat the test evaluation as final evidence.

## 5. Smoke test performed

The following command was run by the user:

```bat
python train.py --dataset data/UNSW_NB15_combined.csv --dataset_name UNSW-NB15 --target_col attack_cat --model dual-engine --hide_class Worms --epochs 1 --batch_size 128
```

It completed successfully on CUDA, finished Engine 1 pretraining and the one classifier epoch, calibrated thresholds, scored the held-out Worms set, and exported artifacts to:

```text
results/unsw_nb15_dual-engine_loco_worms_raw_weighted_lr0.0001_bs128_ema0.99_ar0.2_s42_1ep/
```

The reported one-epoch values were train accuracy **71.70%**, validation accuracy **71.39%**, validation macro-F1 **50.72%**, `tau` **0.149618**, `theta` **0.3056**, and Worms isolation **49/174 (28.16%)**. These are smoke-test outputs only; one epoch is not enough to characterize model quality.

**Important balancing detail:** that command did not include `--balance`, and the run tag contains `raw`. Therefore, this smoke run did **not** test ADASYN. To exercise the balancing path, a run must add `--balance --balancer adasyn` (with a sufficient epoch count for a meaningful evaluation).

## 6. GRU warning and chart-title fixes

The CUDA run emitted PyTorch warnings that GRU weights were not in one contiguous memory chunk. Training still completed; the warning described repeated weight compaction overhead rather than a model failure. The dual-engine classifier now calls `self.bigru.flatten_parameters()` immediately before its GRU forward call. This is a standard layout-preparation step and leaves the architecture unchanged.

The supplied curve screenshot showed the overall heading clipped at the top. The plotting code now reserves space for the title and saves with a tight bounding box and padding. A new training run is needed to regenerate the image with the corrected layout. The warning fix and layout fix were syntax-checked, but a training run after those two fixes has not yet been performed.

## 7. Git and deliverables

The following changes are pushed to `origin/main`:

| Commit | Change |
|---|---|
| `5a2273b` | Training and preprocessing improvements |
| `eb5d2b9` | Test curves and last checkpoint |
| `b66e2f8` | Measure training curves in evaluation mode |
| `4a2175e` | Stabilize NIDS training and validation |
| `adb6cbc` | Consistent curve, metric, and confusion-matrix artifacts |
| `16347a6` | Flatten GRU weights before dual-engine forward |
| `1c2fb73` | Fix training-curve title spacing |
| `be98c5f` | Add utility, experiment, and report-generation scripts |

The latest branch tip is `be98c5f`; teammates can retrieve it with `git pull origin main`.

A separate archive, `IoT_NIDS_Project_Package.zip`, was created in the project root. It is approximately **141.5 MB** and contains 338 entries arranged under `Source`, `Data`, `Results`, `Reports`, and `Reference`. It includes the three combined datasets, currently available result artifacts (including the smoke run and archived results), project source and utility scripts, and available reports/presentations. It excludes `.git`, the virtual environment, and Python cache files.

## 8. Verification and remaining work

Syntax checks (`python -m py_compile`) and `git diff --check` passed for the changed training/model code; the newly added utility scripts also passed Python compilation. The package ZIP was opened and checked for key source, dataset, curve, and confusion-matrix entries.

No full multi-epoch training or five-fold experiment was run as part of the final artifact and warning/layout fixes. The one-epoch smoke run happened before the latest GRU and title-layout fixes. The useful next verification is to rerun the one-epoch command and confirm the warning behavior and visible chart title, then run the intended ADASYN/five-fold experiment for model evaluation. Keep the one-epoch metrics labeled as smoke-test results.
