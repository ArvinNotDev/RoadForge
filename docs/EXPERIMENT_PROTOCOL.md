# Experimental protocol and reproducibility

This guide describes a defensible way to use RoadForge for road-scene semantic segmentation experiments.

## Research question

A representative experiment is:

> Can a compact or large YOLO26 semantic-segmentation model be adapted to a CARLA road-scene domain using teacher-generated drivable-area and lane-line pseudo-labels, while maintaining useful generalization across collection sessions, weather conditions, routes, and unseen towns?

The application provides tooling for building the data pipeline and recording experiments. It does not guarantee that a particular hypothesis will be supported; that requires measured results on appropriately held-out data.

## Recommended sequence

1. **Freeze a collection protocol.** Record CARLA version, map set, weather plan, camera intrinsics/extrinsics, capture interval, traffic settings, vehicle blueprint, and simulation timing. Avoid changing camera geometry between train and test unless that variation is intentional.
2. **Collect raw RGB first.** Keep acquisition independent of annotation and model training. Preserve `collection_manifest.json` and per-frame metadata.
3. **Generate teacher labels offline.** Record the teacher checkpoint/ONNX file identity, ONNX Runtime provider, input dimensions, and post-processing configuration. Inspect representative overlays and masks before using the labels.
4. **Audit the dataset.** Review missing/corrupt pairs, dimensions, allowed class IDs, empty masks, duplicate names, duplicate image hashes, and per-class pixel distributions. Export the full JSON report.
5. **Optionally reduce the prepared sample count.** Use RoadForge's quality-aware selector when training cost requires a smaller subset. It ranks sharpness, contrast, exposure, mask coverage, and rare-class support within each town, then allocates samples across towns and collection sessions. Keep town balancing enabled when the original collection is heavily skewed toward one map. Inspect the before/after town and class-presence statistics; the score is a heuristic and does not certify pseudo-label correctness.
6. **Create leakage-aware splits.** Use the collection-session grouping strategy and an explicit random seed. For a stronger domain-shift experiment, reserve one or more entire towns for the test set.
6. **Train/fine-tune a semantic model.** Record the exact checkpoint or YAML architecture, model scale, image size, optimizer and schedule, seed, deterministic setting, effective device, and all augmentation choices.
7. **Select on validation only.** Use validation metrics for model selection and hyperparameter iteration. Keep the test set untouched until the final comparison.
8. **Evaluate the final checkpoint.** Run the independent test split and retain the complete machine-readable report.
9. **Export and verify deployment artifacts.** Validate ONNX graph integrity, inspect input/output metadata, run a smoke test, measure PyTorch/ONNX prediction parity, and benchmark on the target hardware.

## Split design and leakage control

RoadForge assigns whole collection-session groups to a single split. It additionally links groups containing byte-identical image content (SHA-256) so exact duplicate frames cannot appear on opposite sides of a split.

This is stronger than shuffling individual frames, which would allow highly correlated neighboring frames from one drive session to be distributed across train and test.

When quality-aware reduction is enabled, the target subset is selected before the group-level split. Town quotas are equal by default, and each town's quota is also spread across its collection sessions; disabling town balancing uses source-proportional quotas. Small or indivisible session groups can make final train/validation/test proportions differ from requested ratios.

The splitter uses a seeded, group-level assignment that attempts to approximate requested sample ratios. Since group sizes differ, exact ratios are not guaranteed. A town holdout assigns all groups that contain a selected town to test; session integrity can consequently move additional samples from those groups into test.

These controls do not eliminate every possible source of dependence. Similar scenes collected in different sessions, repeated routes, shared weather conditions, or adjacent simulator states may remain correlated. Report the grouping strategy and interpret generalization claims accordingly.

## Metrics

For each class \(c\), with true positives \(TP_c\), false positives \(FP_c\), and false negatives \(FN_c\):

\[
IoU_c = \frac{TP_c}{TP_c + FP_c + FN_c}
\]

\[
Precision_c = \frac{TP_c}{TP_c + FP_c},
\qquad
Recall_c = \frac{TP_c}{TP_c + FN_c}
\]

\[
Dice_c = \frac{2TP_c}{2TP_c + FP_c + FN_c}
\]

The workbench reports per-class IoU, precision, recall, Dice, pixel accuracy, mean pixel accuracy, overall mean IoU, foreground-only mean IoU, and the confusion matrix. Undefined per-class denominators are represented as null in JSON rather than silently converted to zero.

Overall mIoU averages the class IoUs, including background. Foreground mIoU averages only the non-background classes. State which quantity is used in any paper, report, or résumé result.

## Fine-tuning versus resume

These are intentionally distinct operations:

- **Fine-tune from `best.pt`:** starts a new experiment and new output directory using the selected checkpoint as an initialization. This is the appropriate route when changing the dataset or training configuration.
- **Resume from `last.pt`:** continues the interrupted training run in its original run directory and restores its training state. RoadForge verifies the associated dataset YAML before allowing the resume operation.

A run stores a configuration snapshot, split-manifest copy, training log, metrics, and run metadata. Keep these files with any result you publish.

## Ignore-label handling

An optional ignore value can be enabled for data validation and evaluation. In the current implementation, training is blocked if train/validation masks contain ignore pixels because the installed Ultralytics semantic-training integration does not expose a verified ignore-index setting in this application. This avoids accidentally treating the ignore value as a normal semantic class.

## Reporting checklist

A reproducible experiment report should include:

- simulator version, towns/maps, weather and traffic settings;
- camera resolution, FOV and mounting transform;
- total raw frames, valid image-mask pairs, and class-pixel distribution;
- teacher model origin and cryptographic hash, ONNX input shape, provider, threshold and morphology settings;
- split seed, grouping unit, held-out towns, and actual split proportions;
- student architecture/checkpoint, framework version, input size, epochs, optimizer, batch size, device, deterministic setting, and augmentations;
- per-class metrics and confusion matrix on validation and test;
- inference hardware, warm-up/iteration counts, timing scope, and ONNX Runtime providers;
- known limitations, pseudo-label quality risks, and any manual filtering or corrections.

Do not present a single validation number as a generalization result. Include per-class metrics, test-set protocol, and data provenance.
