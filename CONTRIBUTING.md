# Contributing

Contributions that improve reproducibility, dataset integrity, model compatibility, and documentation are welcome.

## Before opening a change

1. Check existing issues and avoid introducing overlapping features.
2. Keep raw collection files, generated datasets, trained weights, and local workspace output out of source control.
3. Preserve the three-class data contract unless a change explicitly migrates the dataset schema and all dependent code.
4. Keep long-running Qt work off the GUI thread.
5. Avoid changing training semantics or dataset-split behavior without tests.

## Development setup

Use 64-bit Python 3.12:

```bat
setup.bat
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

Run the tests:

```bat
.venv\Scripts\python.exe -m pytest -q
```

The core tests do not require CARLA or pretrained checkpoints. GUI tests require PySide6. The Ultralytics integration test requires the pinned framework release and constructs a local architecture without downloading pretrained weights.

## Pull request expectations

A useful pull request should include:

- a clear problem statement and implementation summary;
- tests for data validation, grouping, metrics, or model-interface behavior when relevant;
- updated documentation for changed input/output formats or defaults;
- notes about compatibility, migration, and any model/license implications;
- evidence for performance claims measured on named hardware and with a stated timing scope.

Do not include private datasets, user-specific absolute paths, model weights, simulator binaries, or API credentials.
