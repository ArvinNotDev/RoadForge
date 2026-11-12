# Third-party notices

RoadForge is an independent project. It relies on third-party libraries and model artifacts whose terms remain separate from the MIT license covering original RoadForge source code.

## YOLOPv2 ONNX preprocessing reference

The offline YOLOPv2 adapter in `datacollector/teacher_labeling.py` contains an adaptation of preprocessing and segmentation-mask extraction patterns from:

- Project: [Kazuhito00/YOLOPv2-ONNX-Sample](https://github.com/Kazuhito00/YOLOPv2-ONNX-Sample)
- Author: KazuhitoTakahashi
- License: MIT
- License file: [upstream LICENSE](https://github.com/Kazuhito00/YOLOPv2-ONNX-Sample/blob/main/LICENSE)

The upstream MIT attribution and permission notice are reproduced below.

```text
MIT License

Copyright (c) 2023 KazuhitoTakahashi

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Ultralytics

RoadForge uses the `ultralytics` Python package and pins the application dependency to version `8.4.92`. Ultralytics source code and models are distributed under terms maintained by Ultralytics; the upstream project identifies AGPL-3.0 licensing and offers separate enterprise licensing. Review the [Ultralytics licensing page](https://www.ultralytics.com/license) and the exact release terms before redistributing a packaged application or model that includes or depends on Ultralytics components.

The MIT license in this repository does not relicense Ultralytics code, pretrained weights, or other third-party artifacts.

## Other runtime dependencies

PySide6 / Qt, PyTorch, NumPy, Pillow, OpenCV, ONNX, ONNX Runtime, and PyYAML are third-party packages. Their own licenses and notices apply. Install them from their official distribution channels and review their terms for your intended use.

## CARLA and model weights

The CARLA Python API/server and YOLOPv2 or YOLO26 weights are not bundled in this repository. Their licenses and use restrictions are determined by their respective upstream projects and model publishers. Users are responsible for obtaining and using those artifacts lawfully.
