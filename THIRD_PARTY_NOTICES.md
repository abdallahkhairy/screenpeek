# Third-party notices

ScreenPeek bundles two face-detection models so that it works offline and
regardless of which OpenCV build is installed.

## YuNet face detector — `face_detection_yunet_2023mar.onnx`

- Source: [OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)
- Authors: Shiqi Yu and contributors (Shenzhen University / OpenCV China)
- License: MIT
- SHA-256: `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4`

## Haar cascade — `haarcascade_frontalface_default.xml`

- Source: [OpenCV](https://github.com/opencv/opencv/tree/4.x/data/haarcascades)
- Original author: Rainer Lienhart
- License: Apache License 2.0 (OpenCV), with the file's own BSD-style header

## OpenCV (installed by `install.sh`, not bundled)

- [opencv-python](https://pypi.org/project/opencv-python/) — Apache License 2.0

## Test fixture — `tests/fixtures/astronaut.png`

- Portrait of astronaut Eileen Collins, NASA. Public domain.
- Distributed via scikit-image's sample data.
