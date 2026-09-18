#!/bin/bash
# One-time setup for ScreenPeek. Creates a local venv so nothing touches
# your system Python. Safe to re-run: it rebuilds the venv from scratch.
set -e
cd "$(dirname "$0")"

if [ -d .venv ]; then
  echo "Removing the old virtual environment..."
  rm -rf .venv
fi

echo "Creating virtual environment..."
python3 -m venv .venv

echo "Installing OpenCV (this takes a minute)..."
./.venv/bin/python -m pip install --upgrade pip --quiet
./.venv/bin/python -m pip install -r requirements.txt

# The YuNet face model ships in this folder. If it has gone missing, fetch it
# from the OpenCV model zoo and verify the checksum. Haar is the fallback.
MODEL=face_detection_yunet_2023mar.onnx
SHA=8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4
if [ ! -s "$MODEL" ]; then
  echo "Face model missing — downloading from the OpenCV model zoo..."
  URL="https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/$MODEL"
  if curl -fsSL -o "$MODEL.tmp" "$URL" \
     && [ "$(shasum -a 256 "$MODEL.tmp" | cut -d' ' -f1)" = "$SHA" ]; then
    mv "$MODEL.tmp" "$MODEL"
    echo "  downloaded and verified."
  else
    rm -f "$MODEL.tmp"
    echo "  could not download it; ScreenPeek will fall back to the slower Haar detector."
  fi
fi

echo
echo -n "Installed: "
./.venv/bin/python -c "import cv2; print('OpenCV', cv2.__version__, '| YuNet available:', hasattr(cv2, 'FaceDetectorYN'))"

cat <<'EOF'

Done.

Test it first:
    ./run.sh --check

Then start watching:
    ./run.sh

First run will ask for camera permission. If it doesn't, enable it manually:
    System Settings > Privacy & Security > Camera > Terminal
then quit Terminal (Cmd+Q) and reopen it.

EOF
