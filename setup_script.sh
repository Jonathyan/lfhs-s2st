#!/bin/bash

# SeamlessM4T v2 Setup Script voor MacBook Pro M1 Pro
# --------------------------------------------------

echo "Creating virtual environment with Python 3.10 (compatible with SeamlessM4T)..."
# Check if Python 3.10 is available
if command -v python3.10 &> /dev/null; then
    python3.10 -m venv venv
else
    echo "Python 3.10 not found. Checking if we can install it with brew..."
    if command -v brew &> /dev/null; then
        echo "Installing Python 3.10 with Homebrew..."
        brew install python@3.10
        python3.10 -m venv venv
    else
        echo "WARNING: Python 3.10 not found and Homebrew not available."
        echo "Using current Python version, but this may cause compatibility issues."
        python3 -m venv venv
    fi
fi

source venv/bin/activate

echo "Installing ffmpeg (required for audio processing)..."
brew install ffmpeg

echo "Upgrading pip and installing wheel..."
pip install --upgrade pip wheel setuptools

echo "Installing PyTorch with MPS support for M1 Mac..."
pip install torch==2.0.1 torchaudio==2.0.2 --extra-index-url https://download.pytorch.org/whl/cpu

echo "Installing core dependencies..."
pip install numpy==1.24.3 pydub==0.25.1 ffmpeg-python==0.2.0 sentencepiece==0.1.99 soundfile==0.12.1 accelerate==0.20.3 protobuf==3.20.3 huggingface-hub

echo "Installing Meta's seamless_communication package..."
pip install git+https://github.com/facebookresearch/seamless_communication.git

# Create directory structure
echo "Creating project directories..."
mkdir -p input output

echo "Setup complete! Next steps:"
echo "1. Place your Dutch sermon audio file in the 'input' folder as 'dutch_sermon.wav'"
echo "2. (Optional) Add a voice reference file in the 'input' folder as 'voice_reference.wav'"
echo "3. Run the translator: python seamless_s2st_mvp.py"