#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════
#  VOXAI — Setup Script
#  Run once from the repo root: bash setup.sh
# ═══════════════════════════════════════════════════════════

set -e
echo ""
echo "  ██╗   ██╗ ██████╗ ██╗  ██╗ █████╗ ██╗"
echo "  ██║   ██║██╔═══██╗╚██╗██╔╝██╔══██╗██║"
echo "  ██║   ██║██║   ██║ ╚███╔╝ ███████║██║"
echo "  ╚██╗ ██╔╝██║   ██║ ██╔██╗ ██╔══██║██║"
echo "   ╚████╔╝ ╚██████╔╝██╔╝ ██╗██║  ██║██║"
echo "    ╚═══╝   ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝"
echo ""
echo "  Zero-Shot Voice Cloning — Setup"
echo "═══════════════════════════════════════"
echo ""

# Must run from repo root
if [ ! -d "backend" ]; then
  echo "❌ Run this from the repo root (voice-clone/), not from inside backend/"
  exit 1
fi

# Check Python version
python3 --version || { echo "❌ Python 3.10+ required"; exit 1; }

# Create virtual env at repo root
echo "→ Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies — same tested versions as the Dockerfile (constraints.txt pins them)
echo "→ Installing dependencies (this may take several minutes)..."
pip install "torch==2.6.0" "torchaudio==2.6.0"
pip install -c constraints.txt "chatterbox-tts==0.1.7"
pip install -c constraints.txt "git+https://github.com/ai4bharat/IndicF5.git"
pip install -c constraints.txt -r backend/requirements.txt

# Create runtime directories
mkdir -p outputs profiles

echo ""
echo "✅ Setup complete!"
echo ""
echo "To run the app:"
echo "  bash run.sh"
echo ""