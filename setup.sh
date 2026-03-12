#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════
#  VOXAI — Setup Script
#  Run once: bash setup.sh
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

# Check Python
python3 --version || { echo "❌ Python 3.9+ required"; exit 1; }

# Create virtual env
echo "→ Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
echo "→ Installing dependencies (this may take a few minutes)..."
pip install -r requirements.txt

# Create directories
mkdir -p outputs profiles

echo ""
echo "✅ Setup complete!"
echo ""
echo "To run the app:"
echo "  source venv/bin/activate"
echo "  bash run.sh"
echo ""
