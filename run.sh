#!/usr/bin/env bash
# ═══════════════════════════════════════════════
#  VOXAI — Start Server
#  Run from repo root: bash run.sh
# ═══════════════════════════════════════════════

set -e

# Must run from repo root
if [ ! -d "backend" ]; then
  echo "❌ Run this from the repo root (voice-clone/), not from inside backend/"
  exit 1
fi

# Activate venv (silently skip if already active or not found)
source venv/bin/activate 2>/dev/null || true

echo ""
echo "  🚀 Starting VOXAI Server..."
echo "  Open in browser: http://localhost:8000"
echo "  API docs:        http://localhost:8000/docs"
echo "  Health check:    http://localhost:8000/health"
echo ""

# Run from repo root so backend.main path resolves correctly
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload