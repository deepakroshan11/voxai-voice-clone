#!/usr/bin/env bash
# ═══════════════════════════════════════════════
#  VOXAI — Start Server
# ═══════════════════════════════════════════════

set -e
source venv/bin/activate 2>/dev/null || true

echo ""
echo "  🚀 Starting VOXAI Server..."
echo "  Open in browser: http://localhost:8000"
echo "  API docs:        http://localhost:8000/docs"
echo ""

cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
