#!/bin/bash
# Start the labelling stack: image server, YOLO pre-annotation backend and Label Studio.
#
# Usage: label/start.sh [images-subfolder]   (default: canteen3)
# Serves images/<subfolder> and writes the Label Studio import file label/<subfolder>.csv.

set -euo pipefail

FOLDER="${1:-canteen3}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR"

if [ ! -d "images/$FOLDER" ]; then
    echo "images/$FOLDER not found" >&2
    exit 1
fi

# 1. Detect the machine's LAN IP so image URLs work from the browser.
#    macOS first, then Linux, then localhost.
CURRENT_IP=""
if command -v ipconfig > /dev/null && command -v route > /dev/null; then
    IFACE=$(route get default 2>/dev/null | awk '/interface/ {print $2}')
    [ -n "$IFACE" ] && CURRENT_IP=$(ipconfig getifaddr "$IFACE" 2>/dev/null || true)
fi
if [ -z "$CURRENT_IP" ] && command -v hostname > /dev/null; then
    CURRENT_IP=$(hostname -I 2>/dev/null | awk '{print $1}' || true)
fi
CURRENT_IP="${CURRENT_IP:-127.0.0.1}"
echo "Detected IP: $CURRENT_IP"

# 2. Stop anything left over from a previous session on our ports.
kill -9 $(lsof -ti:8765,9090,8080) 2>/dev/null || true

# 3. Start the CORS image server (serves ./images on port 8765).
python label/server.py &

# 4. Start the ML backend (port 9090).
python label/label.py &

# 5. Generate the import CSV with the current IP.
python - "$CURRENT_IP" "$FOLDER" <<'EOF'
import os, sys
ip, folder = sys.argv[1], sys.argv[2]
files = sorted(f for f in os.listdir(f"images/{folder}") if f.lower().endswith((".jpg", ".jpeg", ".png")))
with open(f"label/{folder}.csv", "w") as fh:
    fh.write("image\n" + "\n".join(f"http://{ip}:8765/{folder}/{f}" for f in files) + "\n")
print(f"Wrote label/{folder}.csv ({len(files)} images)")
EOF

# 6. Launch Label Studio (port 8080).
export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
label-studio start
