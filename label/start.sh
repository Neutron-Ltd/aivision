#!/bin/bash

# Get the directory where THIS script lives (the 'label' folder)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# Get the project root (one level up from 'label' folder)
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

# 1. Clean up old zombies
kill -9 $(lsof -ti:8765,9090,8080) 2>/dev/null

# 2. Start CORS Image Server (Running from root so it finds /images)
cd "$ROOT_DIR"
python label/server.py & 

# 3. Start ML Backend (Running from root so it finds /runs)
python label/label.py &

# 4. Generate fresh CSV manifest
# Scans images/people2 and saves people2.csv into the label folder
python -c "import os; open('label/people2.csv', 'w').write('image\n' + '\n'.join([f'http://<your-ip>:8765/people2/{f}' for f in os.listdir('images/people2') if f.lower().endswith(('.jpg', '.jpeg', '.png'))]))"

# 5. Launch Label Studio
export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
label-studio start