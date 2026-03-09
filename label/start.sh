#!/bin/bash

# Get the directory where THIS script lives (the 'label' folder)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# Get the project root (one level up from 'label' folder)
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

# 1. Detect current local IP (Mac specific)
# This grabs the IP of the active network interface. 
CURRENT_IP=$(ipconfig getifaddr $(route get default | grep interface | awk '{print $2}'))
echo "Detected IP: $CURRENT_IP"

# 2. Clean up old zombies
kill -9 $(lsof -ti:8765,9090,8080) 2>/dev/null

# 3. Start CORS Image Server
cd "$ROOT_DIR"
python label/server.py & 

# 4. Start ML Backend 
# We pass the CURRENT_IP to the environment just in case
export CURRENT_IP=$CURRENT_IP
python label/label.py &

# 5. Generate fresh CSV manifest with the ACTIVE IP
# This fixes the "issue loading URL" error by ensuring links match your current network
python -c "import os; ip='$CURRENT_IP'; open('label/people.csv', 'w').write('image\n' + '\n'.join([f'http://{ip}:8765/people/{f}' for f in os.listdir('images/people') if f.lower().endswith(('.jpg', '.jpeg', '.png'))]))"

# 6. Launch Label Studio
export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
label-studio start