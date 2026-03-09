1. Nuke the Zombies
Before starting a new session, clear the ports so you don't get "Address already in use" errors.

Bash
kill -9 $(lsof -ti:8765,9090,8080) 2>/dev/null
2. Boot the Image Server (Terminal 1)
This hosts your local files with wide-open CORS so the browser doesn't block them.

Activate your .venv.

Run the custom server:

Bash
python cors_server.py
3. Boot the ML Backend (Terminal 2)
This runs the YOLOv8 inference engine.

Activate your .venv.

Run the barebones model script:

Bash
python label.py
4. Boot Label Studio (Terminal 3)
This runs the UI.

Activate your .venv.

Force the local files flag and launch:

Bash
export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
label-studio start
5. Generate the Import Payload
Run this one-liner anywhere to build a properly formatted CSV. Swap people2 if you change your target directory.

Bash
python -c "import os; open('import.csv', 'w').write('image\n' + '\n'.join([f'http://<your-ip>:8765/people2/{f}' for f in os.listdir('images/people2') if f.lower().endswith(('.jpg', '.jpeg', '.png'))]))"
6. The UI Configuration
Open http://localhost:8080.

Create a Project.

Navigate to Settings > Labeling Interface > Code and paste the XML:

XML
<View>
  <Image name="image" value="$image"/>
  <RectangleLabels name="label" toName="image">
    <Label value="person" background="red"/>
  </RectangleLabels>
</View>
Navigate to Settings > Machine Learning.

Add Model: http://<your-ip>:9090.

Toggle Interactive preannotations ON, then hit Validate and Save.

Go to the project dashboard, click Import, and upload your import.csv.