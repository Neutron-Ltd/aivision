# Labelling with Label Studio

The trained model pre-annotates images in [Label Studio](https://labelstud.io), so you only correct boxes instead of drawing them all.

## Requirements

```bash
pip install label-studio label-studio-ml
```

## Start everything

From the repository root:

```bash
label/start.sh <folder>      # serves images/<folder>, e.g. label/start.sh people
```

This starts:

| Service | Port | Purpose |
|---------|------|---------|
| `label/server.py` | 8765 | Serves `images/` with CORS so the browser can load them |
| `label/label.py` | 9090 | ML backend that returns YOLO predictions |
| Label Studio | 8080 | Labelling UI |

It also writes `label/<folder>.csv`, an import file whose image URLs use your current IP.

To use different weights, set `MODEL_PATH` before starting, e.g. `MODEL_PATH=runs/detect/aiv6s5/weights/best.pt label/start.sh people`.

## Configure the project (first time)

1. Open http://localhost:8080 and create a project.
2. **Settings → Labeling Interface → Code**, paste:

   ```xml
   <View>
     <Image name="image" value="$image"/>
     <RectangleLabels name="label" toName="image">
       <Label value="person" background="red"/>
     </RectangleLabels>
   </View>
   ```

3. **Settings → Model**: add `http://<your-ip>:9090`, turn on interactive pre-annotations, then validate and save.
4. On the project dashboard, **Import** `label/<folder>.csv`.

Export the finished labels in YOLO format to build your `dataset/`.
