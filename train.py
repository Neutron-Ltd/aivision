import torch
from ultralytics import YOLO

def main():
    # 0. Hardware Check
    if torch.cuda.is_available():
        print(f"🚀 Training on GPU: {torch.cuda.get_device_name(0)}")
        device = 0
    elif torch.backends.mps.is_available():
        print("🚀 Training on Apple Silicon (MPS)")
        device = 'mps'
    else:
        print("⚠️ Training on CPU (This will be slow)")
        device = 'cpu'

    # 1. Initialize Model
    # We use 'Medium' (m). It is the sweet spot for 1500+ images.
    # If your GPU runs out of memory, switch to 'yolov8s.pt'.
    model = YOLO('yolov8n.pt')  # downloaded automatically by ultralytics

    # 2. The "Purist" Training Configuration
    results = model.train(
        data='dataset/data.yaml', 
        
        # --- TRAINING DYNAMICS ---
        epochs=300,               # Long training because we turned off aggressive augmentation
        patience=50,              # Stop if no improvement for 50 epochs
        imgsz=640,                # Standard resolution
        batch=64,                 # Lower this to 8 or 4 if you get "Out of Memory" errors
        name='aiv6s5',       # Output folder name
        workers=8,                
        device=device,            
        cache=True,               # Loads images into RAM for speed
        exist_ok=True,            # Overwrite existing folder if needed
        amp=False,                # False = More stable precision (Good for custom datasets)

        # --- "NO FRAGMENTATION" AUGMENTATIONS ---
        mosaic=0.0,       # OFF. No chopping. Shows 100% real images only.
        mixup=0.0,        # OFF. No transparent ghost overlays.
        
        # --- "NO ZOOM" AUGMENTATIONS ---
        scale=0.1,        # VERY LOW (+/- 10%). Keeps the person the actual size they are on camera.
                          # We need a tiny bit to handle people slightly closer/further, but no big zooms.
        
        # --- TOP-DOWN ESSENTIALS ---
        degrees=180.0,    # ON. CRITICAL. A person walking South = Person walking North.
        flipud=0.5,       # ON. Up/Down flip. Doubles your data for top-down views.
        fliplr=0.5,       # ON. Left/Right flip.
        
        # --- COLOR & LIGHTING (The "Hoodie" Fix) ---
        # We KEEP these to teach the model that "Shape > Color".
        hsv_h=0.015,      # Slight color shift
        hsv_s=0.6,        # Moderate saturation shift (Grey vs Red hoodie)
        hsv_v=0.4,        # Brightness shift (Shadow vs Light)
        
        # --- GEOMETRY ---
        shear=0.0,        # OFF. Top-down view is flat; no perspective warping needed.
        perspective=0.0,  # OFF.
        translate=0.1,    # LOW. Allows slight shifting of the image.
    )

    # 3. Export for Deployment (Optional but Recommended)
    success = model.export(format='onnx', opset=12)
    print("🎉 Training & Export Complete!")

if __name__ == '__main__':
    main()