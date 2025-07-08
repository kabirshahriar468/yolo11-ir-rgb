import warnings
import os
warnings.filterwarnings('ignore')

# Disable wandb for Kaggle environment
os.environ['WANDB_DISABLED'] = 'true'
os.environ['WANDB_MODE'] = 'disabled'
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/11-RGBT/yolo11-RGBT-midfusion.yaml')
    # model.info(True,True)
    # model.load('yolov8n.pt') # loading pretrain weights
    model.train(data=R'ultralytics/cfg/datasets/drone-detection-RGBT.yaml',
                cache=False,
                imgsz=640,
                epochs=100,
                batch=16,
                optimizer='AdamW',
                lr0=0.0008,  # Slightly lower for stability
                lrf=0.01,
                momentum=0.937,
                weight_decay=0.0005,
                warmup_epochs=5,
            
                # Loss weights for small objects in challenging conditions
                box=10.0,    # Higher box loss (dataset has very small objects)
                cls=0.3,     # Lower class loss (only 2 classes, fairly balanced)
                dfl=2.0,     # Higher DFL for better localization
            
                # MINIMAL augmentation since dataset is already heavily distorted
                hsv_h=0.005,     # Minimal hue (thermal imagery)
                hsv_s=0.1,       # Very low saturation (thermal + already distorted)
                hsv_v=0.1,       # Very low brightness (dataset has illumination issues)
                degrees=0.0,     # No rotation (dataset has camera instability)
                translate=0.05,  # Minimal translation (preserve small objects)
                scale=0.1,       # Very minimal scaling (objects already small)
                shear=0.0,       # No shear (dataset already unstable)
                perspective=0.0, # No perspective (avoid further distortion)
                flipud=0.0,      # No vertical flip
                fliplr=0.3,      # Light horizontal flip
                mosaic=0.8,      # High mosaic for context
                mixup=0.1,       # Light mixup
                copy_paste=0.2,  # Moderate copy-paste for small objects
            
                # Training settings for challenging dataset
                patience=30,     # More patience for difficult dataset
                save_period=15,
                amp=True,
                cache=True,
                device='0',
                workers=12,       # Reduced workers due to higher imgsz
            
                # Multi-scale training
                rect=False,
                close_mosaic=10,  # Disable mosaic in last 10 epochs
            
                # Additional optimizations
                overlap_mask=True,  # Better handling of overlapping objects
                mask_ratio=4,
                
                  # using SGD
                # lr0=0.002,
                resume='best.pt', # last.pt path
                # amp=False, # close amp
                # fraction=0.2,
                use_simotm="RGBT",
                channels=4,
                project='Drone-detection-RGBT',
                name='YOLO11n-RGBT-midfusion',
    )