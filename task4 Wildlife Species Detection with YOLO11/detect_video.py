from ultralytics import YOLO
import os

def main():
    # 1. Load your custom-trained weights using a relative path
    model_path = "best.pt"
    
    if not os.path.exists(model_path):
        print(f"Error: '{model_path}' not found in the current folder.")
        print("Please ensure you copied your best.pt to this directory.")
        return

    model = YOLO(model_path)

    # 2. Path to your merged wildlife video (Buffalo, Elephant, Rhino, Zebra)
    video_source = "video1.mp4" 

    if not os.path.exists(video_source):
        print(f"Error: '{video_source}' not found.")
        return

    print(f"--- Starting Detection on: {video_source} ---")

    # 3. Run Inference and SAVE the output
    results = model.predict(
        source=video_source,
        save=True,      
        conf=0.25,      # Confidence threshold for detections
        imgsz=640,      # Matches your training resolution
        project="runs/detect", # Saves output in the standard YOLO structure
        name="internship_demo" # Specifically names the output folder
    )

    print("\n--- PROCESS COMPLETE ---")
    print("Find your processed video in: runs/detect/internship_demo/")

if __name__ == '__main__':
    # Mandatory guard for Windows multiprocessing
    main()