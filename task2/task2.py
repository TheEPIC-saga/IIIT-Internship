from ultralytics import YOLO
import os

def main():
    # Path to your data.yaml (update if needed)
    data_yaml = os.path.join(os.path.dirname(__file__), 'AerialSolarPanel', 'data.yaml')
    
    # Choose model (change to yolov8n.pt, yolov8s.pt, etc. as needed)
    model = YOLO('yolov8n.pt')

    # Train with smaller image size and reduced workers
    model.train(data=data_yaml, epochs=50, imgsz=416, project='runs', name='exp_train', workers=0)

    # Validate
    model.val(data=data_yaml, project='runs', name='exp_val', imgsz=416, workers=0)

    # Test (if test set is defined in data.yaml)
    model.val(data=data_yaml, split='test', project='runs', name='exp_test', imgsz=416, workers=0)

if __name__ == "__main__":
    main()
