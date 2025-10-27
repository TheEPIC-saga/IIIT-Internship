from ultralytics import YOLO
import cv2
import os
import glob
import subprocess

def process_images(model, image_paths, output_dir, mode='seg'):
    """Process images with segmentation or detection"""
    os.makedirs(output_dir, exist_ok=True)
    results = model(image_paths, stream=True, conf=0.7)
    
    for idx, result in enumerate(results, 1):
        output_path = os.path.join(output_dir, f"{mode}_result_{idx}.jpg")
        result.save(filename=output_path)  # Keep all annotations and boxes
        print(f"Saved {mode} for image {idx}")

def process_video_frames(model, frames_dir, output_dir, mode='seg'):
    """Process video frames with either segmentation or detection"""
    frame_paths = sorted([
        os.path.join(frames_dir, f) 
        for f in os.listdir(frames_dir) 
        if f.endswith('.jpg')
    ])
    
    results = model(frame_paths, stream=True, conf=0.7)
    
    for idx, result in enumerate(results, 1):
        output_path = os.path.join(output_dir, f"result_{idx}.jpg")
        if mode == 'seg':
            result.save(filename=output_path)
        else:  # annotation mode
            result.save(filename=output_path)
        print(f"Processed frame {idx} with {mode}")


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Get all images from the images folder
    input_folder = os.path.join(current_dir, "images")
    image_paths = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        image_paths.extend(glob.glob(os.path.join(input_folder, ext)))
    
    seg_model = YOLO("yolov8x-seg.pt")
    det_model = YOLO("yolov8x.pt")
    
    # Create separate directories for segmentation and detection results
    seg_img_dir = os.path.join(current_dir, "image_segmentation")
    det_img_dir = os.path.join(current_dir, "image_detection")
    
    # Process images with both models
    process_images(seg_model, image_paths, seg_img_dir, mode='seg')
    process_images(det_model, image_paths, det_img_dir, mode='det')
    

if __name__ == "__main__":
    main()
