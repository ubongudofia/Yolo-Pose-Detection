from ultralytics import YOLO
import cv2

# Load a COCO-pretrained YOLO11n model
model = YOLO("models/yolo11n-pose.pt")

# Train the model on the COCO8 example dataset for 100 epochs
# results = model.train(data="coco8.yaml", epochs=100, imgsz=640)

# Run inference with the YOLO11n model on the 'bus.jpg' image
#results = model("path/to/bus.jpg")
 

  
model(source='rtsp://admin:PwSetit2023!@192.168.1.64:554/streaming/channels/101',show=True, verbose=False, show_conf=False, show_labels= True)