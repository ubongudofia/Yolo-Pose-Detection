import cv2
from ultralytics import YOLO
from flask import Flask,  Response



app = Flask(__name__)
video_source = cv2.VideoCapture("rtsp://admin:PwSetit2023!@192.168.1.64:554/streaming/channels/101")


if not video_source.isOpened():
    print('Error: Unable to load video from source')
else:
    print('Video is streaming')


def process_frame(frame):
    # resize my frame 
    frame = cv2.resize(frame, (1000, 600))
    cv2.putText(frame, 'DSA', (650, 590), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 128, 0), 2)
    return frame


model = YOLO("models/yolo11n-pose.pt")
def display_in_browser():
    while True:
        status, frame = video_source.read()
        if not status or frame is None:
            break
        frame = process_frame(frame)
        frame = get_yolo(frame)
        if cv2.waitKey(1) == 27:  
            break
        fw = cv2.imencode('.jpg',frame)[1].tobytes()
        yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + fw + b'\r\n')
    cv2.destroyAllWindows()


def get_yolo(img):
    res = model(img,verbose=False)
    # names = res[0].names
    person = 0
    cars = 0
    rdata = res[0].boxes
    for data in rdata:
        cls = int(data.cls[0])
        conf =  data.conf[0]
        x1 = int(data.xyxy[0][0])
        y1 = int(data.xyxy[0][1])
        x2 = int(data.xyxy[0][2])
        y2 = int(data.xyxy[0][3])
        img = cv2.rectangle(img,(x1,y1),(x2,y2),(122,150,233),2)
        name = res[0].names[cls]
        cv2.putText(img,name,(x1,y1-10),cv2.FONT_HERSHEY_DUPLEX ,1, (237,149,233),2)

        if name == 'person':
            person += 1
        if name == 'car' or name == 'bus' or name == 'truck':
            cars += 1
        cv2.putText(img,f'Person: {person} Cars: {cars}',(690,20),cv2.FONT_HERSHEY_DUPLEX ,1, (237,149,233),2)

    return img


@app.route('/')
def index():
    return Response(display_in_browser(),mimetype='multipart/x-mixed-replace; boundary=frame')
app.run('0.0.0.0', debug=True)


# def sss():
#     vid = cv2.VideoCapture('rtsp://admin:PwSetit2023!@192.168.1.64:554/streaming/channels/101')
#     while True:
#         _ , frame = vid.read()
#         frame  = get_yolo1(frame)
#         cv2.imshow('DLC', frame)
#         if cv2.waitKey(1) == 27:
#             break

# def get_yolo1(img):
#     res =model(img, verbose=False)
#     rdata = res[0].boxes
#     cars = 0
#     person = 0
#     cars = 0
#     for d in rdata:
#         cls = int(d.cls[0])
#         conf = d.conf[0]
#         x1 = int(d.xyxy[0][1])
#         y1 = int(d.xyxy[0][1])
#         x2 = int(d.xyxy[0][2])
#         y2 = int(d.xyxy[0][3])
#         img = cv2.rectangle(img,(x1,y1),(x2,y2),(255,255,255),1)
#         name = res[0].names[cls]
#         cv2.putText(img,name,(x1,y1-10),cv2.FONT_HERSHEY_PLAIN,0.6,(0,0,255),1)
#     return img

