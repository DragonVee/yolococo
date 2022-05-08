import torch
import numpy as np
import cv2
import time
#import pafy

prev_time = 0
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
#url = "rtsp://admin:QKWQST@192.168.50.231:554/h264/ch1/main/av_stream"
#live = pafy.new(url)
#stream = live.getbest(preftype="mp4")
cap = cv2.VideoCapture("rtsp://admin:KWQST@192.168.50.231:554/")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        cap.release()
        cap = cv2.VideoCapture("rtsp://admin:KWQST@192.168.50.231:554/")
        continue
    frame = cv2.resize(frame,(960,540))
    results = model(frame)
    output_image = np.squeeze(results.render())
    cv2.putText(output_image, f'FPS: {int(1 / (time.time() - prev_time))}',
                (3, 40), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)
    prev_time = time.time()
    cv2.imshow('YOLO COCO 02', output_image)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()