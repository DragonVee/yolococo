import torch
import numpy as np
import cv2
import time
import pafy

prev_time = 0
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
url = "https://www.youtube.com/watch?v=RQA5RcIZlAM&ab_channel=%E3%80%90LIVE%E3%80%91%E6%96%B0%E5%AE%BF%E5%A4%A7%E3%82%AC%E3%83%BC%E3%83%89%E4%BA%A4%E5%B7%AE%E7%82%B9TokyoShinjukuLiveCh"
live = pafy.new(url)
stream = live.getbest(preftype="mp4")
cap = cv2.VideoCapture(stream.url)
while cap.isOpened():
    success, frame = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
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