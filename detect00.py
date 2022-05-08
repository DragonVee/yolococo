import torch
import cv2
import numpy as np

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
# model.conf = 0.5
# print(model)
img=cv2.imread('708e1932-4f62-4d9c-802b-9468ea0360bd.png')
results = model(img)
results.print()
print(results.xyxy)
cv2.imshow('YOLO COCO', np.squeeze(results.render()))
cv2.waitKey(0)

0
