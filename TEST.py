import cv2
cap = cv2.VideoCapture("rtsp://admin:QKWQST@192.168.50.231:554/")


def frameDeal(frame):
    pass


while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        frameDeal(frame)
    else:
        cap.release()
        cap = cv2.VideoCapture("rtsp://admin:QKWQST@192.168.50.231:554/")