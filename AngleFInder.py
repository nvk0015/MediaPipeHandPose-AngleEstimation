import mediapipe as mp
import cv2
import numpy as np
import uuid
import os
import time
import HandPoseModule

#handpose.write_img()
detector = HandPoseModule.HandsDetector()
pTime = 0
cTime = 0
cap = cv2.VideoCapture(0)

while True:
    success, image = cap.read()
    image = detector.findHands(image)
    cTime =time.time()
    fps = 1/(cTime-pTime)
    pTime=cTime
    cv2.putText(image, str(int(fps)), (10,70), cv2.FONT_HERSHEY_DUPLEX,2,(255,255,255), 3)
    cv2.imwrite(os.path.join('OutputImages', '{}.jpg'.format(uuid.uuid4())), image)  
    cv2.imshow('image',image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()


img = cv2.imread('test.jpg')
print(img)

#location = detector.LeftRightClassifier(img)
