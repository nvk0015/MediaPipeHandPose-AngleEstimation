import mediapipe as mp
import cv2
import numpy as np
import uuid
import os
import time

class HandsDetector():
    def __init__(self,mode=False,maxHands=2,modelComplexity=1,detectionCon=0.5,trackCOn=0.5):
        self.mode = mode
        self.maxHands=maxHands
        self.modelComplexity=modelComplexity
        self.detectionCon=detectionCon
        self.trackCOn=trackCOn
        self.mpHands = mp.solutions.hands 
        self.mpDraw = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.hands = self.mpHands.Hands(min_detection_confidence =0.5,min_tracking_confidence=0.5)

    def findHands(self,image,draw=True):
        #setting the writable flag of image to False, with an intention to improve the estimation accuracy
        image.flags.writeable = False
        imageRGB = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imageRGB)
        #setting back the writable falg of image to True, to render  the render and write down the results
        image.flags.writeable = False
        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(image, hand_landmarks, self.mpHands.HAND_CONNECTIONS,
                    self.mpDraw.DrawingSpec(color=(255,0,0),thickness=2,circle_radius=3),
                    self.mpDraw.DrawingSpec(color=(0,255,0),thickness=2,circle_radius=3))
        return image

def main():
    cTime=0  #current time
    pTime=0 # pastime
    #os.mkdir('OutputImages')
    cap = cv2.VideoCapture(0)
    detector = HandsDetector()
    while True:
        success, frame = cap.read()
        image = detector.findHands(frame)

        cTime = time.time()
        fps = 1/(cTime-pTime)
        cv2.putText(image,str(int(fps)), (10,70),cv2.FONT_HERSHEY_COMPLEX, 3, (255,0,0),3)   
        cv2.imwrite(os.path.join('OutputImages', '{}.jpg'.format(uuid.uuid4())), image)     
        cv2.imshow('Mediapipe Hand pose ', image)
        pTime= cTime
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

        

