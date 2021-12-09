import mediapipe as mp
import cv2
import numpy as np
import uuid
import os
import time

mpDraw = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mpHands = mp.solutions.hands 

cTime=0  #current time
pTime=0 # pastime
#os.mkdir('OutputImages')

def write_img():
    global cTime, pTime
    cap = cv2.VideoCapture(0)

    with mpHands.Hands(min_detection_confidence =0.5,min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring the empty camera frame")
                continue
            #to improve performance write image as not writable
            image.flags.writeable = False
            imageRGB = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            results = hands.process(imageRGB)

            #draw the landmark results on image
            image.flags.writeable = True
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mpDraw.draw_landmarks(image, hand_landmarks, mpHands.HAND_CONNECTIONS,mpDraw.DrawingSpec(color=(255,0,0),thickness=2,circle_radius=3),
                                        mpDraw.DrawingSpec(color=(0,255,0),thickness=2,circle_radius=3))
            cTime= time.time()
            fps = 1/(cTime-pTime)
            cv2.putText(image,str(int(fps)), (100,100),cv2.FONT_HERSHEY_COMPLEX, 3, (255,0,0),3)   
            if results.multi_hand_landmarks:
                cv2.imwrite(os.path.join('OutputImages', '{}.jpg'.format(uuid.uuid4())), image)     
            cv2.imshow('Mediapipe Hand pose ', image)
            pTime= cTime
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cv2.destroyAllWindows()
    return 



        

