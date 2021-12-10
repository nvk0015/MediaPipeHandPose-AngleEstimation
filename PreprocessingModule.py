import mediapipe as mp
import cv2
import numpy as np
import uuid
import os
import time

class Preprocessor():
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


    def leftrightClassifier(self, image, handNum =0, landmarkNum=0):

        '''THe input to handNum should be either 0 or 1
        0 -- Left Hand
        1 -- Right Hand
        lanmark number is to be given as input with reference to mediapipe labels'''

        #access hand landamrks of left and right hand 0,1 can be indexed in case of 2 detected hands
        #self.results.multi_hand_landmarks[0] 
        #access coordinates of a particular lanmark
        #self.results.multi_hand_landmarks[0].landmark[self.mpHands.HandLandmark.WRIST]

        image.flags.writeable = False
        imageRGB = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imageRGB)
        image.flags.writeable = False
        for idx, classification in enumerate(self.results.multi_handedness):
            if classification.classification[0].index == handNum:
                label = classification.classification[0].index 
                score = classification.classification[0].score
                text = "{} {}".format(label, round(score,2))

                #extract coordinates
                coords = tuple(np.multiply(
                    np.array((classification.landmark[self.mpHands.HandLandmark.landmarkNum].x,classification.landmark[self.mpHands.HandLandmark.landmarkNum].y)),
                    [640,480]).astype(int))
                    
                output = text, coords
        return output

def main():
    cTime=0  #current time
    pTime=0 # pastime
    #os.mkdir('OutputImages')
    cap = cv2.VideoCapture(0)
    detector = Preprocessor()
    while True:
        success, frame = cap.read()
        image = frame
        location = detector.leftrightClassifier(image)

        cTime = time.time()
        fps = 1/(cTime-pTime)
        cv2.putText(image,str(int(fps)), (10,70),cv2.FONT_HERSHEY_COMPLEX, 3, (255,0,0),3)   
        cv2.putText(image,str(location), (100,70),cv2.FONT_HERSHEY_COMPLEX, 3, (255,0,0),3) 
        cv2.imwrite(os.path.join('OutputImages', '{}.jpg'.format(uuid.uuid4())), image)     
        cv2.imshow('Mediapipe Preprocessor ', image)
        pTime= cTime
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()