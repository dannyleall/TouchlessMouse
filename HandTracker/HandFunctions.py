import cv2
import mediapipe as mp
import math


class HandDetector():

    def __init__(self, mode = False, maxHands = 2, modelComplexity = 1, detectConfidence = 0.5, trackConfidence = 0.5):

        # Creating object from mode parameter.
        self.mode = mode
        self.maxHands = maxHands
        self.modelComplexity = modelComplexity
        self.detectConfidence = detectConfidence
        self.trackConfidence = trackConfidence

        # Initialize for instances.
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplexity, self.detectConfidence, self.trackConfidence)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]


    def FindHands(self, img, draw = True):

        # Convert image to rgb image because hands only accepts rgb.
        imgRgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        self.results = self.hands.process(imgRgb)

        # Performed if a hand is detected.
        if self.results.multi_hand_landmarks:

            # Checks if there are multiple hands and extract one by one.
            for handLandmarks in self.results.multi_hand_landmarks:

                if draw:
                    # Draw the 21 points of the hand landmarks and the line connections.
                    self.mpDraw.draw_landmarks(img, handLandmarks, self.mpHands.HAND_CONNECTIONS)

        # Because we are drawing, we must return the image.
        return img


    def FindPosition(self, img, handNum = 0, draw = True):

        xList =[]
        yList = []
        bbox = []
        self.landmarkList = []
        
        # Performed if a hand is detected.
        if self.results.multi_hand_landmarks:

            myHand = self.results.multi_hand_landmarks[handNum]

            # Get hand landmark id information for specific hand detected.
            for id, landmark in enumerate(myHand.landmark):
                        
                height, width, center = img.shape
                cx, cy = int(landmark.x * width), int(landmark.y * height)
                
                # Append values.
                xList.append(cx)
                yList.append(cy)
                self.landmarkList.append([id, cx, cy])

                # Draw the landmark circles.
                if draw:

                    cv2.circle(img, (cx, cy), 7, (255,0,255), cv2.FILLED)
            
            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = xmin, ymin, xmax, ymax

            # Draw the bbox.
            if draw:

                cv2.rectangle(img, (xmin - 30, ymin - 30), (xmax + 30, ymax + 30), (255, 255, 255))

        return self.landmarkList, bbox

    
    def FingersUp(self):
        
        fingers = []

        # Checks if thumb finger is up by comparing to other landmark values.
        if self.landmarkList[self.tipIds[0]][1] > self.landmarkList[self.tipIds[0] - 1][1]:

            fingers.append(1)

        else: 
            
            fingers.append(0)

        # Checks if any other finger is up.
        for id in range(1, 5):

            if self.landmarkList[self.tipIds[id]][2] < self.landmarkList[self.tipIds[id] - 2][2]:

                fingers.append(1)

            else:

                fingers.append(0)

        return fingers


    def FindDistance(self, p1, p2, img, draw = True, r = 15, t = 3):

        x1, y1 = self.landmarkList[p1][1:]

        x2, y2 = self.landmarkList[p2][1:]

        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:

            cv2.line(img, (x1, y1), (x2, y2), (255, 150, 10), t)
            cv2.circle(img, (x1, y1), r, (255, 150, 10), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 150, 10), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (0, 150, 10), cv2.FILLED)

        length = math.hypot(x2 - x1, y2 - y1)

        return length, img, [x1, y1, x2, y2, cx, cy]