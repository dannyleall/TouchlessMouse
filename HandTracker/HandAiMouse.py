import cv2
import numpy as np
import HandTracker.HandFunctions as HT
import time
import autopy

def OperateMouse():

    cap = cv2.VideoCapture(0)

    # Height and width of camera.
    widthCam, heightCam = 640, 480
    cap.set(3, widthCam)
    cap.set(4, heightCam)

    # Current and present times for fps calculations.
    cTime = 0
    pTime = 0

    # Create detection object from our library for only one hand.
    detector = HT.HandDetector(maxHands=1)

    # Get measurements for monitor screen.
    widthScr, heightScr = autopy.screen.size()

    # Frame reduction variables.
    frameR = 100

    # Smoothening value.
    smooth = 7

    # Previous and current locations variables.
    pLocX, pLocY = 0, 0
    cLocX, cLocY = 0, 0 


    while True:

        # Flip camera so it is not inverted.
        success, img = cap.read()
        img = cv2.flip(img, 1)

        # Find hands in camera.
        img = detector.FindHands(img)

        # Create box around hand and find positions of hand landmarks.
        landmarkList, bbox = detector.FindPosition(img)

        # Get index (8) and middle (12) finger tips.
        if len(landmarkList) != 0:

            x1, y1 = landmarkList[8][1:]
            x2, y2 = landmarkList[12][1:]

            # Check which fingers are up.
            fingers = detector.FingersUp()
            print(fingers)

            # Check if index finger is up.
            if fingers[1] == 1 and fingers[3] == 0 and fingers[4] == 0:
                # Allows for a rectangle to imitate monitor.
                cv2.rectangle(img, (frameR, frameR), (widthCam - frameR, heightCam - frameR), (255, 0, 255), 2)

                # Convert coordinates to monitor resolution coordinates.
                x3 = np.interp(x1, (frameR, widthCam - frameR), (0, widthScr))
                y3 = np.interp(y1, (frameR, heightCam - frameR), (0, heightScr))

                # Smoothen values.
                cLocX = pLocX + (x3 - pLocX) / smooth
                cLocY = pLocY + (y3 - pLocY) / smooth

                # Use finger as cursor.
                autopy.mouse.move(cLocX, cLocY)

                # Circle for cursor hover when moving mouse.
                cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)   

                # Update location values.
                pLocX, pLocY = cLocX, cLocY        

                # Find distance between landmark 5 and 8 for the fingers.
                lengthI, img, lineInfoI = detector.FindDistance(5, 8, img)

                # If length is small, then use as click.
                if lengthI < 110:
                    # Circle for when in left clicking mode.
                    cv2.circle(img, (lineInfoI[4], lineInfoI[5]), 15, (0, 255, 0), cv2.FILLED)

                    # Left click.
                    autopy.mouse.click()
                    time.sleep(.1)

                # Check if index and middle fingers are up.
                if fingers[2] == 1:
                    # Find distance between landmark 9 and 12.
                    lengthM, img, lineInfoM = detector.FindDistance(9, 12, img)

                    # If length of index and middel is small, then right click.
                    if lengthM < 120:
                        # Circle for when in right clicking mode.
                        cv2.circle(img, (lineInfoM[4], lineInfoM[5]), 15, (0, 255, 0), cv2.FILLED)

                        # Right click.                        
                        autopy.mouse.click(autopy.mouse.Button.RIGHT)
                        time.sleep(.1)
        
        # Calculate fps and display it.
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 20, 147), 3)

        # Display camera image.
        cv2.imshow("Image", img)
        cv2.waitKey(1)