
import cv2
import mediapipe as mp
import time
import math

class hand_detector():
    def __init__(self, mode=False, modelComplexity=1, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.modelComplex = modelComplexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplex, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]

    def find_hands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,self.mpHands.HAND_CONNECTIONS)
        return img

    def find_position(self, img, handNo=0, draw=True):
        self.lmList = []
        bbox1, bbox2 = [], []
        x_list, y_list = [], []
        all_hands = []

        # Check if hands are detected
        if self.results.multi_hand_landmarks:
            for handType, handLms in zip(self.results.multi_handedness, self.results.multi_hand_landmarks):
                myHand = {}
                x_list.clear()
                y_list.clear()

                # Process each landmark in the hand
                for id, lm in enumerate(handLms.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    x_list.append(cx)
                    y_list.append(cy)
                    self.lmList.append([id, cx, cy])

                    if draw:
                        cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

                # Calculate bounding box for the current hand
                x_min, x_max = min(x_list), max(x_list)
                y_min, y_max = min(y_list), max(y_list)
                bbox = (x_min, y_min, x_max - x_min, y_max - y_min)

                # Populate myHand dictionary
                myHand["lmList"] = self.lmList
                myHand["bbox"] = bbox
                myHand["center"] = (x_min + (x_max - x_min) // 2, y_min + (y_max - y_min) // 2)

                # Assign bounding box and label based on hand type
                if handType.classification[0].label == "Right":
                    myHand["type"] = "Right"
                    myHand["function"] = "Volume"
                    bbox1 = bbox  # Set bbox1 for the right hand
                    if draw:
                        cv2.rectangle(img, (bbox1[0] - 20, bbox1[1] - 20), (bbox1[0] + bbox1[2] + 20, bbox1[1] + bbox1[3] + 20), (255, 0, 255), 2)
                        cv2.putText(img, "Volume", (x_min - 30, y_min - 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 128, 255), 2)
                else:
                    myHand["type"] = "Left"
                    myHand["function"] = "Mode"
                    bbox2 = bbox  # Set bbox2 for the left hand
                    if draw:
                        cv2.rectangle(img, (bbox2[0] - 20, bbox2[1] - 20), (bbox2[0] + bbox2[2] + 20, bbox2[1] + bbox2[3] + 20), (255, 0, 255), 2)
                        cv2.putText(img, "Mode", (x_min - 30, y_min - 30), cv2.FONT_HERSHEY_PLAIN, 2, (128, 0, 255), 2)

                all_hands.append(myHand)

                # Draw hand landmarks
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        
        else:
            print("No hands detected")

        # Return detected hands and bounding boxes
        return all_hands, self.lmList, bbox1, bbox2


    def fingers_up(self):
        fingers = []
        # Thumb
        if self.lmList[self.tipIds[0]][1] < self.lmList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        # 4 Fingers
        for id in range(1, 5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers

    def find_distance(self, p1, p2, img, draw=True):
        x1, y1 = self.lmList[p1][1], self.lmList[p1][2]  # Thumb tip coord
        x2, y2 = self.lmList[p2][1], self.lmList[p2][2]  # Index tip coord
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        
        if draw:
            cv2.circle(img, (x1, y1), 10, (128, 255, 0), cv2.FILLED)
            cv2.circle(img, (x2, y2), 10, (128, 255, 0), cv2.FILLED)  
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)  
            cv2.circle(img, (cx, cy), 10, (0, 0, 255), cv2.FILLED)
            
            
        length = math.hypot(x2 - x1, y2 - y1)
        #********************************
        # print(length)
        if length <= 50:
            cv2.circle(img, (cx, cy), 10, (128, 255, 0), cv2.FILLED)
        
        return length, img, [x1, y1, x2, y2, cx, cy]