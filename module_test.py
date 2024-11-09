import cv2 
import mediapipe as mp
import time
import math

#? intialize webcam capture function
cap = cv2.VideoCapture(0)

#? create an instance of hand's module
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

#* read the webcam value + convert the image from BGR to RGB
while True:
  success, img = cap.read()
  if not success:
        continue
  
  imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  results = hands.process(imgRGB)
  
  xList = []
  yList = []
  lmList = []
  
  #! print(reults.multi_hand_landmarks)
  if results.multi_hand_landmarks:
    for handLms in results.multi_hand_landmarks:
        print(handLms)
        #? Draw hand landmarks
        mp_drawing.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)
        for id, lm in enumerate(handLms.landmark):
          print(id,lm)
          ih, iw, ic = img.shape
          x,y = int(lm.x*iw), int(lm.y*ih)
          print(id, x, y)
        
  #? Display the image
  cv2.imshow("Hand Tracking", img)
    
  #? Break the loop if 'q' is pressed  
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

#* Release the webcam and destroy windows 
cap.release()
cv2.destroyAllWindows()