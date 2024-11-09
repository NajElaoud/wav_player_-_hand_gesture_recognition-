'''
RGB color codes: 
  black   # 0, 0, 0      red     # 0, 0, 255
  #!green   # 0, 255, 0    # 128, 255, 0
  yellow  # 0, 255, 255
  #!blue    # 255, 0, 0    # 255, 128, 0
  #!magenta # 255, 0, 255  # 128, 0, 255
  violet # 255, 0, 128   orange  # 0, 128, 255
  cyan    # 255, 255, 0  white   # 255, 255, 255
'''

import cv2 
import time
import math
import numpy as np
import mediapipe as mp
import handTrackingModule as hTM

#! pycaw library by AndreMiras 
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

wCam, hCam = 640, 480

#? intialize webcam capture function
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(3,wCam)
cap.set(4,hCam)
pTime = 0
last_print_time = time.time()
print_interval = 0.2  # 200 ms interval

detector = hTM.hand_detector(detectionCon = 0.7, maxHands = 2)


try:
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = interface.QueryInterface(IAudioEndpointVolume)
except Exception as e:
    print(f"Error initializing audio interface: {e}")
#volume.GetMute()
#volume.GetMasterVolumeLevel()
volume_rangge = volume.GetVolumeRange()
#volume.SetMasterVolumeLevel(-20.0, None)
minvol = volume_rangge[0]   # Minimum volume level
maxvol = volume_rangge[1]   # Maximum volume level
vol = 0
volBar = 400
volPer = 0

while True:
  success, img = cap.read()
  if not success: 
    continue
  
  img = cv2.flip(img, 1)
      
  img = detector.find_hands(img)
  hands, lmList, bbox1, bbox2 = detector.find_position(img, draw = True)
  
  if len(lmList) != 0:
    current_time = time.time()
    if current_time - last_print_time >= print_interval:
      #***************************************
      # print(lmList[4], lmList[8])  
      last_print_time = current_time  # Update the last print time
    
    #! filter distance based on size
    box_width, box_height = bbox1[2]-bbox1[0], bbox1[3]-bbox1[1]
    box_area = ~(box_height * box_width) //100
    #********************************
    print("bbox= ",box_area)
    
    if 150 < box_area < 1000:
      #********************************
      # print("distance OK ")
      
      #! find distance between index and thumb
      length, img, line_info = detector.find_distance(4, 8, img)    
      #********************************
      # print(length)
      
      #! convert volume 
      # range from 50 to 200
      # volume range from -65 to 0
      #vol = np.interp(length, [50, 200], [minvol, maxvol])
      volBar = np.interp(length, [50, 200], [400, 150])
      volPer = np.interp(length, [50, 200], [0, 100])
      #********************************
      # print(int(length), vol)
      #volume.SetMasterVolumeLevel(vol, None)
      
      #! reduce resolution
      smoothness = 5 # volume changes by 5
      volPer = smoothness * round(volPer / smoothness) 
      
      #! check if fingers are up
      fingers = detector.fingers_up()
      #********************************
      # print(fingers)
      
      #! check if pinky is down (set volume)
      if fingers[4]  == True:
        volume.SetMasterVolumeLevelScalar(volPer/100, None) # smoother 
        cv2.circle(img, (line_info[4], line_info[5]), 10, (128, 255, 0), cv2.FILLED)
        cv2.putText(img, f'can change volume', (450, 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 128, 0), 1)
      else:
        cv2.putText(img, f'can\'t change volume', (450, 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
      
      #! drawing 
      if volPer < -30:  
        # volume bar
        cv2.rectangle(img, (50, 150), (85,400), (255, 128, 0), 3)
        cv2.rectangle(img, (50, int(volBar)), (85,400), (255, 128, 0), cv2.FILLED)
        # volume percentage
        cv2.putText(img, f'{int(volPer)} %', (55, 420), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 128, 0), 2)
      elif -30 < volPer < -80:
        # volume bar
        cv2.rectangle(img, (50, 150), (85,400), (128, 255, 0), 3)
        cv2.rectangle(img, (50, int(volBar)), (85,400), (128, 255, 0), cv2.FILLED)
        # volume percentage
        cv2.putText(img, f'{int(volPer)} %', (55, 420), cv2.FONT_HERSHEY_COMPLEX, 0.5, (128, 255, 0), 2)
      else:
        # volume bar
        cv2.rectangle(img, (50, 150), (85,400), (128, 0, 255), 3)
        cv2.rectangle(img, (50, int(volBar)), (85,400), (128, 0, 255), cv2.FILLED)
        # volume percentage
        cv2.putText(img, f'{int(volPer)} %', (55, 420), cv2.FONT_HERSHEY_COMPLEX, 0.5, (128, 0, 255), 2)
        
  
  #! frame rate
  cTime = time.time()
  fps = 1 / (cTime - pTime) if cTime - pTime > 0 else 0
  pTime = cTime
  # FPS meter
  cv2.putText(img, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 128, 0), 2)
  
  #? Display the image
  cv2.imshow("Hand Tracking", img)
    
  #? Break the loop if 'q' is pressed  
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break
  

#* Release the webcam and destroy windows 
cap.release()
cv2.destroyAllWindows()