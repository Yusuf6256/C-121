import cv2
import time
from cv2 import bitwise_not
import numpy as np

#to save the output in a file output.AVI
fourcc=cv2.VideoWriter_fourcc(*'XVID')

output_file=cv2.VideoWriter('Output.avi',fourcc,20.0,(1920,1080))

#Starting the webcam
cap=cv2.VideoCapture(0)

#allowing the webcam to start by making the code sleep for 2 seconds
time.sleep(2)

bg=0
#capturing the background for 60 frames
for i in range (60):
    ret,bg=cap.read()

#fliping the background
bg = np.flip(bg,axis=1)

#reading the captured fram until the camera is opened
while(cap.isOpened()):
    ret,img=cap.read()
    if not ret:
        break
    #flipping the image for consistency
    img=np.flip(img,axis=1)
    #coberting the colour from rgb to hsv
    hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    #generating mass to detect the red color
    lower_red=np.array([0,120,50])
    upper_red=np.array([10,255,255])

    mask_1=cv2.inRange(hsv,lower_red,upper_red)
    lower_red=np.array([170,120,70])
    upper_red=np.array([180,255,255])
    mask_2=cv2.inRange(hsv,lower_red,upper_red)
    mask_1=mask_1 + mask_2
    #open and expand image where there is mask 1
    mask_1 = cv2.morphologyEx(mask_1, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) 
    mask_1 = cv2.morphologyEx(mask_1, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8))
    #selecting only the part that doesn't have mask 1 and saving in mask 2
    mask_2=bitwise_not(mask_1)
    #keeping only the part of the images without red colour
    ras_1=cv2.bitwise_and(img,img,mask=mask_2)
    #keeping only the part of the images with read
    ras_2=cv2.bitwise_and(bg,bg,mask=mask_1)
    #generating the final output by merging ras_1 and ras_2
    final_otput=cv2.addWeighted(ras_1,1,ras_2,1,0)
    output_file.write(final_otput)

    #displaying the output to the user
    cv2.imshow('magic',final_otput)
    cv2.waitKey(1)

cap.release()
out.release()
cv2.destroyAllWindows()
