# Python file for capturing dataset images using laptop's camera
import os
import cv2

cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)
count = 1
#Select the category for which we wnat to capture the images
folder = str(input("Enter the category for which you want to collect images\n1 rock\n2 empty\n3 scissors\n4 paper\n"))

while(True):
     ret,frame = cap.read()
     print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
     print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
     # Defining capture region so that the actions can be captured clearly 
     frame = cv2.rectangle(frame,(19,39),(492,392),(0,0,255),2)
     font = cv2.FONT_HERSHEY_SIMPLEX
     # Counting the number of images that has been captured
     frame = cv2.putText(frame,"Images Captured =>{}".format(count),(70,450),font,1,(255, 0, 0),2)
     frame = cv2.putText(frame,"Collecting Images for "+folder+" category",(10,490),font,1,(255,255,255),2)
     frame = cv2.putText(frame,"Press 'q' to exit the process and save images",(300,600),font,1,(1, 255, 0),4)
     cv2.imshow('Images',frame)
     #Cropping the reuired region out of selected images
     start_row,start_col = int(43),int(21)
     end_row,end_col = int(390),int(490)
     frame = frame[start_row:end_row,start_col:end_col]
     #Updating the paths where the images need to be stored
     cv2.imwrite('C://Users//Manish Sehrawat//Downloads//training_data//'+folder+'//image'+str(count)+'.jpg',frame)
     #Press Q to exit the process in between
     count = count + 1
     if count > 400:
         break
     if cv2.waitKey(10) & 0xFF == ord('q'):
         break
cap.release()
cv2.destroyAllWindows()

     
