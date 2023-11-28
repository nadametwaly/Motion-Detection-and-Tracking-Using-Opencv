import numpy as np
import cv2

cap=cv2.VideoCapture("vtest.avi")
frame_width=int(cap.get((cv2.CAP_PROP_FRAME_WIDTH)))
frame_height=int(cap.get((cv2.CAP_PROP_FRAME_HEIGHT)))


fourcc=cv2.VideoWriter_fourcc('X','V','I','D')
out=cv2.VideoWriter("output.avi",fourcc,30,(1280,720))

ret, frame1=cap.read()
ret, frame2=cap.read()
print(frame1.shape)

while cap.isOpened():
    diff=cv2.absdiff(frame1,frame2)
    gray=cv2.cvtColor(diff,cv2.COLOR_BGR2GRAY)
    blur=cv2.GaussianBlur(gray,(5,5),0)
    _, thres=cv2.threshold(blur,60,255,cv2.THRESH_BINARY)
    dilate=cv2.dilate(thres,None,iterations=10)
    contours,__=cv2.findContours(dilate,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    #draw=cv2.drawContours(frame1,contours,-1,(0,255,0),2)
    
    for contour in contours:
        (x,y,w,h)=cv2.boundingRect(contour)
        
        if cv2.contourArea(contour)<900:
            continue
        cv2.rectangle(frame1,(x,y),(x+w,y+h),(0,255,0),2) #thickness=2
        cv2.putText(frame1, "Status: {}".format('Movement'),(10,20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)
        
        image=cv2.resize(frame1,(1280,720))
        out.write(image)
        cv2.imshow("motion detection", frame1)
        
    frame1=frame2
    ret,frame2=cap.read()
    if cv2.waitKey(40) == 27:
        break 
    
cv2.waitKey(0)
cv2.destroyAllWindows()
