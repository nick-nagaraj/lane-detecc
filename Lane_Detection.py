import numpy as np
import cv2

cap = cv2.VideoCapture('Test.mp4')
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))


while (True):
    #Read the frame
    ret, frame = cap.read()
    #Create a black array for the lines of the boundaries to be written on to
    final_right = np.zeros((480,640,3),np.uint8)
    final_left = np.zeros((480,640,3),np.uint8)

    #Define boundary colors to look for
    lower_color = np.uint8([[[140,140,140]]])
    upper_color = np.uint8([[[220,220,220]]])

    #Extract range of defined colors
    mask = cv2.inRange(frame, lower_color, upper_color)

    #Applying filter to remove noise
    frame = cv2.GaussianBlur(frame,(5,5),0)

    #Combining mask and frame
    res = cv2.bitwise_and(frame,frame, mask= mask)

    #Applying Edge Detection
    res = cv2.Canny(mask, 100,150, 3)

    #Removing upper portion of the input video to remove unwanted features
    cv2.rectangle(res, (0,0), (640, 200), (0,0,0), -1)

    #Applying dilation to exaggerate the extracted boundary points
    kernel = np.ones((5,5), np.uint8)
    dilation = cv2.dilate(res, kernel, iterations = 1)

    #Dividing dilation into left and right so as to apply line detection HoughLines on both sides simultaneously
    dilation_left = dilation.copy()
    dilation_right = dilation.copy()
    cv2.rectangle(dilation_left, (350,0),(640,480), (0,0,0), -1)
    cv2.rectangle(dilation_right, (0,0), (320,480), (0,0,0), -1)

    #Applying line detection using cv2.HoughLines for right side.
    lines = cv2.HoughLines(dilation_right,12,np.pi/180,15)
    for rho,theta in lines[0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        cv2.line(final_right,(x1,y1),(x2,y2),(0,0,255),5)

    cv2.rectangle(final_right, (0,0), (640, 200), (0,0,0), -1)

    #Applying line detection using cv2.HoughLines for left side
    lines = cv2.HoughLines(dilation_left,12,np.pi/180,15)
    for rho,theta in lines[0]:
         a = np.cos(theta)
         b = np.sin(theta)
         x0 = a*rho
         y0 = b*rho
         x1 = int(x0 + 1000*(-b))
         y1 = int(y0 + 1000*(a))
         x2 = int(x0 - 1000*(-b))
         y2 = int(y0 - 1000*(a))
         cv2.line(final_left,(x1,y1),(x2,y2),(0,0,255),5)

    cv2.rectangle(final_left, (0,0), (640, 200), (0,0,0), -1)

    #Adding left and right sides of written boundaries.
    final = cv2.bitwise_or(final_left, final_right, final_right)
    #Superimposing the written lines onto the video.
    final = cv2.bitwise_or(final_right, frame, frame)

    #Display frame and the dilated boundary points
    cv2.imshow('Final', frame)
    cv2.imshow('dilation', dilation)

    out.write(frame)

    k = cv2.waitKey(1)
    if k == 27:
        break
cap.release()
out.release()
cv2.destroyAllWindows()
