import cv2
import numpy as np


##Capturing the video 
cap = cv2.VideoCapture('video.mp4')

##For drawing the line
count_line_position = 550

##Min. width and Min.height of bounding box around vehicle
min_width_rect = 80
min_height_rect = 80


##Initializing the Subtractor algorithm, which extracts vehicles from the background
algo = cv2.bgsegm.createBackgroundSubtractorMOG()


def centre_handle(x,y,w,h):
    x1 = int(w/2)
    y1 = int(h/2)
    cx = x+x1
    cy = y+y1

    return cx,cy

##For counting the number of vehicles
detect = [] 

offset = 6

counter = 0

while True:
    ret, frame1 = cap.read()

    ##Creating each frame of the video to Grey from RGB
    grey = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)

    ##Smoothing/blurring/removing noise whereever possible
    blur = cv2.GaussianBlur(grey, (3,3), 5)

    ##Applying the Subtractor algorithm to each and every frame of the video
    img_sub = algo.apply(blur)


    ##Applying Dilation on each and every frame
    dilat = cv2.dilate(img_sub, np.ones((5,5)))

    ##Applying morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))

    ##Applying advanced morphological transformations (assigns shapes)
    dilatada = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, kernel)

    dilatada = cv2.morphologyEx(dilatada, cv2.MORPH_CLOSE, (5,5))

    ##Finding contours
    contour_shape, h = cv2.findContours(dilatada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    ##Drawing the line, if the vehicle crosses this line vehicle_count will be incremented
    cv2.line(frame1, (25, count_line_position), (1200, count_line_position), (255,127,0), 3)


    ##Drawing bounding boxes around each vehicle
    for (i,c) in enumerate(contour_shape):
        (x,y,w,h) = cv2.boundingRect(c)
        validate_contour = (w >= min_width_rect) and (h >= min_height_rect)

        if not validate_contour:  ##if rectange doesn't satisfy height and width contions, just continue
            continue

        cv2.rectangle(frame1, (x,y), (x+w, y+h), (0,0,255), 2)

        centre = centre_handle(x,y,w,h)
        detect.append((centre))
        cv2.circle(frame1, centre, 4, (0,0,255), -1)


        for (x,y) in detect:
            if(y < count_line_position + offset) and (y > count_line_position - offset) :
                counter = counter + 1

            ##changing the colour of line, when counter is incremented
            cv2.line(frame1, (25, count_line_position), (1200, count_line_position), (0,0,0), 3)
            detect.remove((x,y)) ##remove this new colour of line
            #print("Vehicle COunter: "+ str(counter))

    cv2.putText(frame1, "VEHICLE COUNTER: "+ str(counter), (450,70), cv2.FONT_HERSHEY_COMPLEX, 2, (0.0,255), 5)



    cv2.imshow('Original Video',frame1)

    #cv2.imshow('Detector Video', dilatada)
   

    if cv2.waitKey(1) == 13:
        break

cv2.destroyAllWindows()
cap.release()