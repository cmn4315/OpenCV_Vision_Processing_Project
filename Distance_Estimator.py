# Python code for Multiple Color Detection 
  
  
import numpy as np 
import math
import cv2 
  
  
# Capturing video through webcam 
webcam = cv2.VideoCapture(0) 
  
# Start a while loop 
while(1): 
    #Define angle (in degrees) per pixel
    app = 0.118

    # Reading the video from the 
    # webcam in image frames 
    _, imageFrame = webcam.read() 
  
    # Convert the imageFrame in  
    # BGR(RGB color space) to  
    # HSV(hue-saturation-value) 
    # color space 
    hsvFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV) 
  
    # Set range for red color and  
    # define mask 
    red_lower = np.array([136, 87, 111], np.uint8) 
    red_upper = np.array([180, 255, 255], np.uint8) 
    red_mask = cv2.inRange(hsvFrame, red_lower, red_upper) 
  
    # Set range for green color and  
    # define mask 
    green_lower = np.array([25, 52, 72], np.uint8) 
    green_upper = np.array([102, 255, 255], np.uint8) 
    green_mask = cv2.inRange(hsvFrame, green_lower, green_upper) 
  
    # Set range for blue color and 
    # define mask 
    blue_lower = np.array([94, 80, 2], np.uint8) 
    blue_upper = np.array([120, 255, 255], np.uint8) 
    blue_mask = cv2.inRange(hsvFrame, blue_lower, blue_upper) 
      
    # Morphological Transform, Dilation 
    # for each color and bitwise_and operator 
    # between imageFrame and mask determines 
    # to detect only that particular color 
    kernal = np.ones((5, 5), "uint8") 
      
    # For red color 
    red_mask = cv2.dilate(red_mask, kernal) 
    res_red = cv2.bitwise_and(imageFrame, imageFrame,  
                              mask = red_mask) 
      
    # For green color 
    green_mask = cv2.dilate(green_mask, kernal) 
    res_green = cv2.bitwise_and(imageFrame, imageFrame, 
                                mask = green_mask) 
      
    # For blue color 
    blue_mask = cv2.dilate(blue_mask, kernal) 
    res_blue = cv2.bitwise_and(imageFrame, imageFrame, 
                               mask = blue_mask) 
   
    # Creating contour to track red color 
    rcontours, hierarchy = cv2.findContours(red_mask, 
                                           cv2.RETR_TREE, 
                                           cv2.CHAIN_APPROX_SIMPLE) 
    # Creating contour to track blue color 
    bcontours, hierarchy = cv2.findContours(blue_mask, 
                                           cv2.RETR_TREE, 
                                           cv2.CHAIN_APPROX_SIMPLE) 
    # Creating contour to track green color 
    gcontours, hierarchy = cv2.findContours(green_mask, 
                                           cv2.RETR_TREE, 
                                           cv2.CHAIN_APPROX_SIMPLE) 
      
    for pic, gcontour in enumerate(gcontours):
        for pic, bcontour in enumerate(bcontours):

            area = cv2.contourArea(gcontour) 
            area2 = cv2.contourArea(bcontour) 
            if(area > 300 and area2 > 300): 
                gx, gy, gw, gh = cv2.boundingRect(gcontour) 
                imageFrame = cv2.rectangle(imageFrame, (gx, gy),  
                                           (gx + gw, gy + gh), 
                                           (0, 255, 0), 2) 
           
                cv2.putText(imageFrame, "Green Colour", (gx, gy),
                            cv2.FONT_HERSHEY_SIMPLEX,  
                            1.0, (0, 255, 0)) 

                greenX = int(gx + (gw / 2.0))
                greenY = int(gy + (gh / 2.0))  
                cv2.putText(imageFrame, ".", (greenX, greenY), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, 
                            (0, 255, 0))
                if (greenX >= 320):
                    greenX = greenX - 320
                elif (greenX < 320):
                    greenX = -greenX + 320
                #print (greenX) 
                ga = math.radians(greenX*app)
                #print(ga)
   

            
                bx, by, bw, bh = cv2.boundingRect(bcontour) 
                imageFrame = cv2.rectangle(imageFrame, (bx, by), 
                                           (bx + bw, by + bh), 
                                           (255, 0, 0), 2) 
             
                cv2.putText(imageFrame, "Blue Colour", (bx, by), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            1.0, (255, 0, 0)) 

                blueX = int(bx + (bw / 2.0))
                blueY = int(by + (bh / 2.0))  
                cv2.putText(imageFrame, ".", (blueX, blueY), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, 
                            (255, 0, 0))
                if (blueX >= 320):
                    blueX = blueX - 320
                elif (blueX < 320):
                    blueX = -blueX + 320
                #print (blueX) 
                ba = math.radians(blueX*app)
                #print(ba)

                d = 5/(math.tan(ba) + math.tan(ga))
                print(d)


                

         
    # Program Termination 
    cv2.imshow("Multiple Color Detection in Real-Time", imageFrame) 
    if cv2.waitKey(27) & 0xFF == ord('q'): 
        cap.release() 
        cv2.destroyAllWindows() 
        break