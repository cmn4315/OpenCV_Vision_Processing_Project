import cv2
import imutils
import numpy as np

cap = cv2.VideoCapture(1)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=2.25, fy=1.75, interpolation=cv2.INTER_AREA)
    
    # convert the frame to grayscale, blur it, and threshold it
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5,5), 0)
    thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]

    # extract contours from the webcam frame
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # loop over the contours and draw them on the webcam frame
    for c in cnts:
        cv2.drawContours(frame, [c], -1, (0, 0, 255), 2)

    # display the total number of shapes on the image
    text = "I found {} total shapes".format(len(cnts))
    cv2.putText(frame, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # define the list of boundaries
    boundaries = [
        ([17, 15, 100], [50, 56, 200]),
        #([86, 31, 4], [220, 88, 50]),
        #([25, 146, 190], [62, 174, 250]),
        #([103, 86, 65], [145, 133, 128])
    ]

    # loop over the boundaries
    for (lower, upper) in boundaries:
        # create NumPy arrays from the boundaries
        lower = np.array(lower, dtype = "uint8")
        upper = np.array(upper, dtype = "uint8")
        # find the colors within the specified boundaries and apply
        # the mask
        mask = cv2.inRange(frame, lower, upper)
        output = cv2.bitwise_and(frame, frame, mask = mask)


    cv2.imshow("Window", frame)

    a = cv2.waitKey(1)
    if a == 27:
        break

cap.release()
cv2.destroyAllWindows()
