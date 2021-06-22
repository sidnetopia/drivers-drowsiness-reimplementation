import numpy as np
import cv2
from drowsiness import Drowsiness

cap = cv2.VideoCapture(0)
drowsiness_obj = Drowsiness()

closed_eyes_count = 0
yawn_count = 0
yawn_frequency = 0
frame_count = 1
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    gray_frame = gray_frame.reshape((480, 640, 1))
    print(gray_frame.shape[0])
    # Our operations on the frame come here
    drowsiness_obj.set_image(gray_frame)
    drowsiness_obj.apply_calhe()
    drowsiness_obj.detect_face()
    drowsiness_obj.draw_landmarks()
    
    # drowsiness detection
    if (drowsiness_obj.detect_closed_eyes()):
        closed_eyes_count += 1
    else:
        closed_eyes_count = 0
    
    if (drowsiness_obj.detect_yawn()):
        yawn_count += 1

    yawn_frequency = yawn_count / frame_count

    if (yawn_frequency >= .6 and closed_eyes_count >= 48):
        print("drowsy driver")
        yawn_frequency = 0
        frame_count = 0
        closed_eyes_count = 0
        
    # print("closed_eyes_count{}".format(closed_eyes_count))
    # print("yawn_count{}".format(yawn_count))
    # print("frame_count{}".format(frame_count))
    # print("yawn_frequency{}".format(yawn_frequency))

    frame_count += 1

    # Display the resulting frame
    cv2.imshow('frame',drowsiness_obj.get_image())
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()