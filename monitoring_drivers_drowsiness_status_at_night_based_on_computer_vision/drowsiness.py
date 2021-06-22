# import the necessary packages
import face_utils
import numpy as np
import argparse
import dlib
import cv2

class Drowsiness:
    def __init__(self):
        self.img = None
        self.rect = None
        self.shape = None

        # initialize dlib's face detector (HOG-based) and then create
        # the facial landmark predictor
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    def set_image(self, img):
        self.img = img

    def get_image(self):
        return self.img

    def apply_calhe(self):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        self.img = clahe.apply(self.img)

    def detect_face(self):
        rects = self.detector(self.img, 1)

        # This assumes that only driver face will be covered
        if (rects):
            self.rect = rects[0]
            shape = self. predictor(self.img, self.rect)
            self.shape = face_utils.shape_to_np(shape)

    def draw_landmarks(self):
        if (self.rect):
            (x, y, w, h) = face_utils.rect_to_bb(self.rect)
            for landmark_idx, (x, y) in enumerate(self.shape):
                # print(landmark_idx)
                cv2.circle(self.img, (x, y), 3, (255, 0, 0), 3)
                cv2.putText(self.img, str(landmark_idx), (x - 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    def detect_closed_eyes(self):
        # EAR
        if (self.shape is not None and self.shape.any()):
            # EAR RIGHT
            p2p6 = ((self.shape[41][0] - self.shape[37][0]) ** 2 + (self.shape[41][1] - self.shape[37][1]) ** 2) ** 1/2
            p3p5 = ((self.shape[38][0] - self.shape[40][0]) ** 2 + (self.shape[38][1] - self.shape[40][1]) ** 2) ** 1/2
            p1p4 = ((self.shape[36][0] - self.shape[39][0]) ** 2 + (self.shape[36][1] - self.shape[39][1]) ** 2) ** 1/2
            ear_right = (p2p6 + p3p5)/p1p4

            # EAR LEFT
            p2p6 = ((self.shape[43][0] - self.shape[47][0]) ** 2 + (self.shape[43][1] - self.shape[47][1]) ** 2) ** 1/2
            p3p5 = ((self.shape[44][0] - self.shape[46][0]) ** 2 + (self.shape[44][1] - self.shape[46][1]) ** 2) ** 1/2
            p1p4 = ((self.shape[42][0] - self.shape[45][0]) ** 2 + (self.shape[42][1] - self.shape[45][1]) ** 2) ** 1/2
            ear_left = (p2p6 + p3p5)/p1p4

            ear = (ear_left + ear_right)/2
            if (ear < 0.3):
                return True

        return False # no closure of eyes

    
    def detect_yawn(self):
        # MON
        if (self.shape is not None and self.shape.any()):
            # MOR
            p51p59 = ((self.shape[50][0] - self.shape[58][0]) ** 2 + (self.shape[50][1] - self.shape[58][1]) ** 2) ** 1/2
            p52p58 = ((self.shape[51][0] - self.shape[57][0]) ** 2 + (self.shape[51][1] - self.shape[57][1]) ** 2) ** 1/2
            p53p57 = ((self.shape[52][0] - self.shape[56][0]) ** 2 + (self.shape[52][1] - self.shape[56][1]) ** 2) ** 1/2
            p49p55 = ((self.shape[48][0] - self.shape[54][0]) ** 2 + (self.shape[48][1] - self.shape[54][1]) ** 2) ** 1/2

            mor = (p51p59 + p52p58 + p53p57)/p49p55
            if (mor > 0.6):
                return True

        return False # no yawn