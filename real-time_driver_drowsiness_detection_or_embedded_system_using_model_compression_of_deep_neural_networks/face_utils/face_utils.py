from mtcnn import MTCNN
import cv2

class FaceDetector():
    def __init__(self):
        pass
    
    def set_face(self, img):
        detections = MTCNN().detect_faces(img)
        
        img_with_dets = img.copy()
        min_conf = 0.9
   
        self.faces = []
        self.mouths = []
        self.left_eyes = []
        self.right_eyes = []
        for det in detections:
            if det['confidence'] >= min_conf:
                x, y, width, height = det['box']
                keypoints = det['keypoints']
                cv2.rectangle(img_with_dets, (x,y), (x+width,y+height), (0,155,255), 2)
                cv2.circle(img_with_dets, (keypoints['left_eye']), 2, (0,155,255), 2)
                cv2.circle(img_with_dets, (keypoints['right_eye']), 2, (0,155,255), 2)
                cv2.circle(img_with_dets, (keypoints['nose']), 2, (0,155,255), 2)
                cv2.circle(img_with_dets, (keypoints['mouth_left']), 2, (0,155,255), 2)
                cv2.circle(img_with_dets, (keypoints['mouth_right']), 2, (0,155,255), 2)

                to_cropped = img.copy()

                face = to_cropped[y:y+height, x:x+width]

                # crop eyes
                half_width = ((width * 0.3)/2)
                half_height = ((height * 0.3)/2)
                left_eye = to_cropped[
                    int(keypoints['left_eye'][1] - half_height):int(keypoints['left_eye'][1] + half_height),
                    int(keypoints['left_eye'][0] - half_width):int(keypoints['left_eye'][0] + half_width),
                ]
                right_eye = to_cropped[
                    int(keypoints['right_eye'][1] - half_height):int(keypoints['right_eye'][1] + half_height),
                    int(keypoints['right_eye'][0] - half_width):int(keypoints['right_eye'][0] + half_width),
                ]

                # crop mouth
                m_half_width = ((width * 0.3)/1.5)
                m_midpoint_x = (keypoints['mouth_left'][0] + keypoints['mouth_right'][0]) / 2
                m_midpoint_y = (keypoints['mouth_left'][1] + keypoints['mouth_right'][1]) / 2
                
                mouth = to_cropped[
                    int(m_midpoint_y - half_height):int(m_midpoint_y + half_height),
                    int(m_midpoint_x - m_half_width):int(m_midpoint_x + m_half_width),
                ]
                
                self.faces.append(face)
                self.left_eyes.append(left_eye)
                self.right_eyes.append(right_eye)
                self.mouths.append(mouth)
                
    def get_faces(self):
        return self.faces
    
    def get_left_eyes(self):
        return self.left_eyes
    
    def get_right_eyes(self):
        return self.right_eyes
    
    def get_mouths(self):
        return self.mouths