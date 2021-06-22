# Monitoring-Drivers-Drowsiness

This is the reimplementation of the paper entitled "Monitoring Driver’s Drowsiness Status at Night Based on Computer Vision."

The main advantages of this impelentation are:
1. Easy to implement
2. Night based detection
3. Cheap in comparison with Haar based and Viola-Jones

Methodology:
1. Convert frame to grayscale using Contrast Limited Adaptive Histogram Equalization (CLAHE)
..* Frame is divided into 8x8 small blocks, seperately equalizing its histogram. It'll also restrict the noise values by limiting the histogram bin.
2. Facial landmarks
..* Used Dlib library that has trained two shape predictor models on the iBug 300-W dataset. It localizes 68 and 5 landmark points in the face within the frame. .
..* Based on based on Histogram of Oriented Gradients (HOG) and linear SVM classifier.
..* The model used is named `shape_predictor_68_face_landmarks.dat` in the file directory.
3. Cues extraction
..1. Calculate the Eyes aspect Ratio (EAR) and Mouth Opening Ratio using the landmarks taken from the classifier. 
..2. If the number of continuous frames in which eyes is closed and Yawn Frequency is greater than fixed predefined threshold (48 frames and 6 yawns respectively) the driver is in drowsiness state.


Reference:
V. Valsan A, P. P. Mathai and I. Babu, "Monitoring Driver’s Drowsiness Status at Night Based on Computer Vision," 2021 International Conference on Computing, Communication, and Intelligent Systems (ICCCIS), 2021, pp. 989-993, doi: 10.1109/ICCCIS51004.2021.9397180.
