from keras_video import VideoFrameGenerator
import os
import glob
import numpy as np
import cv2 as cv
from math import floor
import logging
import re

from face_utils.face_utils import FaceDetector



class Baseline4Generator(VideoFrameGenerator):
    """
    Create a generator that return batches of frames from video
    - rescale: float fraction to rescale pixel data (commonly 1/255.)
    - nb_frames: int, number of frames to return for each sequence
    - classes: list of str, classes to infer
    - batch_size: int, batch size for each loop
    - use_frame_cache: bool, use frame cache (may take a lot of memory for \
        large dataset)
    - shape: tuple, target size of the frames
    - shuffle: bool, randomize files
    - transformation: ImageDataGenerator with transformations
    - split: float, factor to split files and validation
    - nb_channel: int, 1 or 3, to get grayscaled or RGB images
    - glob_pattern: string, directory path with '{classname}' inside that \
        will be replaced by one of the class list
    - use_header: bool, default to True to use video header to read the \
        frame count if possible
    You may use the "classes" property to retrieve the class list afterward.
    The generator has that properties initialized:
    - classes_count: number of classes that the generator manages
    - files_count: number of video that the generator can provides
    - classes: the given class list
    - files: the full file list that the generator will use, this \
        is usefull if you want to remove some files that should not be \
        used by the generator.
    """

    def __init__(
            self,
            rescale=1/255.,
            nb_frames: int = 5,
            classes: list = None,
            batch_size: int = 16,
            use_frame_cache=False,
            target_shape: tuple = (224, 224),
            shuffle: bool = True,
            transformation=None,
            split_test: float = None,
            split_val: float = None,
            nb_channel=3,
            glob_pattern: str = './videos/{classname}/*.avi',
            use_headers: bool = True,
            *args,
            **kwargs):
        
        super().__init__(
            rescale,
            nb_frames,
            classes,
            batch_size,
            use_frame_cache=False,
            target_shape,
            shuffle,
            transformation=None,
            split_test,
            split_val,
            nb_channel=3,
            glob_pattern,
            use_headers,
            *args,
            **kwargs)
        
    def __getitem__(self, index):
        classes = self.classes
        shape = self.target_shape
        nbframe = self.nbframe

        labels = []
        face_images = []
        left_eyes_images = []
        right_eyes_images = []
        mouth_eyes_images = []

        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        transformation = None

        for i in indexes:
            # prepare a transformation if provided
            if self.transformation is not None:
                transformation = self._random_trans[i]

            video = self.files[i]
            classname = self._get_classname(video)

            # create a label array and set 1 to the right column
            label = np.zeros(len(classes))
            col = classes.index(classname)
            label[col] = 1.
            frames = self._get_frames(
                video,
                nbframe,
                shape,
                force_no_headers=not self.use_video_header)
            if frames[0] is None:
                # avoid failure, nevermind that video...
                continue

            # add the sequence in batch
            face_images.append(frames[0])
            left_eyes_images.append(frames[1])
            right_eyes_images.append(frames[2])
            mouth_eyes_images.append(frames[3])
            
            labels.append(label)

        return [
            np.array(face_images), 
            np.array(left_eyes_images), 
            np.array(right_eyes_images), 
            np.array(mouth_eyes_images)
        ], np.array(labels)
    
    
    def _get_frames(self, video, nbframe, shape, force_no_headers=False):
        print(video)
        cap = cv.VideoCapture(video)
        total_frames = self.count_frames(cap, video, force_no_headers)
        orig_total = total_frames
        if total_frames % 2 != 0:
            total_frames += 1
        frame_step = floor(total_frames/(nbframe-1))
        # TODO: fix that, a tiny video can have a frame_step that is
        # under 1
        frame_step = max(1, frame_step)
        face_frames = []
        left_eye_frames = []
        right_eye_frames = []
        mouth_eye_frames = []
        frame_i = 0
        
        # CLAHE
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        face_detector = FaceDetector()
        while True:
            grabbed, frame = cap.read()
            if not grabbed:
                break

            frame_i += 1
            if frame_i == 1 or frame_i % frame_step == 0 or frame_i == orig_total:
                # resize
                frame = cv.resize(frame, shape)
                frame = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
                
                frame = clahe.apply(frame)
                face_detector.set(frame)
                faces = face_detector.face_detector.get_faces()
                left_eyes = face_detector.get_left_eyes()
                right_eyes = face_detector.get_right_eyes()
                mouths = face_detector.get_mouths()
                
                # to np
                face_frame = img_to_array(faces[0]) * self.rescale
                left_eye_frame = img_to_array(left_eyes[0]) * self.rescale
                right_eye_frame = img_to_array(right_eyes[0]) * self.rescale
                mouth_eye_frame = img_to_array(mouths[0]) * self.rescale
                
                # keep frame
                face_frames.append(face_frame)
                left_eye_frames.append(left_eye_frame)
                right_eye_frames.append(right_eye_frame)
                mouth_eye_frames.append(mouth_eye_frame)

            if len(face_frames) == nbframe:
                break

        cap.release()

        if not force_no_headers and len(face_frames) != nbframe:
            # There is a problem here
            # That means that frame count in header is wrong or broken,
            # so we need to force the full read of video to get the right
            # frame counter
            return self._get_frames(
                    video,
                    nbframe,
                    shape,
                    force_no_headers=True)

        if force_no_headers and len(face_frames) != nbframe:
            # and if we really couldn't find the real frame counter
            # so we return None. Sorry, nothing can be done...
            log.error("Frame count is not OK for video %s, "
                      "%d total, %d extracted" % (
                        video, total_frames, len(face_frames)))
            return None

        return [
            np.array(face_frames), 
            np.array(left_eye_frames), 
            np.array(right_eye_frames), 
            np.array(mouth_eye_frames)
        ]