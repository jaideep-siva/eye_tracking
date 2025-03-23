from __future__ import division
import cv2
import json
import os
from .pupil import Pupil


class Calibration(object):
    """
    This class calibrates the pupil detection algorithm by finding the
    best binarization threshold value for the person and the webcam.
    """

    def __init__(self):
        self.nb_frames = 20
        self.thresholds_left = []
        self.thresholds_right = []

    def is_complete(self):
        """Returns true if the calibration is completed"""
        return len(self.thresholds_left) >= self.nb_frames and len(self.thresholds_right) >= self.nb_frames

    def threshold(self, side):
        """Returns the threshold value for the given eye.

        Argument:
            side: Indicates whether it's the left eye (0) or the right eye (1)
        """
        if side == 0:
            return int(sum(self.thresholds_left) / len(self.thresholds_left)) if self.thresholds_left else 30
        elif side == 1:
            return int(sum(self.thresholds_right) / len(self.thresholds_right)) if self.thresholds_right else 30

    def reset(self):
        """Resets the calibration data"""
        self.thresholds_left = []
        self.thresholds_right = []

    def save(self, filename="calibration.json"):
        """Saves the calibration data to a file"""
        data = {
            "thresholds_left": self.thresholds_left,
            "thresholds_right": self.thresholds_right
        }
        with open(filename, "w") as f:
            json.dump(data, f)

    def load(self, filename="calibration.json"):
        """Loads the calibration data from a file"""
        if os.path.exists(filename):
            with open(filename, "r") as f:
                data = json.load(f)
                self.thresholds_left = data.get("thresholds_left", [])
                self.thresholds_right = data.get("thresholds_right", [])

    @staticmethod
    def iris_size(frame):
        """Returns the percentage of space that the iris takes up on
        the surface of the eye.

        Argument:
            frame (numpy.ndarray): Binarized iris frame
        """
        if frame is None or frame.size == 0:
            return 0.48  # Default value if frame is invalid

        frame = frame[5:-5, 5:-5]
        height, width = frame.shape[:2]
        nb_pixels = height * width
        nb_blacks = nb_pixels - cv2.countNonZero(frame)
        return nb_blacks / nb_pixels

    @staticmethod
    def find_best_threshold(eye_frame):
        """Calculates the optimal threshold to binarize the
        frame for the given eye.

        Argument:
            eye_frame (numpy.ndarray): Frame of the eye to be analyzed
        """
        if eye_frame is None or eye_frame.size == 0:
            return 30  # Default threshold if frame is invalid

        average_iris_size = 0.48
        trials = {}

        for threshold in range(5, 100, 10):  # Reduced step size for performance
            iris_frame = Pupil.image_processing(eye_frame, threshold)
            trials[threshold] = Calibration.iris_size(iris_frame)

        best_threshold, iris_size = min(trials.items(), key=(lambda p: abs(p[1] - average_iris_size)))
        return best_threshold

    def evaluate(self, eye_frame, side):
        """Improves calibration by taking into consideration the
        given image.

        Arguments:
            eye_frame (numpy.ndarray): Frame of the eye
            side: Indicates whether it's the left eye (0) or the right eye (1)
        """
        if eye_frame is None or eye_frame.size == 0:
            return

        threshold = self.find_best_threshold(eye_frame)

        if side == 0:
            self.thresholds_left.append(threshold)
        elif side == 1:
            self.thresholds_right.append(threshold)