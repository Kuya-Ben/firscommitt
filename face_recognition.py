"""
face_recognition.py
===================
The recognition engine of the project.

It loads the face encodings created by train_faces.py and answers the question:
    "Whose face is this?"

HOW RECOGNITION WORKS (short version)
-------------------------------------
1. Haar Cascade (in main.py) finds WHERE the faces are.
2. The dlib deep-learning model (inside the `face_recognition` library) turns
   each face into a list of 128 numbers called an "encoding" (a face fingerprint).
3. We compare that fingerprint with every stored fingerprint using Euclidean
   distance. The smallest distance wins. If it is below MATCH_TOLERANCE the
   person is recognised, otherwise the face is "Unknown".

NOTE ON THE FILE NAME
---------------------
This file is named face_recognition.py (as required by the assignment) which is
also the name of the third-party library. We therefore load the library through
utils.get_face_recognition_lib() to avoid Python importing this file by mistake.
"""

import math
import os
import pickle
from collections import namedtuple

import numpy as np

import utils

# The real dlib-based library (see utils.get_face_recognition_lib for details).
fr = utils.get_face_recognition_lib()

# One result per detected face.
#   name       : person's name or "Unknown"
#   confidence : 0.0 - 1.0 score derived from the distance (higher = more certain)
#   distance   : raw face distance (lower = more similar)
RecognitionResult = namedtuple("RecognitionResult", ["name", "confidence", "distance"])


def distance_to_confidence(distance, tolerance=utils.MATCH_TOLERANCE):
    """
    Convert a face distance into a 0-1 "confidence" score.

    This is the conversion suggested in the face_recognition documentation.
    It is a similarity score, NOT a true probability. At distance == tolerance
    the score is 0.5; distance 0 gives 1.0.
    """
    face_range = 1.0 - tolerance
    linear = (1.0 - distance) / (face_range * 2.0)
    if distance > tolerance:
        return max(0.0, min(1.0, linear))
    value = 1.0 - (distance / (face_range * 2.0))
    value = value + ((1.0 - value) * math.pow((value - 0.5) * 2.0, 0.2))
    return max(0.0, min(1.0, value))


class FaceRecognizer:
    """Loads stored encodings and identifies faces found in a frame."""

    def __init__(self, encodings_path=utils.ENCODINGS_PATH, tolerance=utils.MATCH_TOLERANCE):
        self.tolerance = tolerance
        self.known_names = []
        self.known_encodings = np.empty((0, 128))
        self.load(encodings_path)

    # ------------------------------------------------------------------ loading
    def load(self, encodings_path):
        """Read the pickle file created by train_faces.py."""
        if not os.path.exists(encodings_path):
            raise FileNotFoundError(
                "No trained data found (models/encodings.pickle).\n"
                "  1) Register faces:  python capture_faces.py\n"
                "  2) Train:           python train_faces.py"
            )
        with open(encodings_path, "rb") as f:
            data = pickle.load(f)
        self.known_names = list(data["names"])
        self.known_encodings = np.array(data["encodings"])
        print(f"[INFO] Loaded {len(self.known_names)} face samples "
              f"for {len(self.registered_names)} registered people.")

    @property
    def registered_names(self):
        """Sorted list of the unique registered people."""
        return sorted(set(self.known_names))

    # -------------------------------------------------------------- recognition
    def recognize(self, rgb_frame, locations):
        """
        Identify every face.

        rgb_frame : full-size frame in RGB order (use utils.bgr_to_rgb first)
        locations : list of (top, right, bottom, left) face boxes
        returns   : list of RecognitionResult in the same order as `locations`
        """
        if not locations:
            return []

        # Step 1: compute the 128-number fingerprint of every face.
        # We pass the boxes found by Haar so dlib does NOT search again.
        encodings = fr.face_encodings(rgb_frame, known_face_locations=locations, num_jitters=1)

        results = []
        for location_index in range(len(locations)):
            # face_encodings can return fewer items if a box was unusable.
            if location_index >= len(encodings) or len(self.known_names) == 0:
                results.append(RecognitionResult("Unknown", 0.0, 1.0))
                continue

            encoding = encodings[location_index]

            # Step 2: distance to every stored sample.
            distances = fr.face_distance(self.known_encodings, encoding)

            # Step 3: best (smallest) distance for each PERSON.
            best_name, best_distance = "Unknown", 1.0
            for person in self.registered_names:
                mask = np.array([n == person for n in self.known_names])
                person_best = float(np.min(distances[mask]))
                if person_best < best_distance:
                    best_name, best_distance = person, person_best

            # Step 4: accept only if the match is close enough.
            if best_distance <= self.tolerance:
                confidence = distance_to_confidence(best_distance, self.tolerance)
                results.append(RecognitionResult(best_name, confidence, best_distance))
            else:
                results.append(RecognitionResult("Unknown", 0.0, best_distance))
        return results


if __name__ == "__main__":
    # Quick self-test:  python face_recognition.py
    recognizer = FaceRecognizer()
    print("Registered people:", ", ".join(recognizer.registered_names))
