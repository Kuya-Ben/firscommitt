"""
train_faces.py
==============
STEP 2 - "TRAIN" (ENROLL) THE REGISTERED FACES.

Reads every photo in  dataset/<person>/  , converts each face to a 128-number
encoding with the face_recognition library and saves all encodings to
models/encodings.pickle .

IMPORTANT: this is NOT training a neural network from scratch. The library
already contains a deep model (a dlib ResNet) that was pre-trained on millions
of faces. "Training" here simply means computing and storing the encodings of
OUR people so main.py can compare live faces against them.

Usage:
    python train_faces.py
"""

import os
import pickle

import cv2

import utils

fr = utils.get_face_recognition_lib()

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")
MAX_IMAGE_WIDTH = 800      # shrink very large photos to keep training fast


def encode_image(image_path):
    """
    Return the 128-number encoding of the face in one image, or None if no
    face could be processed.
    """
    bgr = cv2.imread(image_path)
    if bgr is None:
        return None                                    # unreadable file

    # Image operation: shrink huge images (speed).
    if bgr.shape[1] > MAX_IMAGE_WIDTH:
        scale = MAX_IMAGE_WIDTH / bgr.shape[1]
        bgr = utils.resize_frame(bgr, scale)

    # Image operation: BGR -> RGB (the library expects RGB).
    rgb = utils.bgr_to_rgb(bgr)

    # Find the face inside the saved photo.
    locations = fr.face_locations(rgb, number_of_times_to_upsample=1, model="hog")

    if len(locations) == 0:
        # Our photos are already face crops, so if the detector fails we
        # simply treat the WHOLE image as the face.
        height, width = rgb.shape[:2]
        locations = [(0, width, height, 0)]
    elif len(locations) > 1:
        # Keep the biggest face only.
        locations = [max(locations, key=lambda b: (b[2] - b[0]) * (b[1] - b[3]))]

    encodings = fr.face_encodings(rgb, known_face_locations=locations, num_jitters=1)
    return encodings[0] if encodings else None


def main():
    utils.ensure_folders()

    # Every sub-folder of dataset/ is one person.
    people = sorted(d for d in os.listdir(utils.DATASET_DIR)
                    if os.path.isdir(os.path.join(utils.DATASET_DIR, d)))
    if not people:
        print("[ERROR] dataset/ is empty. Run  python capture_faces.py  first.")
        return

    known_encodings, known_names = [], []
    summary = []

    for folder in people:
        display_name = utils.folder_to_display_name(folder)
        person_dir = os.path.join(utils.DATASET_DIR, folder)
        images = sorted(f for f in os.listdir(person_dir) if f.lower().endswith(IMAGE_EXTENSIONS))

        used = 0
        for file_name in images:
            encoding = encode_image(os.path.join(person_dir, file_name))
            if encoding is None:
                print(f"  [SKIP] {folder}/{file_name}")
                continue
            known_encodings.append(encoding)
            known_names.append(display_name)
            used += 1

        summary.append((display_name, len(images), used))
        print(f"[OK] {display_name}: {used}/{len(images)} photos encoded")

    if not known_encodings:
        print("[ERROR] No faces could be encoded. Capture clearer photos and retry.")
        return

    # Save names + encodings together in one file.
    with open(utils.ENCODINGS_PATH, "wb") as f:
        pickle.dump({"names": known_names, "encodings": known_encodings}, f)

    print("\n" + "=" * 46)
    print(f"{'Person':<22}{'Photos':>8}{'Encoded':>10}")
    print("-" * 46)
    for name, total, used in summary:
        print(f"{name:<22}{total:>8}{used:>10}")
    print("=" * 46)
    print(f"Saved {len(known_encodings)} encodings for {len(summary)} people "
          f"-> {utils.ENCODINGS_PATH}")
    print("Next step: run  python main.py")


if __name__ == "__main__":
    main()
