"""
capture_faces.py
================
STEP 1 - REGISTER A PERSON.

Opens the webcam, detects the face with a Haar Cascade and saves cropped face
photos into  dataset/<person_name>/  .

Usage:
    python capture_faces.py                       (asks for the name)
    python capture_faces.py --name "Juan Dela Cruz" --samples 25

Controls:
    SPACE : start / pause capturing
    q     : quit
"""

import argparse
import os
import time

import cv2

import utils

CAPTURE_INTERVAL = 0.6     # seconds between two saved photos (so poses differ)
MIN_SHARPNESS = 35.0       # photos blurrier than this are skipped (tune if needed)
CROP_MARGIN = 0.25         # keep 25% extra border around the face


def count_existing_images(person_dir):
    """How many photos does this person already have? (lets us add more later)"""
    return len([f for f in os.listdir(person_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))])


def crop_face(frame, location):
    """Cut the face (plus a small margin) out of the full-size frame."""
    top, right, bottom, left = location
    margin = int(CROP_MARGIN * max(right - left, bottom - top))
    h, w = frame.shape[:2]
    # max/min keep the crop inside the image borders.
    y1, y2 = max(0, top - margin), min(h, bottom + margin)
    x1, x2 = max(0, left - margin), min(w, right + margin)
    return frame[y1:y2, x1:x2]


def main():
    parser = argparse.ArgumentParser(description="Register a person's face.")
    parser.add_argument("--name", help="Person's name (asked interactively if omitted)")
    parser.add_argument("--samples", type=int, default=20, help="Number of photos to save")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index")
    args = parser.parse_args()

    # ---- 1. Decide the folder name -----------------------------------------
    raw_name = args.name or input("Enter the person's full name: ")
    folder_name = utils.sanitize_name(raw_name)
    if not folder_name:
        print("[ERROR] Invalid name. Use letters, numbers and spaces only.")
        return

    utils.ensure_folders()
    person_dir = os.path.join(utils.DATASET_DIR, folder_name)
    os.makedirs(person_dir, exist_ok=True)
    next_index = count_existing_images(person_dir) + 1

    # ---- 2. Load the Haar Cascade and open the webcam ----------------------
    cascade = utils.load_cascade()
    cap = utils.open_camera(args.camera)

    print(f"[INFO] Registering '{utils.folder_to_display_name(folder_name)}'.")
    print("[INFO] Press SPACE to start capturing. Slowly turn your head a little "
          "and change expression. Press 'q' to quit.")

    capturing = False
    saved = 0
    last_save_time = 0.0

    while saved < args.samples:
        ok, frame = cap.read()
        if not ok:
            print("[ERROR] Could not read from webcam.")
            break

        # ---- 3. Image operations: resize -> grayscale -> blur/equalise ------
        small = utils.resize_frame(frame, utils.FRAME_SCALE)
        gray_small = utils.to_gray(small)
        gray_small = utils.enhance_gray(gray_small)

        # ---- 4. Haar Cascade face detection --------------------------------
        boxes = utils.detect_faces_haar(cascade, gray_small)
        locations = utils.boxes_to_locations(boxes, utils.FRAME_SCALE)

        display = frame.copy()
        message, color = "No face detected", utils.RED

        if len(locations) > 1:
            message, color = "Only ONE person please", utils.ORANGE
        elif len(locations) == 1:
            top, right, bottom, left = locations[0]
            cv2.rectangle(display, (left, top), (right, bottom), utils.GREEN, 2)
            message, color = "Face detected", utils.GREEN

            if capturing and (time.time() - last_save_time) >= CAPTURE_INTERVAL:
                # Quality check: skip blurry photos.
                gray_full = utils.to_gray(frame)
                face_gray = gray_full[max(top, 0):bottom, max(left, 0):right]
                if face_gray.size == 0 or utils.sharpness(face_gray) < MIN_SHARPNESS:
                    message, color = "Too blurry - hold still", utils.ORANGE
                else:
                    face_img = crop_face(frame, locations[0])
                    file_name = f"{folder_name}_{next_index:03d}.jpg"
                    cv2.imwrite(os.path.join(person_dir, file_name), face_img)
                    next_index += 1
                    saved += 1
                    last_save_time = time.time()
                    print(f"[SAVED] {file_name}  ({saved}/{args.samples})")

        # ---- 5. On-screen information --------------------------------------
        state = "CAPTURING" if capturing else "PAUSED - press SPACE to start"
        utils.put_text_bg(display, message, (10, 30), scale=0.65, color=utils.WHITE, bg=color)
        utils.put_text_bg(display, f"Saved: {saved}/{args.samples}   [{state}]",
                          (10, 62), scale=0.55)
        # Progress bar
        bar_width = int(display.shape[1] * saved / max(args.samples, 1))
        cv2.rectangle(display, (0, display.shape[0] - 10), (bar_width, display.shape[0]),
                      utils.GREEN, -1)

        cv2.imshow("Register Face", display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" "):
            capturing = not capturing

    cap.release()
    cv2.destroyAllWindows()

    print(f"\n[DONE] Saved {saved} new photos in {person_dir}")
    print("Next step: run  python train_faces.py")


if __name__ == "__main__":
    main()
