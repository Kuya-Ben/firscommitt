"""
main.py
=======
STEP 3 - RUN THE LIVE ATTENDANCE SYSTEM.

Pipeline for every webcam frame:

    webcam frame
        |-> resize (50%)  -> grayscale -> motion detection   (bonus topic)
        |-> blur + histogram equalisation
        |-> Haar Cascade face detection
        |-> BGR -> RGB  -> face_recognition encodings -> identify person
        |-> confirm for CONFIRM_FRAMES frames -> write Attendance.csv (once per day)
        |-> draw boxes, names, status, date/time, FPS on the frame

Usage:
    python main.py
    python main.py --camera 1 --no-motion

Keys:  q = quit   s = save screenshot   m = motion detection on/off
"""

import argparse
import os
import time
from datetime import datetime

import cv2

import utils
from face_recognition import FaceRecognizer


def build_results(locations, recognition, today_records, confirm_counts):
    """
    Combine detected boxes with recognition output, update the confirmation
    counters, write attendance when confirmed, and decide what text / colour
    to show for each face.

    Returns (list_of_face_dicts, message_or_None).
    """
    faces = []
    toast = None
    seen = set()

    for location, result in zip(locations, recognition):
        name = result.name

        if name == "Unknown":
            faces.append({"location": location, "name": name, "confidence": 0.0,
                          "status": "Not registered", "color": utils.RED})
            continue

        seen.add(name)
        confirm_counts[name] = confirm_counts.get(name, 0) + 1

        if name in today_records:
            # Already marked earlier today -> show it but do NOT write again.
            status = f"Present (since {today_records[name]})"
            color = utils.GREEN
        elif confirm_counts[name] >= utils.CONFIRM_FRAMES:
            # Recognised several times in a row -> record attendance now.
            newly_marked, time_str = utils.mark_attendance(name, today_records)
            if newly_marked:
                toast = f"Attendance marked: {name} at {time_str}"
                status, color = f"Present ({time_str})", utils.GREEN
            else:
                status, color = "Verifying...", utils.ORANGE
        else:
            status = f"Verifying {confirm_counts[name]}/{utils.CONFIRM_FRAMES}"
            color = utils.ORANGE

        faces.append({"location": location, "name": name, "confidence": result.confidence,
                      "status": status, "color": color})

    # Anybody who was NOT seen this round loses their progress.
    for name in list(confirm_counts):
        if name not in seen:
            confirm_counts[name] = 0

    return faces, toast


def main():
    parser = argparse.ArgumentParser(description="Smart Face Attendance Monitoring System")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index (default 0)")
    parser.add_argument("--no-motion", action="store_true", help="Disable motion gating")
    args = parser.parse_args()

    # ---------------------------------------------------------------- set-up --
    utils.ensure_folders()
    utils.init_attendance_file()
    cascade = utils.load_cascade()          # Haar Cascade classifier
    try:
        recognizer = FaceRecognizer()       # loads models/encodings.pickle
    except FileNotFoundError as error:
        print(f"[ERROR] {error}")
        return

    cap = utils.open_camera(args.camera)
    motion_detector = utils.MotionDetector()
    fps_counter = utils.FPSCounter()

    today = datetime.now().strftime("%Y-%m-%d")
    today_records = utils.get_today_records(today)   # {name: time} already present
    confirm_counts = {}                              # consecutive recognitions per person

    motion_enabled = not args.no_motion
    frame_index = 0
    processed_count = 0
    faces = []                                       # results reused on skipped frames
    last_face_time = 0.0
    toast_text, toast_until = "", 0.0
    motion_text = "ON" if motion_enabled else "OFF"

    print("[INFO] System running. Press 'q' in the video window to quit.")

    # ------------------------------------------------------------- main loop --
    while True:
        ok, frame = cap.read()
        if not ok:
            print("[ERROR] Could not read from webcam.")
            break
        frame_index += 1
        now = datetime.now()

        # New day started while the program was running -> reset daily data.
        if now.strftime("%Y-%m-%d") != today:
            today = now.strftime("%Y-%m-%d")
            today_records = utils.get_today_records(today)
            confirm_counts.clear()

        # ---- Image operations ------------------------------------------------
        small = utils.resize_frame(frame, utils.FRAME_SCALE)     # 1) resize
        gray = utils.to_gray(small)                              # 2) grayscale

        # ---- Motion detection (bonus) ---------------------------------------
        moving, motion_percent = motion_detector.update(gray)
        if motion_enabled:
            motion_text = "ACTIVE" if moving else "IDLE"

        # ---- Heavy pipeline only on every N-th frame -------------------------
        if frame_index % utils.PROCESS_EVERY_N == 0:
            processed_count += 1

            # Run detection if: motion gating is off, OR something moved,
            # OR a face was seen recently, OR it is a periodic "heartbeat" check.
            active = (not motion_enabled
                      or moving
                      or (time.time() - last_face_time) < utils.IDLE_TIMEOUT
                      or processed_count % utils.HEARTBEAT_EVERY == 0)

            if active:
                # 3) blur + equalise, then Haar Cascade detection
                boxes = utils.detect_faces_haar(cascade, utils.enhance_gray(gray))
                locations = utils.boxes_to_locations(boxes, utils.FRAME_SCALE)

                if locations:
                    last_face_time = time.time()
                    rgb = utils.bgr_to_rgb(frame)                # 4) BGR -> RGB
                    recognition = recognizer.recognize(rgb, locations)
                else:
                    recognition = []

                faces, toast = build_results(locations, recognition,
                                             today_records, confirm_counts)
                if toast:
                    toast_text, toast_until = toast, time.time() + 3.0
            else:
                faces = []                                       # room is idle

        # ---- Drawing (image analysis & transformation) ----------------------
        display = frame.copy()
        for face in faces:
            utils.draw_face_box(display, face["location"], face["name"],
                                face["confidence"], face["status"], face["color"])

        fps = fps_counter.tick()
        utils.draw_header(display, now, len(recognizer.registered_names),
                          len(today_records), fps, motion_text)
        utils.draw_footer(display, toast_text if time.time() < toast_until else "")

        cv2.imshow("Smart Face Attendance Monitoring System", display)

        # ---- Keyboard --------------------------------------------------------
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):                 # q or ESC
            break
        elif key == ord("s"):                     # screenshot of the annotated frame
            path = os.path.join(utils.SCREENSHOT_DIR,
                                f"screenshot_{now.strftime('%Y%m%d_%H%M%S')}.png")
            cv2.imwrite(path, display)
            print(f"[INFO] Screenshot saved: {path}")
        elif key == ord("m"):                     # toggle motion gating
            motion_enabled = not motion_enabled
            motion_text = "ON" if motion_enabled else "OFF"

    # --------------------------------------------------------------- clean-up --
    cap.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Stopped. {len(today_records)} people present today. "
          f"See {utils.ATTENDANCE_FILE}")


if __name__ == "__main__":
    main()
