# Smart Face Attendance Monitoring System Using OpenCV

A beginner-friendly Computer Vision project that watches a **live webcam**, detects faces with a **Haar Cascade**, recognises registered people with the `face_recognition` library and writes their attendance (Name, Date, Time, Status) into `attendance/Attendance.csv` — **once per person per day**.

---

## 1. Project Overview

| Step | Script | What it does |
|------|--------|--------------|
| 1 | `capture_faces.py` | Registers a person: saves face photos from the webcam into `dataset/<name>/` |
| 2 | `train_faces.py` | Converts every photo into a 128-number face encoding and saves `models/encodings.pickle` |
| 3 | `main.py` | Runs the live attendance system |

Supporting files: `face_recognition.py` (recognition engine) and `utils.py` (shared helpers, settings, drawing, CSV logic).

**Pipeline per frame**

```
Webcam frame -> resize (50%) -> grayscale -> motion check (bonus)
             -> blur + histogram equalisation -> Haar Cascade face detection
             -> BGR->RGB -> face encoding -> compare with registered faces
             -> confirm 3 times in a row -> write Attendance.csv (once per day)
             -> draw boxes, names, status, date/time, FPS
```

## 2. Features

- Live webcam processing with OpenCV
- **Haar Cascade** face detection
- **Image operations**: resize, BGR→RGB, grayscale, Gaussian blur, histogram equalisation
- **Image analysis & transformation**: bounding boxes, name labels, confidence score, attendance status, translucent overlay with date/time
- **Bonus: motion detection** (frame differencing) so the pipeline relaxes when the room is empty
- Register any number of people
- Attendance saved to `Attendance.csv` (Name, Date, Time, Status = Present)
- **No duplicates**: a person is marked only once per day, even if the program is restarted
- Confirmation rule: a person must be recognised in 3 consecutive processed frames before being marked (reduces false positives)
- On-screen: registered count, present-today count, FPS, motion status
- Keyboard: `q` quit, `s` screenshot, `m` motion on/off

## 3. Installation

1. Install **Python 3.9 – 3.11** (3.12 also works if `setuptools<81` is installed, already in `requirements.txt`).
2. Open a terminal inside the `FaceAttendance` folder.
3. (Recommended) create a virtual environment:

   ```bash
   python -m venv venv
   venv\Scripts\activate          # Windows
   source venv/bin/activate       # macOS / Linux
   ```
4. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

### Troubleshooting `dlib` installation

`face_recognition` depends on `dlib`, which must be compiled the first time.

- **Windows**: install *Visual Studio Build Tools* (C++ workload) and CMake, **or** try a prebuilt wheel: `pip install dlib-bin`
- **macOS**: `brew install cmake` then `pip install dlib`
- **Linux**: `sudo apt install build-essential cmake` then `pip install dlib`
- Error *"Please install `face_recognition_models`"*: run `pip install "setuptools<81" face_recognition_models`

## 4. Required Libraries

| Library | Used for |
|---------|----------|
| `opencv-python` | webcam, Haar Cascade, image operations, drawing |
| `face_recognition` (+ `dlib`) | 128-d face encodings, face distance |
| `numpy` | array maths |
| `pandas` | reading and checking `Attendance.csv` |
| `datetime`, `os`, `csv`, `time`, `pickle`, `argparse` | built into Python |

## 5. How to Register Faces

```bash
python capture_faces.py
```

Type the person's name, press **SPACE** and slowly move your head a little (left, right, up, down, smile). The script saves 20 sharp face photos into `dataset/<Name>/`.

Options:

```bash
python capture_faces.py --name "Juan Dela Cruz" --samples 25 --camera 0
```

Repeat for every person. Running it again for the same name **adds** more photos.

Tips for good registration: good front lighting, one person only, no sunglasses/mask, a few different head angles.

> Alternative: you can also drop your own photos (`.jpg`/`.png`, one face each) into `dataset/<Name>/`.

## 6. How to Train

```bash
python train_faces.py
```

This reads every photo, computes its encoding and saves `models/encodings.pickle`. Run it **every time** you register a new person or add photos.

> "Training" here means *enrolment*. The deep model inside `face_recognition` is already pre-trained; we only store the encodings of our own people.

## 7. How to Run

```bash
python main.py
```

Optional: `python main.py --camera 1` (external webcam) or `python main.py --no-motion` (disable motion gating).

Stand in front of the camera. The box is **orange** ("Verifying 1/3 ...") and turns **green** ("Present (HH:MM:SS)") once attendance is saved. Unknown faces get a **red** box.

Open `attendance/Attendance.csv` to view the records (close it in Excel before re-running the program, because Excel locks the file).

## 8. Folder Structure

```
FaceAttendance/
├── dataset/                 <- one sub-folder of photos per person
│   ├── person1/
│   └── person2/
├── models/
│   ├── haarcascade_frontalface_default.xml
│   └── encodings.pickle     <- created by train_faces.py
├── attendance/
│   └── Attendance.csv       <- attendance log
├── screenshots/             <- screenshots (press 's' while running)
├── main.py                  <- live attendance application
├── capture_faces.py         <- register faces
├── train_faces.py           <- create encodings
├── face_recognition.py      <- recognition engine (FaceRecognizer class)
├── utils.py                 <- settings, image ops, motion, CSV, drawing
├── requirements.txt
├── README.md
├── PRESENTATION_GUIDE.md    <- demo plan and Q&A preparation
└── documentation.pdf
```

> **Note on the file name `face_recognition.py`:** the project file has the same name as the third-party library. To stop Python importing the wrong one, `utils.get_face_recognition_lib()` loads the real library safely. Always import the library through that function, never with a plain `import face_recognition`.

## 9. Sample Outputs

> The examples below show the *format* of the output. Replace/add your own screenshots in `screenshots/` after running the project.

**Console while running**

```
[INFO] Loaded 40 face samples for 2 registered people.
[INFO] System running. Press 'q' in the video window to quit.
[ATTENDANCE] Maria Santos marked Present on 2026-10-13 at 09:02:41
[ATTENDANCE] Juan Dela Cruz marked Present on 2026-10-13 at 09:05:17
[INFO] Screenshot saved: screenshots/screenshot_20261013_090530.png
[INFO] Stopped. 2 people present today. See attendance/Attendance.csv
```

**Attendance.csv**

```
Name,Date,Time,Status
Maria Santos,2026-10-13,09:02:41,Present
Juan Dela Cruz,2026-10-13,09:05:17,Present
```

**Video window (layout)**

```
+--------------------------------------------------------+
| Smart Face Attendance            Registered: 2         |
| 2026-10-13   09:02:41            Present today: 1      |
|                                  FPS: 24.3 | Motion: ACTIVE |
|                                                        |
|        [ Maria Santos (91%) ]  <- green box            |
|        [        face        ]                          |
|        Present (09:02:41)                              |
|                                                        |
| Attendance marked: Maria Santos at 09:02:41            |
| q: quit | s: screenshot | m: motion on/off             |
+--------------------------------------------------------+
```

## 10. Tuning (in `utils.py`)

| Setting | Meaning | If you have problems |
|---------|---------|----------------------|
| `MATCH_TOLERANCE` (0.5) | max face distance for a match | Unknown people accepted → lower it (0.45). Registered people rejected → raise it (0.55) |
| `CONFIRM_FRAMES` (3) | consecutive recognitions needed | Raise for stricter marking |
| `FRAME_SCALE` (0.5) | detection frame size | Raise to 0.75 to detect smaller/farther faces |
| `PROCESS_EVERY_N` (2) | process every N-th frame | Raise for more FPS on slow computers |
| `HAAR_MIN_NEIGHBORS` (5) | Haar strictness | Raise if false boxes appear; lower if faces are missed |

## 11. Common Problems

| Problem | Solution |
|---------|----------|
| "Cannot open webcam" | Close Zoom/Teams, allow camera permission, try `--camera 1` |
| "No trained data found" | Run `capture_faces.py` then `train_faces.py` |
| Everyone shows "Unknown" | Re-register in better light with more angles, re-run `train_faces.py` |
| Low FPS | Raise `PROCESS_EVERY_N`, lower `FRAME_SCALE`, keep motion detection on |
| "Cannot write Attendance.csv" | Close the file in Excel |

## 12. Privacy Note

Face images and encodings are personal data. Register only people who have given consent, keep `dataset/` and `models/encodings.pickle` private, and do not commit them to public repositories.
