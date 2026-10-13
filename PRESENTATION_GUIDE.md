# 15-Minute Presentation Guide

## Before the presentation (10 minutes of prep)

- Register **2–3 people** (yourself + 1–2 classmates) and run `train_faces.py` beforehand.
- Run `main.py` once, press `s` to save 2–3 screenshots, and paste them into `documentation.pdf`.
- **Delete today's rows** from `Attendance.csv` (keep the header) so the live demo shows a fresh marking.
- Test the webcam, lighting (light in front of you, not behind) and close Excel.
- Have a backup: a short screen recording of a successful run, in case the webcam fails.

## Time plan

| Min | Part | What to say / do |
|-----|------|------------------|
| 0:00–1:30 | **Problem & objective** | Manual roll call wastes time and allows "proxy" attendance. Goal: automatic, contactless attendance with no duplicates. |
| 1:30–4:00 | **Architecture** | Show the pipeline diagram from the README. Walk through the 3 scripts: capture → train → main. Mention the 3 required CV topics. |
| 4:00–7:00 | **Code walkthrough** | Open `utils.py`: show `resize_frame`, `to_gray`, `bgr_to_rgb`, `enhance_gray` (image operations), `detect_faces_haar` (Haar), `draw_face_box` / `draw_header` (analysis & transformation). Open `face_recognition.py`: explain encodings + distance + tolerance. |
| 7:00–8:30 | **Registration demo** | Run `capture_faces.py --name "Demo Person" --samples 10`. Show photos appearing in `dataset/`. Run `train_faces.py` and show the summary table. |
| 8:30–12:30 | **Live demo** | Run `main.py`. Point out: live feed, date/time, registered count, FPS, motion status. Step in → orange "Verifying" → green "Present" → toast message. Open `Attendance.csv`. **Stay in front again** → no second row (duplicate prevention). Show an **unknown** person (red box). Press `m` to toggle motion detection. Press `s` for a screenshot. |
| 12:30–14:00 | **Results & limitations** | Accuracy test table, limits (lighting, frontal faces, photo spoofing), future work. |
| 14:00–15:00 | **Conclusion & questions** | Restate: Haar + image operations + analysis/transformation + motion bonus, working live system. |

## Demo tips

- Narrate what is happening on screen while you move; silence feels long.
- If recognition is slow to confirm, explain the 3-frame confirmation rule — it is a feature, not a bug.
- Keep `MATCH_TOLERANCE` ready in `utils.py` to show how strictness is tuned.

---

## Five Likely Instructor Questions (with answers)

**1. Why do you use a Haar Cascade for detection and `face_recognition` for recognition? Aren't they the same thing?**
No — they solve different problems. *Detection* answers "where is a face?" and the Haar Cascade does it quickly on a grayscale image using many simple rectangular brightness features combined in a cascade of classifiers (AdaBoost). *Recognition* answers "whose face is it?" and is done by a deep neural network (dlib ResNet) that converts the face into a 128-number encoding. Using fast Haar detection first, then recognising only the detected faces, keeps the system real-time.

**2. How does the system decide who a person is, and what is the confidence score?**
Each live face is converted into a 128-d encoding. We compute the Euclidean distance to every stored encoding and keep the smallest distance per person. If the best distance is ≤ `MATCH_TOLERANCE` (0.5) the person is recognised; otherwise "Unknown". The percentage shown is a similarity score mapped from that distance — it is *not* a statistical probability.

**3. How do you prevent a person from being marked twice in one day?**
At start-up `get_today_records()` reads `Attendance.csv` with pandas and builds a dictionary of names already present today. `mark_attendance()` checks that dictionary (and the file again) before writing. Because the check uses the saved file, it still works if the program is closed and restarted. At midnight the program reloads the records for the new date. Additionally, a person must be recognised in 3 consecutive processed frames before being marked, which avoids accidental wrong entries.

**4. What are the main weaknesses of the system and how could someone cheat it?**
Haar Cascades mostly detect *frontal* faces and are sensitive to lighting, so side views or dark rooms may fail. Recognition accuracy drops with masks, strong shadows, or poor registration photos. The system has **no liveness detection**, so a printed photo or a phone screen of a registered person could be accepted. Future work: blink/head-movement liveness check, a CNN/DNN detector, a database instead of CSV, and an admin dashboard.

**5. Why did you resize the frame and add motion detection? What do they improve?**
Resizing to 50% reduces the number of pixels about 4×, so Haar detection runs much faster — we then map the boxes back to the full-size frame so recognition still uses the sharper image. Motion detection (frame differencing) is a bonus feature: when nothing moves and no face was seen recently, the system skips face detection and only does a periodic check, saving CPU. As soon as someone moves into view, detection wakes up automatically.

---

### Extra questions to be ready for

- *What does "training" mean here?* — Computing and storing encodings; the network itself is pre-trained.
- *Why BGR→RGB?* — OpenCV loads colour as BGR, but `face_recognition` expects RGB; skipping it gives wrong colours and worse accuracy.
- *How would you scale this to 1,000 people?* — Use a database and an approximate nearest-neighbour index (FAISS), average several encodings per person, and run on a GPU.
- *What about privacy?* — Collect consent, store only encodings where possible, encrypt/limit access to `dataset/`.
