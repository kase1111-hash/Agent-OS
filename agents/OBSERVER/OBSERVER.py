# observer.py
# Prototype for OBSERVER module: Webcam motion detection with behavioral interpretation.
# Detects motion patterns, interprets via LLM, logs metadata only (privacy-focused).
# Adapted from PyImageSearch motion detection tutorial.

import cv2
import imutils
import time
import datetime
import argparse
import ollama
import os
import tempfile

# Parse command-line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-a", "--min-area", type=int, default=500, help="minimum contour area for motion detection")
ap.add_argument("-m", "--model", default="llava", help="Ollama vision model (e.g., llava)")
args = vars(ap.parse_args())

# Initialize video stream from webcam
vs = cv2.VideoCapture(0)
time.sleep(2.0)  # Warm up camera

# Initialize background frame
first_frame = None

# State tracking variables
current_state = "stillness"  # Initial assumption: no motion
state_start_time = time.time()
log_entries = []  # List to store behavioral metadata

print("OBSERVER started. Press 'q' to quit/pause.")

while True:
    # Read frame
    ret, frame = vs.read()
    if not ret:
        break

    # Preprocess: resize, grayscale, blur
    frame_resized = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # Initialize background if first frame
    if first_frame is None:
        first_frame = gray
        continue

    # Compute delta and threshold
    frame_delta = cv2.absdiff(first_frame, gray)
    thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # Detect motion if any large contours
    motion_detected = False
    for c in cnts:
        if cv2.contourArea(c) < args["min_area"]:
            continue
        motion_detected = True
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame_resized, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Update state
    new_state = "motion" if motion_detected else "stillness"
    if new_state != current_state:
        # Calculate duration of previous state
        duration = time.time() - state_start_time
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Capture temp image for LLM analysis (privacy: delete after)
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_img:
            cv2.imwrite(temp_img.name, frame_resized)
        
        # Prompt LLM for behavioral description
        try:
            response = ollama.chat(
                model=args["model"],
                messages=[
                    {
                        "role": "user",
                        "content": "Describe the user's posture and activity in this image briefly, focusing on behavioral indicators like pacing (thinking), stillness (focus), leaning back (contemplation), or frustration. Do not identify the person or store details.",
                        "images": [temp_img.name]
                    }
                ]
            )
            description = response["message"]["content"]
        except Exception as e:
            description = f"Error in LLM call: {str(e)}"
        
        # Delete temp image
        os.unlink(temp_img.name)

        # Classify simple patterns (expand with more rules or another LLM)
        if current_state == "stillness":
            pattern = "focus" if duration > 60 else "brief pause"  # e.g., >1 min = focus
        else:
            pattern = "thinking/pacing" if duration > 180 else "brief activity"  # e.g., >3 min = pacing
        
        # Log metadata (integrate with Primary Model)
        log_entry = f"{timestamp}: {current_state.capitalize()} for {duration:.1f} seconds ({pattern}). Description: {description}"
        log_entries.append(log_entry)
        print(log_entry)

        # Reset state
        current_state = new_state
        state_start_time = time.time()

    # Display feed (for debugging; remove in production for lower resources)
    cv2.putText(frame_resized, f"State: {current_state}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.imshow("Observer Feed", frame_resized)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup
vs.release()
cv2.destroyAllWindows()

# Save logs (e.g., to file for synthesis layer)
with open("observer_logs.txt", "w") as f:
    f.write("\n".join(log_entries))

print("OBSERVER stopped. Logs saved to observer_logs.txt.")
