import cv2
import time
import numpy as np
from gaze_tracking import GazeTracking

# Wheelchair control states
class WheelchairState:
    STOP = "STOP"
    FORWARD = "FORWARD"
    BACKWARD = "BACKWARD"
    LEFT = "LEFT"
    RIGHT = "RIGHT"

# Initialize gaze tracking and webcam
gaze = GazeTracking()
webcam = cv2.VideoCapture(2)

# Wheelchair control variables
wheelchair_state = WheelchairState.STOP
last_state_change_time = time.time()
eye_closed_timeout = 3  # Stop wheelchair if eyes are closed for 3 seconds

# Buffer for smoothing gaze direction
gaze_buffer_size = 5
gaze_buffer = []

# Calibration data
calibration_data = {
    "center_gaze_ratio": 0.5,
    "left_threshold": 0.6,
    "right_threshold": 0.4,
    "blinking_threshold": 3.8,
}

# Comprehensive calibration phase
def calibrate():
    print("Starting comprehensive calibration...")

    def collect_gaze_data(prompt, duration):
        print(prompt)
        time.sleep(2)  # Give the user time to prepare
        gaze_ratios = []
        blinking_ratios = []

        start_time = time.time()
        while time.time() - start_time < duration:
            _, frame = webcam.read()
            gaze.refresh(frame)

            if gaze.pupils_located:
                gaze_ratio = gaze.horizontal_ratio()
                gaze_ratios.append(gaze_ratio)
                blinking_ratios.append((gaze.eye_left.blinking + gaze.eye_right.blinking) / 2)

            time.sleep(0.1)  # Wait for a short duration between frames

        if gaze_ratios:
            return np.mean(gaze_ratios), np.mean(blinking_ratios)
        else:
            return None, None

    # Step 1: Center gaze (forward)
    center_gaze_ratio, _ = collect_gaze_data("Please look at the center of the screen (forward).", duration=5)
    if center_gaze_ratio is not None:
        calibration_data["center_gaze_ratio"] = center_gaze_ratio
        print(f"Center gaze ratio: {center_gaze_ratio}")

    # Step 2: Left gaze
    left_gaze_ratio, _ = collect_gaze_data("Please look to the left.", duration=5)
    if left_gaze_ratio is not None:
        calibration_data["left_threshold"] = (center_gaze_ratio + left_gaze_ratio) / 2
        print(f"Left gaze ratio: {left_gaze_ratio}")

    # Step 3: Right gaze
    right_gaze_ratio, _ = collect_gaze_data("Please look to the right.", duration=5)
    if right_gaze_ratio is not None:
        calibration_data["right_threshold"] = (center_gaze_ratio + right_gaze_ratio) / 2
        print(f"Right gaze ratio: {right_gaze_ratio}")

    # Step 4: Blinking
    _, blinking_ratio = collect_gaze_data("Please blink your eyes.", duration=5)
    if blinking_ratio is not None:
        calibration_data["blinking_threshold"] = blinking_ratio * 1.2  # Add a 20% buffer
        print(f"Blinking ratio: {blinking_ratio}")

    print("Calibration complete!")
    print(f"Calibration data: {calibration_data}")

# Run comprehensive calibration
calibrate()

while True:
    # Get a new frame from the webcam
    _, frame = webcam.read()

    # Analyze the frame
    gaze.refresh(frame)
    frame = gaze.annotated_frame()
    text = ""

    # Get horizontal gaze ratio
    if gaze.pupils_located:
        gaze_ratio = gaze.horizontal_ratio()
        gaze_buffer.append(gaze_ratio)
        if len(gaze_buffer) > gaze_buffer_size:
            gaze_buffer.pop(0)

        # Smooth the gaze ratio using a moving average
        smoothed_gaze_ratio = np.mean(gaze_buffer)

        # Determine gaze direction
        blinking_ratio = (gaze.eye_left.blinking + gaze.eye_right.blinking) / 2
        if blinking_ratio > calibration_data["blinking_threshold"]:
            text = "Blinking"
            if time.time() - last_state_change_time > eye_closed_timeout:
                wheelchair_state = WheelchairState.STOP  # Stop if eyes are closed for too long
        elif smoothed_gaze_ratio <= calibration_data["right_threshold"]:  # Right gaze
            text = "Looking right"
            wheelchair_state = WheelchairState.RIGHT
            last_state_change_time = time.time()
        elif smoothed_gaze_ratio >= calibration_data["left_threshold"]:  # Left gaze
            text = "Looking left"
            wheelchair_state = WheelchairState.LEFT
            last_state_change_time = time.time()
        else:  # Center gaze
            text = "Looking center"
            wheelchair_state = WheelchairState.FORWARD
            last_state_change_time = time.time()
    else:
        text = "Eyes not detected"
        wheelchair_state = WheelchairState.STOP

    # Display wheelchair state
    cv2.putText(frame, f"State: {wheelchair_state}", (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)
    cv2.putText(frame, text, (90, 120), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)

    # Display pupil coordinates
    left_pupil = gaze.pupil_left_coords()
    right_pupil = gaze.pupil_right_coords()
    cv2.putText(frame, "Left pupil:  " + str(left_pupil), (90, 180), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
    cv2.putText(frame, "Right pupil: " + str(right_pupil), (90, 210), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)

    # Display the frame
    cv2.imshow("Wheelchair Control", frame)

    # Exit on ESC key
    if cv2.waitKey(1) == 27:
        break

# Release resources
webcam.release()
cv2.destroyAllWindows()