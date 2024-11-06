import cv2 as cv
import numpy as np
import module as m  # Assuming 'module' is a custom module you have created
import time
import serial 

ser = serial.Serial('COM4', 115200)

# Variables
COUNTER = 0
TOTAL_BLINKS = 0
DOUBLE_BLINKS = 0
CLOSED_EYES_FRAME = 3
cameraID = 0
FRAME_COUNTER = 0
START_TIME = time.time()
FPS = 0
FORWARD_STATE = False  # Tracks whether the current state is "forward" or "stop"

# Double blink detection variables
LAST_BLINK_FRAME = -1
DOUBLE_BLINK_FRAME_THRESHOLD = 30  # Increased from 15 to 30 for a longer double blink window

# Initialize camera object
camera = cv.VideoCapture(cameraID)

# Check if camera opens correctly
if not camera.isOpened():
    print("Error: Unable to access the camera.")
    exit()

# Get video properties
fps = camera.get(cv.CAP_PROP_FPS)
width = camera.get(cv.CAP_PROP_FRAME_WIDTH)
height = camera.get(cv.CAP_PROP_FRAME_HEIGHT)
print(f"Width: {width}, Height: {height}, FPS: {fps}")

while True:
    FRAME_COUNTER += 1
    ret, frame = camera.read()
    #frame = frame[::-1]

    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Convert frame to grayscale
    grayFrame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    height, width = grayFrame.shape
    circleCenter = (int(width / 2), 50)

    # Call the face detector function
    image, face = m.faceDetector(frame, grayFrame)

    if face is not None:
        # Call landmarks detector function
        image, PointList = m.faceLandmakDetector(frame, grayFrame, face, False)

        cv.putText(frame, f'FPS: {round(FPS, 1)}',
                   (460, 20), m.fonts, 0.7, m.YELLOW, 2)

        # Extract right eye points
        RightEyePoint = PointList[36:42]

        # Blink detection for the right eye
        rightRatio, rTop, rBottom = m.blinkDetector(RightEyePoint)

        # Blink ratio
        blinkRatio = rightRatio

        # Draw circles to represent blink ratio
        cv.circle(image, circleCenter, int(blinkRatio * 4.3), m.CHOCOLATE, -1)
        cv.circle(image, circleCenter, int(blinkRatio * 3.2), m.CYAN, 2)
        cv.circle(image, circleCenter, int(blinkRatio * 2), m.GREEN, 3)

        # Blink detection logic
        if blinkRatio > 4:
            COUNTER += 1
            cv.putText(image, 'Blink', (70, 50), m.fonts, 0.8, m.LIGHT_BLUE, 2)
            print("Blink detected")
        else:
            if COUNTER > CLOSED_EYES_FRAME:
                TOTAL_BLINKS += 1
                COUNTER = 0
                print(f"Total Blinks: {TOTAL_BLINKS}")

                # Double blink detection
                if LAST_BLINK_FRAME != -1 and (FRAME_COUNTER - LAST_BLINK_FRAME) <= DOUBLE_BLINK_FRAME_THRESHOLD:
                    DOUBLE_BLINKS += 1
                    print(f"Double Blink detected! Total Double Blinks: {DOUBLE_BLINKS}")

                    # Toggle the forward/stop state
                    FORWARD_STATE = not FORWARD_STATE
                    if FORWARD_STATE:
                        ser.write(b"forward\n")
                        print("Sending command: forward")
                    else:
                        ser.write(b"stop\n")
                        print("Sending command: stop")

                # Update the last blink frame
                LAST_BLINK_FRAME = FRAME_COUNTER

        # Eye tracking for right eye only
        mask, pos, color = m.EyeTracking(frame, grayFrame, RightEyePoint)

        # Print eye position in terminal
        if pos == 'Left':
            print("Eye moved right")
            ser.write(b"left\n")
        elif pos == 'Right':
            print("Eye moved right")
            ser.write(b"right\n")
        elif pos == 'Center':
            print("Eye in center")
            if not FORWARD_STATE:
                ser.write(b"stop\n")

        # Draw background for text
        cv.line(image, (30, 90), (100, 90), color[0], 30)
        cv.line(image, (25, 50), (135, 50), m.WHITE, 30)

        # Write text for right eye position
        cv.putText(image, f'{pos}', (35, 95), m.fonts, 0.6, color[1], 2)
        cv.putText(image, 'Right Eye', (35, 55), m.fonts, 0.6, color[1], 2)

        # Display the frame
        cv.imshow('Frame', image)
    else:
        # Display the original frame if no face is detected
        cv.imshow('Frame', frame)

    # Calculate the time elapsed and frame rate
    SECONDS = time.time() - START_TIME
    FPS = FRAME_COUNTER / SECONDS

    # Press 'q' to quit the loop
    key = cv.waitKey(1)
    if key == ord('q'):
        break

# Release camera and close windows
camera.release()
cv.destroyAllWindows()

