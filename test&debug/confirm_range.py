import mss
import numpy as np
import cv2
import time

# Initialize mss to capture the screen
sct = mss.mss()

# Select the first monitor (or adjust based on your setup)
monitor = sct.monitors[1]

# Variables to track min/max HSV values
min_h = 180
max_h = 0
min_s = 255
max_s = 0
min_v = 255
max_v = 0

time.sleep(5)

# Loop to continuously check and track the pixel color
while True:
    # Capture the entire screen as an image
    sct_img = sct.grab(monitor)

    # Convert the screenshot to a numpy array and then to RGB format
    frame = cv2.cvtColor(np.asarray(sct_img), cv2.COLOR_BGRA2RGB)

    # Cut the frame to the game window size (based on the offset)
    frame = frame[46:1080 + 46, 12:1920 + 12]  # Adjust based on your window position

    # Get the pixel color at the specified coordinates
    #pixel_color = frame[985, 211]  # (y, x) format
    pixel_color = frame[918, 700]  # (y, x) format

    # Convert the pixel color to HSV
    hsv_pixel = cv2.cvtColor(np.uint8([[pixel_color]]), cv2.COLOR_RGB2HSV)[0][0]

    # Update the min/max HSV values
    min_h = min(min_h, hsv_pixel[0])
    max_h = max(max_h, hsv_pixel[0])
    min_s = min(min_s, hsv_pixel[1])
    max_s = max(max_s, hsv_pixel[1])
    min_v = min(min_v, hsv_pixel[2])
    max_v = max(max_v, hsv_pixel[2])

    # Print the RGB color and the current HSV range
    print(f"RGB Color: {pixel_color}")
    print(f"HSV Color: {hsv_pixel}")
    print(f"HSV Range: min [{min_h}, {min_s}, {min_v}], max [{max_h}, {max_s}, {max_v}]")

    # Wait for 1 second before capturing the next frame
    time.sleep(0.5)