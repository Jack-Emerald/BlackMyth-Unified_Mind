import mss
import numpy as np
import cv2
import time

# Initialize mss to capture the screen
sct = mss.mss()

# Select the first monitor (or adjust based on your setup)
monitor = sct.monitors[1]

# Define the threshold for white pixels (full HP)
lower_white = np.array([5, 5, 215])  # Lower bound for white color
upper_white = np.array([15, 15, 235])  # Upper bound for white color

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
    #pixel_color = frame[918, 700]  # (y, x) format
    pixel_color = frame[918, 770]  # (y, x) format
    pixel_image = frame[913:921, 757:952]
    hsv = cv2.cvtColor(pixel_image, cv2.COLOR_RGB2HSV)  # Convert to HSV color space
    mask = cv2.inRange(hsv, lower_white, upper_white)  # Create a mask for white pixels
    #cv2.imshow('Mask', mask)
    if np.sum(mask == 255)>=1:
        print("game start.")
        #break

    # Convert the pixel color to HSV
    hsv_pixel = cv2.cvtColor(np.uint8([[pixel_color]]), cv2.COLOR_RGB2HSV)[0][0]

    '''    
    min_h = min(min_h, hsv_pixel[0])
    max_h = max(max_h, hsv_pixel[0])
    min_s = min(min_s, hsv_pixel[1])
    max_s = max(max_s, hsv_pixel[1])
    min_v = min(min_v, hsv_pixel[2])
    max_v = max(max_v, hsv_pixel[2])
    '''
    # Find the min and max HSV values within the pixel image
    min_h = min(min_h, np.min(hsv[:, :, 0]))  # min hue
    max_h = max(max_h, np.max(hsv[:, :, 0]))  # max hue
    min_s = min(min_s, np.min(hsv[:, :, 1]))  # min saturation
    max_s = max(max_s, np.max(hsv[:, :, 1]))  # max saturation
    min_v = min(min_v, np.min(hsv[:, :, 2]))  # min value
    max_v = max(max_v, np.max(hsv[:, :, 2]))  # max value

    # Print the RGB color and the current HSV range
    #print(f"RGB Color: {pixel_color}")
    #print(f"HSV Color: {hsv_pixel}")
    print(f"HSV Range: min [{min_h}, {min_s}, {min_v}], max [{max_h}, {max_s}, {max_v}]")
    # Wait for 1 second before capturing the next frame
    time.sleep(0.2)