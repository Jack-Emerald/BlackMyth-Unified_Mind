import mss
import numpy as np
import cv2
import time

# Define the screen dimensions for the game window
IMG_WIDTH = 1920  # Game window width
IMG_HEIGHT = 1080  # Game window height

# Perform template matching (if you have the HP bar template)
# For this step, we assume you've already found the position of the HP bar using template matching.
# For the sake of this example, let's assume you know the region, for instance:
# HP bar's top-left corner (x, y) and width (w) and height (h)
x, y, w, h = 201,980,325, 8

x1, y1, w1, h1 = 675,913,570, 8

# Define the threshold for white pixels (full HP)
lower_white = np.array([0, 0, 175])  # Lower bound for white color
upper_white = np.array([180, 30, 255])  # Upper bound for white color

# Initialize mss to capture the screen
sct = mss.mss()

# Select the first monitor (or adjust based on your setup)
monitor = sct.monitors[1]

# Loop to continuously check the HP bar
while True:
    # Capture the entire screen as an image
    sct_img = sct.grab(monitor)

    # Convert the screenshot to a numpy array and then to RGB format
    frame = cv2.cvtColor(np.asarray(sct_img), cv2.COLOR_BGRA2RGB)

    # Cut the frame to the game window size (based on the offset)
    frame = frame[46:IMG_HEIGHT + 46, 12:IMG_WIDTH + 12]  # Adjust based on your window position

    # Extract the HP bar region from the frame
    hp_image = frame[y:y + h, x:x + w]
    boss_hp_image = frame[y1:y1 + h1, x1:x1 + w1]



    # Print the color of the pixel at the bottom-right corner
    # bottom_right_color = hp_image[h - 5, w - 1]
    # print(f"Color of pixel at bottom-right corner: {bottom_right_color}")

    # Apply color filtering to detect white pixels (representing HP)
    hsv = cv2.cvtColor(hp_image, cv2.COLOR_RGB2HSV)  # Convert to HSV color space
    mask = cv2.inRange(hsv, lower_white, upper_white)  # Create a mask for white pixels
    hsv_boss = cv2.cvtColor(boss_hp_image, cv2.COLOR_RGB2HSV)  # Convert to HSV color space
    mask_boss = cv2.inRange(hsv_boss, lower_white, upper_white)  # Create a mask for white pixels

    # Debug: Show the mask to check if white pixels are being detected
    cv2.imshow('Mask', mask)
    print(f"White pixel count in player hp bar: {np.sum(mask == 255)}")
    cv2.imshow('Mask', mask_boss)
    print(f"White pixel count in boss hp bar: {np.sum(mask_boss == 255)}")

    # Find all white pixels (full HP) in the mask
    matches = np.argwhere(mask == 255)
    full_hp_percentage = len(matches) / (hp_image.shape[1] * hp_image.shape[0])  # Calculate HP percentage

    boss_matches = np.argwhere(mask_boss == 255)
    boss_hp_percentage = len(boss_matches) / (boss_hp_image.shape[1] * boss_hp_image.shape[0])  # Calculate HP percentage

    # Print the calculated HP percentage
    print(f"player HP percentage: {full_hp_percentage * 100:.2f}%")
    print(f"boss HP percentage: {boss_hp_percentage * 100:.2f}%")

    # Wait for 0.5 seconds before capturing the next frame
    time.sleep(0.5)
