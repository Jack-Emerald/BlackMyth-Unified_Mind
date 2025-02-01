import mss
import numpy as np
import cv2

# Define the game window size
IMG_WIDTH = 1920
IMG_HEIGHT = 1080

# Initialize mss for screen capture
sct = mss.mss()

# Select the monitor (1 is the primary monitor)
monitor = sct.monitors[1]

# Capture the entire screen
sct_img = sct.grab(monitor)

# Convert the screenshot to a NumPy array and then to RGB format
frame = cv2.cvtColor(np.asarray(sct_img), cv2.COLOR_BGRA2RGB)

# Cut the frame to the size of the game window (adjust the crop coordinates)
frame = frame[46:IMG_HEIGHT + 46, 12:IMG_WIDTH + 12]

# Save the captured game window image
cv2.imwrite('captured_game_window.png', frame)

# Optionally, you can display the captured frame
cv2.imshow("Captured Frame", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
