import cv2
import numpy as np

# Load the game window screenshot and template (HP bar)
game_frame = cv2.imread('pic2.png')
template = cv2.imread('pic1.png')

# Perform template matching
result = cv2.matchTemplate(game_frame, template, cv2.TM_CCOEFF_NORMED)

# Get the location of the best match
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
x, y = max_loc

# Get the width and height of the template
h, w = template.shape[:2]

# set bar position manually
x,y,w,h = 201,980,322, 10


# Draw a rectangle around the matched region for visualization
cv2.rectangle(game_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Calculate the four corners of the detected HP bar region
top_left = (x, y)
top_right = (x + w, y)
bottom_left = (x, y + h)
bottom_right = (x + w, y + h)

# Print the coordinates of the four corners
print(f"Top-left corner: {top_left}")
print(f"Top-right corner: {top_right}")
print(f"Bottom-left corner: {bottom_left}")
print(f"Bottom-right corner: {bottom_right}")

# Show the result
cv2.imshow('Matched Result', game_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Extract the HP bar region from the game frame
hp_bar_region = game_frame[y:y + h, x:x + w]

# Now you can apply the existing HP calculation logic on this region
# Example: curr_hp = get_current_hp(hp_bar_region)
