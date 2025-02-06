import cv2
import numpy as np

def mark_position(a,b,c,d):
    # set bar position manually
    x, y, w, h = a,b,c,d

    # Draw a rectangle around the matched region for visualization
    cv2.rectangle(game_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Calculate the four corners of the detected HP bar region
    top_left = (x, y)
    top_right = (x + w, y)
    bottom_left = (x, y + h)
    bottom_right = (x + w, y + h)


# Load the game window screenshot and template (HP bar)
game_frame = cv2.imread('captured_game_window.png')
#template = cv2.imread('pic1.png')

# Perform template matching
#result = cv2.matchTemplate(game_frame, template, cv2.TM_CCOEFF_NORMED)

# Get the location of the best match
#min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
#x, y = max_loc

# Get the width and height of the template
#h, w = template.shape[:2]

#player hp bar
mark_position(201,983,325, 8)
#boss hp bar
mark_position(677,913,570, 8)

mark_position(700,918,1, 1)

#change points 3,2,1
mark_position(1815,997,1, 1)
mark_position(1802,1020,1, 1)
mark_position(1782,1040,1, 1)


mark_position(211, 985,1, 1)

#vanquished word(boss dead)
mark_position(310,150,250, 50)

#return word(player dead)
mark_position(120,150,130, 50)

mark_position(950,450,1, 1)

# Show the result
cv2.imshow('Matched Result', game_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Extract the HP bar region from the game frame
#hp_bar_region = game_frame[y:y + h, x:x + w]

# Now you can apply the existing HP calculation logic on this region
# Example: curr_hp = get_current_hp(hp_bar_region)
