import mss
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt

# Define the screen dimensions for the game window
IMG_WIDTH = 1920  # Game window width
IMG_HEIGHT = 1080  # Game window height


# HP bar's top-left corner (x, y) and width (w) and height (h)
x, y, w, h = 201,980,325, 8

#boss
#x1, y1, w1, h1 = 675,913,570, 8
#elite
x1, y1, w1, h1 = 757,913,395, 8

charge_points = [(1040, 1782), (1020, 1802), (997, 1815)]
'''
lower_white = np.array([0, 0, 175])  # Lower bound for white color
upper_white = np.array([180, 40, 255])  # Upper bound for white color
'''
# Define the threshold for white pixels (full HP)
lower_white = np.array([0, 0, 100])  # Lower bound for white color
upper_white = np.array([180, 160, 230])  # Upper bound for white color



# Define the threshold for the charge point colors in HSV
# We need a range that captures the colors [255, 255, 234], [255, 255, 226], [255, 255, 220]
lower_charge = np.array([0, 0, 200])  # Lower bound for light yellow color (light yellow shades)
upper_charge = np.array([50, 100, 220])  # Upper bound for light yellow color

def in_game_status():

    # Initialize mss to capture the screen
    sct = mss.mss()

    # Select the first monitor (or adjust based on your setup)
    monitor = sct.monitors[1]

    # Store player HP for graphing after the loop
    player_hp_history = []
    boss_hp_history = []

    previous_boss_hp_percentage = 1.0

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

        left_middle_pixel = frame[y + h // 2, x]
        print(f"Left-middle pixel of player HP bar: {left_middle_pixel}")

        # Print the color of the pixel at charge points
        first_charge = frame[1040, 1782]
        #print(f"Color of pixel at 1 charge: {first_charge}")
        second_charge = frame[1020, 1802]
        #print(f"Color of pixel at 2 charge: {second_charge}")
        third_charge = frame[997, 1815]
        #print(f"Color of pixel at 3 charge: {third_charge}")

        # Print the color of the pixel at charge points
        first_charge = frame[1040, 1782]
        # print(f"Color of pixel at 1 charge: {first_charge}")

        # Apply color filtering to detect white pixels (representing HP)
        hsv = cv2.cvtColor(hp_image, cv2.COLOR_RGB2HSV)  # Convert to HSV color space
        mask = cv2.inRange(hsv, lower_white, upper_white)  # Create a mask for white pixels
        hsv_boss = cv2.cvtColor(boss_hp_image, cv2.COLOR_RGB2HSV)  # Convert to HSV color space
        mask_boss = cv2.inRange(hsv_boss, lower_white, upper_white)  # Create a mask for white pixels

        # Initialize the charge point count
        charge_count = 0

        # Loop through the defined charge points and check their colors individually
        for charge in charge_points:
            # Get the pixel color at the charge point (directly)
            charge_pixel = frame[charge[0], charge[1]]

            # Convert the charge point pixel to HSV (no need to convert the whole frame)
            hsv_pixel = cv2.cvtColor(np.uint8([[charge_pixel]]), cv2.COLOR_RGB2HSV)[0][0]

            # Manually check if the pixel is within the range defined for charge points
            if (lower_charge[0] <= hsv_pixel[0] <= upper_charge[0] and
                    lower_charge[1] <= hsv_pixel[1] <= upper_charge[1] and
                    lower_charge[2] <= hsv_pixel[2] <= upper_charge[2]):
                charge_count += 1


        # Print the number of charge points that are activated
        #print(f"Number of charged points: {charge_count}/3")

        # Debug: Show the mask to check if white pixels are being detected
        cv2.imshow('Mask', mask)
        #print(f"White pixel count in player hp bar: {np.sum(mask == 255)}")
        cv2.imshow('Mask', mask_boss)
        #print(f"White pixel count in boss hp bar: {np.sum(mask_boss == 255)}")

        # Find all white pixels (full HP) in the mask
        matches = np.argwhere(mask == 255)
        full_hp_percentage = len(matches) / (hp_image.shape[1] * hp_image.shape[0])  # Calculate HP percentage

        boss_matches = np.argwhere(mask_boss == 255)

        boss_hp_percentage = len(boss_matches) / (boss_hp_image.shape[1] * boss_hp_image.shape[0])  # Calculate HP percentage
        # if boss_hp_percentage > previous_boss_hp_percentage:
        #     boss_hp_percentage = previous_boss_hp_percentage
        # else:
        #     previous_boss_hp_percentage = boss_hp_percentage

        # Store the player HP for graphing after the loop ends
        player_hp_history.append(full_hp_percentage * 100)  # Store as percentage
        boss_hp_history.append(boss_hp_percentage * 100)

        # Print the calculated HP percentage
        #print(f"player HP percentage: {full_hp_percentage * 100:.2f}%")
        print(f"boss HP percentage: {boss_hp_percentage * 100:.2f}%")

        if (boss_hp_percentage < 0.01 or full_hp_percentage < 0.01): #本局游戏结束
            break

        # Wait for 0.5 seconds before capturing the next frame
        time.sleep(0.5)

    return player_hp_history, boss_hp_history

def plot_hp_graph(player_hp_history,boss_hp_history):
    # Plot the player's HP over time
    plt.plot(player_hp_history)
    plt.title("Player HP Over Time")
    plt.xlabel("Time (Frame #)")
    plt.ylabel("Player HP (%)")
    plt.grid(True)
    plt.show()

    plt.plot(boss_hp_history)
    plt.title("boss HP Over Time")
    plt.xlabel("Time (Frame #)")
    plt.ylabel("boss HP (%)")
    plt.grid(True)
    plt.show()

def victory_check():
    return

if __name__ == "__main__":
    time.sleep(3)
    # Get the player HP history during the game
    player_hp_history,boss_hp_history = in_game_status()

    # After the game ends, plot the HP graph
    plot_hp_graph(player_hp_history,boss_hp_history)
     #检测是否胜利
