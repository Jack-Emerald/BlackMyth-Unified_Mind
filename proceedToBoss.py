import pydirectinput
import time
import cv2
import mss
import numpy as np

class proceedToBoss:
    """Proceed to boss class: hard coded configurations to begin fighting the bosses"""

    '''Constructor'''
    def __init__(self, BOSS):
        self.BOSS = BOSS  # Boss number | 99/100 reserved for PVP

    '''Proceed to boss function'''
    def perform(self):
        """Performing the actions need to be taken before fight the bosses;
        currently only one configuration: challenge mode bosses (1)"""
        if self.BOSS == 1: # Challenge mode bosses
            print("Fight BOSS with challenge mode")
            self.boss1()
        else:
            print("ProceedToBoss Configuration Not Found")

    '''Wait until we can begin fighting the bosses (loading screen)'''
    @staticmethod
    def wait_till_begin():
        # Initialize mss to capture the screen
        sct = mss.mss()

        # Select the first monitor (or adjust based on your setup)
        monitor = sct.monitors[1]

        # Define the threshold for white pixels (full HP)
        lower_white = np.array([5, 5, 215])  # Lower bound for white color
        upper_white = np.array([15, 15, 235])  # Upper bound for white color

        print("Wait till begin", end="")

        # Loop to continuously check and track the pixel color
        while True:
            # Capture the entire screen as an image
            sct_img = sct.grab(monitor)

            # Convert the screenshot to a numpy array and then to RGB format
            frame = cv2.cvtColor(np.asarray(sct_img), cv2.COLOR_BGRA2RGB)

            # Cut the frame to the game window size (based on the offset)
            frame = frame[46:1080 + 46, 12:1920 + 12]  # Adjust based on your window position

            # Get the pixel color at the specified coordinates
            pixel_image = frame[450:451, 950:951]
            hsv = cv2.cvtColor(pixel_image, cv2.COLOR_RGB2HSV)  # Convert to HSV color space
            mask = cv2.inRange(hsv, lower_white, upper_white)  # Create a mask for white pixels
            cv2.imshow('Mask', mask)

            if np.sum(mask == 255) >= 1:
                print("LET'S FIGHT!!!")
                break

            print(".", end="")
            time.sleep(0.2) # Sleep for a while for the next loop

    '''Configuration 1: challenge mode bosses'''
    def boss1(self):
        #self.put_on_lantern()
        time.sleep(3)
        print("Begin to challenge the boss...")
        pydirectinput.press('e')
        time.sleep(1)
        pydirectinput.press('e')
        time.sleep(16)
        self.wait_till_begin()
        time.sleep(3)
        pydirectinput.press('o')
        print("Ready or not, Bot, FIGHT!!!!!")

if __name__ == "__main__":
    # Test the functionality here
    time.sleep(1)
    proceedToBoss(1).perform()
