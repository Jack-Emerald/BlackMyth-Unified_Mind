import time

import cv2
import numpy as np
import pytesseract
import mss


class ScreenChecker:
    def __init__(self, debug_mode=False):
        self.DEBUG_MODE = debug_mode

    def check_for_conclusion_screen(self, frame):
        # The way we determine if we are in a loading screen is by checking if the text "return" or "vanquished" is in the screen.
        # If it is we are in a conclusion screen. If it is not we are not in a loading screen.
        vanquish_text_image = frame[150:150 + 50,
                              310:310 + 250]  # Cutting the frame to the location of the text "next" (bottom left corner)
        return_text_image = frame[150:150 + 50,
                              120:120+130]

        # Resize to a more readable size for text extraction (optional based on resolution)
        vanquish_text_image = cv2.resize(vanquish_text_image, ((205 - 155) * 3, (1040 - 1015) * 3))
        return_text_image = cv2.resize(return_text_image, ((205 - 155) * 3, (1040 - 1015) * 3))

        # Define the color range to detect white (for example, in case of a white text on dark background)
        lower = np.array([0, 0, 75])  # Removing color from the image (mostly white tones)
        upper = np.array([255, 255, 255])

        # Convert the image to HSV and create a mask for white areas
        hsv1 = cv2.cvtColor(vanquish_text_image, cv2.COLOR_RGB2HSV)
        mask1 = cv2.inRange(hsv1, lower, upper)

        hsv2 = cv2.cvtColor(return_text_image, cv2.COLOR_RGB2HSV)
        mask2 = cv2.inRange(hsv2, lower, upper)

        # Extract text from the mask using pytesseract
        pytesseract_output1 = pytesseract.image_to_string(mask1, lang='eng', config='--psm 6 --oem 3')
        pytesseract_output2 = pytesseract.image_to_string(mask2, lang='eng', config='--psm 6 --oem 3')

        # Check if "Vanquished" or "vanquished" appears in the extracted text
        player_win = "Van" in pytesseract_output1 or "van" in pytesseract_output1
        boss_win = "Ret" in pytesseract_output2 or "turn" in pytesseract_output2

        #cv2.rectangle(frame, (120, 150), (120 + 130, 150 + 50), (0, 255, 0), 2)  # Drawing rectangle on the frame

        # Debugging output if enabled
        if self.DEBUG_MODE:
            matches = np.argwhere(mask1 == 255)
            percent_match = len(matches) / (mask1.shape[0] * mask1.shape[1])
            print(f"vanquish match percentage: {percent_match * 100:.2f}%")

            matches = np.argwhere(mask2 == 255)
            percent_match = len(matches) / (mask2.shape[0] * mask2.shape[1])
            print(f"return match percentage: {percent_match * 100:.2f}%")

        return player_win, boss_win


# Test the method
if __name__ == "__main__":
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    # Initialize mss to capture the screen
    sct = mss.mss()

    # Select the first monitor (or adjust based on your setup)
    monitor = sct.monitors[1]


    while True:
        time.sleep(2)
        print("Fighting...")
        # Capture the entire screen as an image
        sct_img = sct.grab(monitor)

        # Convert the screenshot to a numpy array and then to RGB format
        frame = cv2.cvtColor(np.asarray(sct_img), cv2.COLOR_BGRA2RGB)

        # Cut the frame to the game window size (based on the offset)
        frame = frame[46:1080 + 46, 12:1920 + 12]  # Adjust based on your window position

        # Create the screen checker instance with debugging enabled
        checker = ScreenChecker(debug_mode=False)

        player_win, boss_win = checker.check_for_conclusion_screen(frame)

        if player_win or boss_win:
            break


        # Display the captured frame with the rectangle drawn around the text region
        #cv2.imshow("Captured Screen with Frame", frame)

        # Wait for a key press to close the window
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
    if boss_win:
        print("Boss wins!", boss_win)
    elif player_win:
        print("Player wins!", player_win)
    else:
        print("error!")


    # Display the captured frame with the rectangle drawn around the text region
    #cv2.imshow("Captured Screen with Frame", frame)

    # Wait for a key press to close the window
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

