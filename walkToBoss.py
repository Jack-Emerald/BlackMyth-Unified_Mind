import pydirectinput
import time
import cv2
import mss
import numpy as np

class walkToBoss:
    '''Walk to boss class - hard coded paths from the bonfire to the boss'''

    '''Constructor'''

    def __init__(self, BOSS):
        self.BOSS = BOSS  # Boss number | 99/100 reserved for PVP

    '''Walk to boss function'''

    def perform(self):
        '''PVE'''
        if self.BOSS == 1:
            print("we fight shigandang")
            self.boss1()
        elif self.BOSS == 2:
            print("boss2")
            self.boss1()

        # '''PVP'''
        elif self.BOSS == 99:
            print("boss1")
            self.matchmaking()
        elif self.BOSS == 100:
            print("boss1")
            self.duel_arena_lockon()

        else:
            print("👉👹 Boss not found")

    '''Put on lantern'''

    def wait_till_begin(self):
        # Initialize mss to capture the screen
        sct = mss.mss()

        # Select the first monitor (or adjust based on your setup)
        monitor = sct.monitors[1]

        # Define the threshold for white pixels (full HP)
        lower_white = np.array([5, 5, 215])  # Lower bound for white color
        upper_white = np.array([15, 15, 235])  # Upper bound for white color

        print("wait till begin", end=" ")

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
                print("Get ready.")
                break

            print(".", end=" ")
            time.sleep(0.2)

    '''1 Margit, The fell Omen'''

    def boss1(self):
        #self.put_on_lantern()
        time.sleep(3)
        print("👉👹 challenge the boss.")
        pydirectinput.press('e')
        time.sleep(1)
        pydirectinput.press('e')
        time.sleep(16)
        self.wait_till_begin()
        time.sleep(3)
        pydirectinput.press('o')
        print("👉👹 Fight!!!!!")

    '''2 Beastman of Farum Azula'''

    def boss2(self):
        #self.put_on_lantern()
        print("👉👹 walking #0 from the bonfire")
        pydirectinput.keyDown('w')
        pydirectinput.keyDown('a')
        time.sleep(1.2)
        pydirectinput.keyUp('a')
        time.sleep(2.5)
        pydirectinput.keyDown('d')
        time.sleep(0.5)
        pydirectinput.keyUp('d')
        pydirectinput.keyDown('a')
        time.sleep(0.2)
        pydirectinput.keyUp('a')
        time.sleep(4)
        print("👉👹 walking #1 around the corner")
        pydirectinput.keyDown('d')
        time.sleep(1)
        pydirectinput.keyUp('d')
        time.sleep(3)
        print("👉👹 walking #2 start sprinting")
        pydirectinput.keyDown('shift')
        time.sleep(2.5)
        print("👉👹 walking #3 to the fog gate")
        pydirectinput.keyDown('d')
        time.sleep(0.7)
        pydirectinput.keyUp('d')
        time.sleep(2)
        pydirectinput.keyDown('a')
        time.sleep(0.5)
        pydirectinput.keyUp('a')
        time.sleep(1)
        pydirectinput.keyDown('a')
        time.sleep(0.2)
        pydirectinput.keyUp('a')
        time.sleep(0.6)
        pydirectinput.keyUp('shift')
        pydirectinput.keyUp('w')
        pydirectinput.press('f')
        time.sleep(3.7)
        print("👉👹 walking #4 lock on to the boss")
        pydirectinput.keyDown('w')
        pydirectinput.press('tab')
        time.sleep(1)
        pydirectinput.keyUp('w')

    '''3 Scally misbegotten'''

    def boss3(self):
        #self.put_on_lantern()
        print("👉👹 walking #0 from the bonfire")
        pydirectinput.keyDown('shift')
        pydirectinput.keyDown('w')
        time.sleep(4)
        pydirectinput.keyDown('a')
        print("👉👹 walking #1 around the corner")
        time.sleep(1.5)
        pydirectinput.keyUp('a')
        print("👉👹 walking #2 fall down ledge")
        time.sleep(8)
        pydirectinput.keyDown('a')
        print("👉👹 walking #3 around the corner")
        time.sleep(1.3)
        pydirectinput.keyUp('a')
        print("👉👹 walking #4 to the fog gate")
        time.sleep(8)
        pydirectinput.keyUp('w')
        pydirectinput.keyUp('shift')
        pydirectinput.press('f')
        time.sleep(3.2)
        print("👉👹 walking #1 lock on to the boss")
        pydirectinput.keyDown('w')
        pydirectinput.press('shift')
        time.sleep(1)
        pydirectinput.keyUp('w')
        pydirectinput.press('tab')
        print("👉👹 walking done")

    '''4 Patches (buggy)'''

    def boss4(self):
        #self.put_on_lantern()
        print("👉👹 walking #0 from the bonfire")
        pydirectinput.keyDown('shift')
        pydirectinput.keyDown('w')
        time.sleep(2.1)
        print("👉👹 walking #1 around the corner")
        pydirectinput.keyDown('d')
        time.sleep(0.6)
        pydirectinput.keyUp('d')
        time.sleep(0.35)
        pydirectinput.keyDown('d')
        time.sleep(0.3)
        pydirectinput.keyUp('d')
        time.sleep(0.8)
        print("👉👹 walking #2 around that same corner")
        pydirectinput.keyDown('d')
        time.sleep(0.4)
        pydirectinput.keyUp('d')
        time.sleep(0.1)
        pydirectinput.keyDown('d')
        time.sleep(0.1)
        pydirectinput.keyUp('d')
        pydirectinput.keyDown('a')
        time.sleep(0.4)
        pydirectinput.keyUp('a')
        print("👉👹 walking #3 walking straight")
        time.sleep(3)
        print("👉👹 walking #4 around the corner")
        pydirectinput.keyDown('d')
        time.sleep(1.8)
        print("👉👹 walking #5 down the path")
        pydirectinput.keyUp('d')
        time.sleep(2.5)
        print("👉👹 walking #6 to the fog gate")
        pydirectinput.keyDown('d')
        time.sleep(0.5)
        pydirectinput.keyUp('d')
        time.sleep(2)

        pydirectinput.keyUp('w')
        pydirectinput.keyUp('shift')
        pydirectinput.press('f')
        time.sleep(3.7)
        print("👉👹 walking #7 lock on to the boss")
        pydirectinput.keyDown('w')
        time.sleep(1.2)
        pydirectinput.keyUp('w')
        pydirectinput.press('tab')
        print("👉👹 walking done")

    '''5 Erdtree burrial watchdog'''

    def boss5(self):
        #self.put_on_lantern()
        print("👉👹 walking #0 from the bonfire")
        pydirectinput.keyDown('shift')
        pydirectinput.keyDown('w')
        time.sleep(4.1)
        print("👉👹 walking #1 around the corner")
        pydirectinput.keyDown('d')
        time.sleep(1.7)
        pydirectinput.keyUp('d')
        print("👉👹 walking #2 to the fog gate")
        time.sleep(10)
        pydirectinput.keyUp('w')
        pydirectinput.keyUp('shift')
        pydirectinput.press('f')
        time.sleep(3.7)
        print("👉👹 walking #3 lock on to the boss")
        pydirectinput.keyDown('w')
        time.sleep(0.5)
        pydirectinput.keyUp('w')
        pydirectinput.press('tab')
        print("👉👹 walking done")

    '''6 Graven warden duelist (badly buggy)'''

    def boss6(self):
        #self.put_on_lantern()
        print("👉👹 walking #0 from the bonfire")
        pydirectinput.keyDown('shift')
        pydirectinput.keyDown('w')
        time.sleep(8.3)
        print("👉👹 walking #1 around the corner")
        pydirectinput.keyDown('d')
        time.sleep(1.76)
        pydirectinput.keyUp('d')
        print("👉👹 walking #2 down the hallway")
        time.sleep(1.5)
        pydirectinput.keyUp('shift')
        time.sleep(0.9)
        print("👉👹 walking #3 spam dodge")
        pydirectinput.keyUp('shift')
        pydirectinput.press('shift')
        time.sleep(0.1)
        pydirectinput.press('shift')
        time.sleep(0.15)
        pydirectinput.press('shift')
        time.sleep(0.15)
        pydirectinput.press('shift')
        time.sleep(0.1)
        pydirectinput.press('shift')
        pydirectinput.keyDown('w')
        pydirectinput.keyDown('shift')
        time.sleep(6)
        pydirectinput.keyUp('w')
        pydirectinput.keyUp('shift')
        pydirectinput.press('f')
        time.sleep(3.7)
        print("👉👹 walking #4 lock on to the boss")
        pydirectinput.press('tab')
        pydirectinput.keyDown('w')
        time.sleep(0.5)
        pydirectinput.keyUp('w')
        print("👉👹 walking done")

    '''7 Mad Punpkinhead'''

    def boss7(self):
        #self.put_on_lantern()
        print("👉👹 walking #0 from the bonfire")
        pydirectinput.keyDown('shift')
        pydirectinput.keyDown('w')
        time.sleep(2)
        print("👉👹 walking #1 to the ruins")
        pydirectinput.keyDown('d')
        time.sleep(1)
        pydirectinput.keyUp('d')
        pydirectinput.keyDown('a')
        time.sleep(0.85)
        pydirectinput.keyUp('a')
        time.sleep(1.8)
        print("👉👹 walking #2 into the basement")
        pydirectinput.keyUp('shift')
        pydirectinput.keyDown('a')
        time.sleep(1.5)
        pydirectinput.keyUp('a')
        pydirectinput.keyDown('d')
        time.sleep(4)
        pydirectinput.keyUp('d')
        time.sleep(4.5)
        pydirectinput.keyUp('w')
        pydirectinput.keyUp('shift')
        pydirectinput.press('f')
        time.sleep(3.7)
        print("👉👹 walking #3 lock on to the boss")
        pydirectinput.press('tab')
        time.sleep(0.1)
        print("👉👹 walking done")

    '''8 Malenia blade of miquella'''

    def boss8(self):
        #self.put_on_lantern()
        print("👉👹 walking #1 walk to Malenia")
        pydirectinput.keyDown('shift')
        pydirectinput.keyDown('w')
        time.sleep(10)
        pydirectinput.keyUp('w')
        pydirectinput.keyUp('shift')
        pydirectinput.press('f')
        time.sleep(3.7)
        print("👉👹 walking #2 lock on to the boss")
        pydirectinput.keyDown('w')
        time.sleep(1)
        pydirectinput.keyUp('w')
        pydirectinput.press('tab')
        time.sleep(0.1)
        print("👉👹 walking done")

    '''PvP Matchmaking'''

    def matchmaking(self):
        pydirectinput.press('f')
        time.sleep(0.5)
        pydirectinput.press('up')
        time.sleep(0.5)
        pydirectinput.press('e')
        time.sleep(0.5)
        pydirectinput.press('e')
        print("⚔️ PvP Matchmaking")

    '''PvP Duel Arena Lockon'''

    def duel_arena_lockon(self):
        time.sleep(2)
        pydirectinput.press('tab')
        time.sleep(0.1)
        print("⚔️ Duelist locked on")


# Run the function to test it
def test():

    time.sleep(1)
    walkToBoss(1).perform()


if __name__ == "__main__":
    test()
