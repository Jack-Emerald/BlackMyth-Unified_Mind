import cv2
import numpy as np
import time
import pytesseract


class EldenReward:
    '''Reward Class'''

    '''Constructor'''

    def __init__(self, config):
        pytesseract.pytesseract.tesseract_cmd = config["PYTESSERACT_PATH"]  # Setting the path to pytesseract.exe
        self.GAME_MODE = config["GAME_MODE"]
        self.DEBUG_MODE = config["DEBUG_MODE"]
        self.prev_hp = 1.0
        self.curr_hp = 1.0
        self.time_since_dmg_taken = time.time()
        self.death = False
        self.previous_charge = 0
        self.curr_charge = 0
        self.previous_boss_hp = 1.0
        self.curr_boss_hp = 1.0
        self.time_since_boss_dmg = time.time()
        self.boss_death = False
        self.game_won = False
        self.image_detection_tolerance = 0.02  # The image detection of the hp bar is not perfect. So we ignore changes smaller than this value. (0.02 = 2%)

    '''Detecting the current player hp'''

    def get_current_hp(self, frame):
        #x, y, w, h = 201, 980, 325, 8
        hp_image = frame[980:980 + 8, 201:201 + 325]  # Cut out the hp bar from the frame

        lower = np.array([0, 0, 100])  # Lower bound for white color
        upper = np.array([180, 160, 230])  # Upper bound for white color

        hsv = cv2.cvtColor(hp_image, cv2.COLOR_RGB2HSV)  # Apply the filter
        mask = cv2.inRange(hsv, lower, upper)  # Also apply

        matches = np.argwhere(mask == 255)  # Number for all the white pixels in the mask
        curr_hp = len(matches) / (hp_image.shape[1] * hp_image.shape[
            0])  # Calculating percent of white pixels in the mask (current hp in percent)

        #curr_hp += 0.02  # Adding +2% of hp for color noise

        if curr_hp >= 0.96:  # If the hp is above 96% we set it to 100% (also color noise fix)
            curr_hp = 1.0

        if self.DEBUG_MODE: print('💊 Health: ', curr_hp)
        return curr_hp

    '''Detecting the current player charge point'''

    def get_current_charge(self, frame):
        charge_points = [(1040, 1782), (1020, 1802), (997, 1815)]

        lower_charge = np.array([0, 0, 200])  # Lower bound for light yellow color (light yellow shades)
        upper_charge = np.array([50, 100, 255])  # Upper bound for light yellow color

        # Print the color of the pixel at charge points
        first_charge = frame[1040, 1782]
        #print(f"Color of pixel at 1 charge: {first_charge}")
        second_charge = frame[1020, 1802]
        #print(f"Color of pixel at 2 charge: {second_charge}")
        third_charge = frame[997, 1815]
        #print(f"Color of pixel at 3 charge: {third_charge}")

        # Initialize the charge point count
        self.curr_charge = 0

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
                self.curr_charge += 1

        if self.DEBUG_MODE: print('Number of charged points: ', self.curr_charge)
        return self.curr_charge

    '''Detecting the current boss hp'''

    def get_boss_hp(self, frame):
        if self.GAME_MODE == "PVE":
            boss_hp_image = frame[913:921, 675:1245]  # cutting frame for boss hp bar (always same size)
        elif self.GAME_MODE == "PVe":
            boss_hp_image = frame[913:921, 757:1152]  # cutting frame for boss hp bar (always same size)

        lower = np.array([0, 0, 175])  # Lower bound for white color
        upper = np.array([180, 30, 255])  # Upper bound for white color

        hsv = cv2.cvtColor(boss_hp_image, cv2.COLOR_RGB2HSV)  # Apply the filter
        mask = cv2.inRange(hsv, lower, upper)

        # self.render_frame(boss_hp_image)
        # self.render_frame(mask)

        matches = np.argwhere(mask == 255)  # Number for all the white pixels in the mask
        boss_hp = len(matches) / (boss_hp_image.shape[1] * boss_hp_image.shape[
            0])  # Calculating percent of white pixels in the mask (current boss hp in percent)

        # same noise problem but the boss hp bar is larger so noise is less of a problem
        if self.DEBUG_MODE: print('👹 Boss HP: ', boss_hp)

        return boss_hp

    '''Detecting if the boss is damaged in PvE'''  # 🚧 This is not implemented yet!!

    def detect_boss_damaged(self, frame):

        if (self.previous_boss_hp - self.curr_boss_hp) > 0.01:  # if there are more than 1 percent pixels change, return true
            return True
        else:
            return False

    '''Debug function to render the frame'''

    def render_frame(self, frame):
        cv2.imshow('debug-render', frame)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()

    '''Update function that gets called every step and returns the total reward and if the agent died or the boss died'''

    def update(self, frame, first_step):
        # 📍 1 Getting current values
        # 📍 2 Hp Rewards
        # 📍 3 Boss Rewards
        # 📍 4 PvP Rewards
        # 📍 5 Total Reward / Return

        '''📍1 Getting/Setting current values'''
        self.curr_hp = self.get_current_hp(frame)

        self.previous_charge = self.curr_charge
        self.curr_charge = self.get_current_charge(frame)

        self.previous_boss_hp = self.curr_boss_hp #record previous hp to see if there is a damage
        self.curr_boss_hp = self.get_boss_hp(frame)

        #eliminate light influence
        if self.curr_boss_hp>self.previous_boss_hp:
            self.curr_boss_hp = self.previous_boss_hp

        if first_step: self.time_since_dmg_taken = time.time() - 10  # Setting the time_since_dmg_taken to 10 seconds ago so we dont get a negative reward at the start of the game

        self.death = False
        if self.curr_hp <= 0.01 + self.image_detection_tolerance:  # If our hp is below 1% we are dead
            self.death = True
            self.curr_hp = 0.0

        self.boss_death = False
        if self.curr_boss_hp <= 0.005:  # If the boss hp is below 1% the boss is dead (small tolerance because we want to be sure the boss is actually dead)
            self.boss_death = True

        '''📍2 Hp Rewards'''
        hp_reward = 0
        if not self.death:
            hp_change = self.curr_hp - self.prev_hp
            if hp_change > 0.02:  # Reward if we healed)
                reward_rate = 1
                if self.prev_hp > 0.7 and hp_change > 0.1:
                    reward_rate = 0.5
                    hp_reward -= 100
                    print("negative heal")
                elif self.prev_hp < 0.5:
                    reward_rate = 1.5
                    hp_reward += 100
                    print("positive heal")
                hp_reward = 200*hp_change*reward_rate
            elif hp_change < self.image_detection_tolerance:  # Negative reward if we took damage
                hp_reward = -250*hp_change
                self.time_since_dmg_taken = time.time()

            if self.curr_hp >0.5: # reward for every step if we remain hp > 50%, encourage high hp.
                hp_reward += 10
            elif self.curr_hp < 0.3:
                hp_reward -= 10 # negative reward for every step if we remain hp < 50%, encourage avoid low hp.
        else:
            #hp_reward = -420  # Large negative reward for dying
            pass

        time_since_taken_dmg_reward = 0
        if time.time() - self.time_since_dmg_taken > 4:  # Reward if we have not taken damage for 5 seconds (every step for as long as we dont take damage)
            time_since_taken_dmg_reward = 25

        self.prev_hp = self.curr_hp  # Update prev_hp to curr_hp



        '''📍3 Boss Rewards'''
        boss_dmg_reward = 0
        if self.boss_death:  # Large reward if the boss is dead
            # boss_dmg_reward = 840
            pass
        else:
            if self.detect_boss_damaged(
                    frame):  # Reward if we damaged the boss (small tolerance because its a large bar)
                boss_dmg_reward = 7500 * (self.previous_boss_hp - self.curr_boss_hp)
                self.time_since_boss_dmg = time.time()
            if time.time() - self.time_since_boss_dmg > 4:  # Negative reward if we have not damaged the boss for 5 seconds (every step for as long as we dont damage the boss)
                boss_dmg_reward = -40

        percent_through_fight_reward = 0
        if self.curr_boss_hp < 0.97:  # Increasing reward for every step we are alive depending on how low the boss hp is
            percent_through_fight_reward = (1 - self.curr_boss_hp) * 20

        # '''📍4 charge rewards'''
        charge_reward = 0
        # charge_change = self.previous_charge - self.curr_charge
        # if charge_change > 0: # reward if we use charge points
        #     charge_reward = 100*charge_change

        '''📍5 Total Reward / Return'''
        total_reward = hp_reward + boss_dmg_reward + charge_reward + time_since_taken_dmg_reward + percent_through_fight_reward
        total_reward = round(total_reward, 3)

        return total_reward, self.death, self.boss_death, self.game_won


'''Testing code'''
if __name__ == "__main__":
    env_config = {
        "PYTESSERACT_PATH": r'C:\Program Files\Tesseract-OCR\tesseract.exe',  # Set the path to PyTesseract
        "MONITOR": 1,  # Set the monitor to use (1,2,3)
        "DEBUG_MODE": False,  # Renders the AI vision (pretty scuffed)
        "GAME_MODE": "PVE",  # PVP or PVE
        "BOSS": 1,  # 1-6 for PVE (look at walkToBoss.py for boss names) | Is ignored for GAME_MODE PVP
        "DESIRED_FPS": 24
        # Set the desired fps (used for actions per second) (24 = 2.4 actions per second) #not implemented yet       #My CPU (i9-13900k) can run the training at about 2.4SPS (steps per secons)
    }
    reward = EldenReward(env_config)

    IMG_WIDTH = 1920  # Game capture resolution
    IMG_HEIGHT = 1080

    import mss

    sct = mss.mss()
    monitor = sct.monitors[1]
    sct_img = sct.grab(monitor)
    frame = cv2.cvtColor(np.asarray(sct_img), cv2.COLOR_BGRA2RGB)
    frame = frame[46:IMG_HEIGHT + 46, 12:IMG_WIDTH + 12]  # cut the frame to the size of the game

    reward.update(frame, True)
    time.sleep(1)
    reward.update(frame, False)

