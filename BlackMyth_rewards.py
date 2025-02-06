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
        self.max_hp = config[
            "PLAYER_HP"]  # This is the hp value of your character. We need this to capture the right length of the hp bar.
        self.prev_hp = 1.0
        self.curr_hp = 1.0
        self.time_since_dmg_taken = time.time()
        self.death = False
        self.max_stam = config["PLAYER_STAMINA"]
        self.previous_charge = 0
        self.curr_charge = 0
        self.previous_boss_hp = 1.0
        self.curr_boss_hp = 1.0
        self.time_since_boss_dmg = time.time()
        self.time_since_pvp_damaged = time.time()
        self.time_alive = time.time()
        self.boss_death = False
        self.game_won = False
        self.image_detection_tolerance = 0.02  # The image detection of the hp bar is not perfect. So we ignore changes smaller than this value. (0.02 = 2%)

    '''Detecting the current player hp'''

    def get_current_hp(self, frame):
        #x, y, w, h = 201, 980, 325, 8
        hp_image = frame[980:980 + 8, 201:201 + 325]  # Cut out the hp bar from the frame
        if self.DEBUG_MODE: self.render_frame(hp_image)

        lower = np.array([0, 0, 175])  # Lower bound for white color
        upper = np.array([180, 30, 255])  # Upper bound for white color

        hsv = cv2.cvtColor(hp_image, cv2.COLOR_RGB2HSV)  # Apply the filter
        mask = cv2.inRange(hsv, lower, upper)  # Also apply
        if self.DEBUG_MODE: self.render_frame(mask)

        matches = np.argwhere(mask == 255)  # Number for all the white pixels in the mask
        curr_hp = len(matches) / (hp_image.shape[1] * hp_image.shape[
            0])  # Calculating percent of white pixels in the mask (current hp in percent)

        #curr_hp += 0.02  # Adding +2% of hp for color noise

        if curr_hp >= 0.96:  # If the hp is above 96% we set it to 100% (also color noise fix)
            curr_hp = 1.0

        if self.DEBUG_MODE: print('💊 Health: ', curr_hp)
        return curr_hp

    '''Detecting the current player charge point'''

    def get_current_stamina(self, frame):
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
        boss_hp_image = frame[913:921, 675:1245]  # cutting frame for boss hp bar (always same size)
        if self.DEBUG_MODE: self.render_frame(boss_hp_image)

        lower = np.array([0, 0, 175])  # Lower bound for white color
        upper = np.array([180, 30, 255])  # Upper bound for white color

        hsv = cv2.cvtColor(boss_hp_image, cv2.COLOR_RGB2HSV)  # Apply the filter
        mask = cv2.inRange(hsv, lower, upper)
        if self.DEBUG_MODE: self.render_frame(mask)

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

    '''Detecting if the enemy is damaged in PvP'''

    def detect_pvp_damaged(self, frame):
        cut_frame = frame[150:400, 350:1700]

        lower = np.array(
            [24, 210, 0])  # This filter really inst perfect but its good enough bcause stamina is not that important
        upper = np.array([25, 255, 255])  # Also Filter
        hsv = cv2.cvtColor(cut_frame, cv2.COLOR_RGB2HSV)  # Apply the filter
        mask = cv2.inRange(hsv, lower, upper)  # Also apply
        matches = np.argwhere(mask == 255)  # Number for all the white pixels in the mask
        if len(matches) > 30:  # if there are more than 30 white pixels in the mask, return true
            return True
        else:
            return False

    '''Detecting if the duel is won in PvP'''

    def detect_win(self, frame):
        cut_frame = frame[730:800, 550:1350]
        lower = np.array([0, 0, 75])  # Removing color from the image
        upper = np.array([255, 255, 255])
        hsv = cv2.cvtColor(cut_frame, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        pytesseract_output = pytesseract.image_to_string(mask, lang='eng',
                                                         config='--psm 6 --oem 3')  # reading text from the image cutout
        game_won = "Combat ends in your victory!" in pytesseract_output or "combat ends in your victory!" in pytesseract_output  # Boolean if we see "combat ends in your victory!" on the screen
        return game_won

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
        self.curr_charge = self.get_current_stamina(frame)

        self.previous_boss_hp = self.curr_boss_hp #record previous hp to see if there is a damage
        self.curr_boss_hp = self.get_boss_hp(frame)

        if first_step: self.time_since_dmg_taken = time.time() - 10  # Setting the time_since_dmg_taken to 10 seconds ago so we dont get a negative reward at the start of the game

        self.death = False
        if self.curr_hp <= 0.01 + self.image_detection_tolerance:  # If our hp is below 1% we are dead
            self.death = True
            self.curr_hp = 0.0

        self.boss_death = False
        if self.GAME_MODE == "PVE":  # Only if we are in PVE mode
            if self.curr_boss_hp <= 0.1:  # If the boss hp is below 1% the boss is dead (small tolerance because we want to be sure the boss is actually dead)
                self.boss_death = True

        '''📍2 Hp Rewards'''
        hp_reward = 0
        if not self.death:
            if self.curr_hp > self.prev_hp + self.image_detection_tolerance:  # Reward if we healed)
                hp_reward = 300
            elif self.curr_hp < self.prev_hp - self.image_detection_tolerance:  # Negative reward if we took damage
                hp_reward = -69
                self.time_since_dmg_taken = time.time()
        else:
            hp_reward = -420  # Large negative reward for dying

        time_since_taken_dmg_reward = 0
        if time.time() - self.time_since_dmg_taken > 5:  # Reward if we have not taken damage for 5 seconds (every step for as long as we dont take damage)
            time_since_taken_dmg_reward = 25

        self.prev_hp = self.curr_hp  # Update prev_hp to curr_hp



        '''📍3 Boss Rewards'''
        if self.GAME_MODE == "PVE":  # Only if we are in PVE mode
            boss_dmg_reward = 0
            if self.boss_death:  # Large reward if the boss is dead
                boss_dmg_reward = 840
            else:
                if self.detect_boss_damaged(
                        frame):  # Reward if we damaged the boss (small tolerance because its a large bar)
                    boss_dmg_reward = 40
                    self.time_since_boss_dmg = time.time()
                if time.time() - self.time_since_boss_dmg > 5:  # Negative reward if we have not damaged the boss for 5 seconds (every step for as long as we dont damage the boss)
                    boss_dmg_reward = -25

            percent_through_fight_reward = 0
            if self.curr_boss_hp < 0.97:  # Increasing reward for every step we are alive depending on how low the boss hp is
                percent_through_fight_reward = self.curr_boss_hp * 15

        '''📍4 charge rewards'''
        charge_reward = 0
        charge_change = self.previous_charge - self.curr_charge
        if charge_change > 0: # reward if we use charge points
            charge_reward = 20*charge_change


        '''📍4 PVP rewards'''
        pvp_reward = 0
        if self.GAME_MODE != "PVE":  # Only if we are in PVP mode
            enemy_damaged = self.detect_pvp_damaged(frame)  # Detect if the enemy is damaged
            if enemy_damaged:  # Reward if the enemy is damaged
                pvp_reward = 69
                self.time_since_pvp_damaged = time.time()
            else:
                if time.time() - self.time_since_pvp_damaged > 5:  # Negative reward if we have not damaged the enemy for 5 seconds (every step for as long as we dont damage the enemy)
                    pvp_reward = -25
                    # print("🔫 Duelist not damaged for 5s")
                else:
                    pvp_reward = 0

            # staying alive reward
            '''                     #a time alive reward could cause problems because the agent will still get rewarded even if performing bad when the time alive reward is higher than the other punishments
            time_alive_reward = 0
            if time.time() - self.time_alive > 5:                                   #Reward if we have been alive for 5 seconds (we give an increasinig reward for every second we are alive)
                time_alive_reward = time.time() - self.time_alive - 5
                print("🕒 Time alive reward: ", time_alive_reward)
                pvp_reward += time_alive_reward
            '''

            # winning
            self.game_won = self.detect_win(frame)  # not implemented yet
            if self.game_won:
                pvp_reward = 420

        '''📍5 Total Reward / Return'''
        if self.GAME_MODE == "PVE":  # Only if we are in PVE mode
            total_reward = hp_reward + boss_dmg_reward + charge_reward + time_since_taken_dmg_reward + percent_through_fight_reward
        else:
            total_reward = hp_reward + time_since_taken_dmg_reward + pvp_reward

        total_reward = round(total_reward, 3)

        return total_reward, self.death, self.boss_death, self.game_won


'''Testing code'''
if __name__ == "__main__":
    env_config = {
        "PYTESSERACT_PATH": r'C:\Program Files\Tesseract-OCR\tesseract.exe',  # Set the path to PyTesseract
        "MONITOR": 1,  # Set the monitor to use (1,2,3)
        "DEBUG_MODE": False,  # Renders the AI vision (pretty scuffed)
        "GAME_MODE": "PVE",  # PVP or PVE
        "BOSS": 8,  # 1-6 for PVE (look at walkToBoss.py for boss names) | Is ignored for GAME_MODE PVP
        "BOSS_HAS_SECOND_PHASE": True,  # Set to True if the boss has a second phase (only for PVE)
        "PLAYER_HP": 1679,  # Set the player hp (used for hp bar detection)
        "PLAYER_STAMINA": 121,  # Set the player stamina (used for stamina bar detection)
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

