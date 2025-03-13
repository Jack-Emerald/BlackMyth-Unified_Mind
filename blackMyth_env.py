import cv2
import gym
import mss
import time
import numpy as np
from gym import spaces
import pydirectinput
from blackMyth_rewards import BlackMythReward
from proceedToBoss import proceedToBoss
# Pytesseract is not just a simple pip install;
# Please also download it from https://tesseract-ocr.github.io/tessdoc/Downloads.html;
# Make sure the "PYTESSERACT_PATH" setting is your path in main.py
import pytesseract

N_CHANNELS = 3              # Image format
IMG_WIDTH = 1920            # Game capture resolution
IMG_HEIGHT = 1080           # default 1080p
MODEL_WIDTH = int(800 / 2)  # Ai vision resolution
MODEL_HEIGHT = int(450 / 2)

'''Ai action list'''
DISCRETE_ACTIONS = {
                    'w': 'run_forwards',
                    's': 'run_backwards',
                    'space': 'dodge',
                    's+space': 'dodge_backwards',
                    'k': 'attack',
                    'h': 'strong_attack',
                    'hold h': 'charge_strong_attack',
                    'unhold h':'do charge_strong_attack',
                    '1': 'spell1',
                    '2': 'spell2',
                    '3': 'spell3',
                    'r':'heal'
                    }

NUMBER_DISCRETE_ACTIONS = len(DISCRETE_ACTIONS)
NUM_ACTION_HISTORY = 10  # Number of actions the agent can remember


class BlackMythEnv(gym.Env):
    """Custom Black Myth: Wukong Environment that follows gym interface"""

    def __init__(self, config):
        """Setting up the environment"""
        super(BlackMythEnv, self).__init__()

        '''Setting up the gym spaces'''
        # Discrete action space with NUM_ACTION_HISTORY actions to choose from
        self.action_space = spaces.Discrete(NUMBER_DISCRETE_ACTIONS)
        # Observation space (img, prev_actions, state)
        spaces_dict = {
            'img': spaces.Box(low=0, high=255,
                              shape=(MODEL_HEIGHT, MODEL_WIDTH, N_CHANNELS),
                              dtype=np.uint8), # Image of the game
            'prev_actions': spaces.Box(low=0, high=1,
                                       shape=(NUM_ACTION_HISTORY, NUMBER_DISCRETE_ACTIONS, 1),
                                       dtype=np.uint8),  # Last 10 actions as one hot encoded array
            'state': spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32), # health of the player and the boss in percent
        }
        self.observation_space = gym.spaces.Dict(spaces_dict)

        '''Setting up the variables'''''
        pytesseract.pytesseract.tesseract_cmd = config["PYTESSERACT_PATH"]  # Setting the path to pytesseract.exe
        self.sct = mss.mss()  # Initializing CV2 and MSS (used to take screenshots)
        self.reward = 0  # Reward of the previous step
        self.rewardGen = BlackMythReward(config)  # Setting up the reward generator class
        self.death = False  # If the agent died
        self.duel_won = False  # If the agent won the duel
        self.t_start = time.time()  # Time when the training started
        self.done = False  # If the game is done
        self.step_iteration = 0  # Current iteration (number of steps taken in this fight)
        self.first_step = True  # If this is the first step
        self.max_reward = None  # The maximum reward that the agent has gotten in this fight
        self.reward_history = []  # Array of the rewards to calculate the average reward of fight
        self.action_history = []  # Array of the actions that the agent took.
        self.time_since_heal = time.time()  # Time since the last heal
        self.time_since_spell1 = time.time()
        self.time_since_spell2 = time.time()
        self.time_since_spell3 = time.time()
        self.time_since_charge = time.time()
        self.action_name = ''  # Name of the action for logging
        self.MONITOR = config["MONITOR"]  # Monitor to use
        self.DEBUG_MODE = config["DEBUG_MODE"]  # If we are in debug mode
        self.GAME_MODE = config["GAME_MODE"]  # If we are in PVP or PVE mode
        self.DESIRED_FPS = config["DESIRED_FPS"]  # Desired FPS
        self.proceed_to_boss = proceedToBoss(config["BOSS"])  # Class to proceed to the boss


    '''One hot encoding of the last 10 actions'''
    @staticmethod
    def oneHotPrevActions(actions):
        oneHot = np.zeros(shape=(NUM_ACTION_HISTORY, NUMBER_DISCRETE_ACTIONS, 1))
        for i in range(NUM_ACTION_HISTORY):
            if len(actions) >= (i + 1):
                oneHot[i][actions[-(i + 1)]][0] = 1
        # print(oneHot)
        return oneHot

    '''Grabbing a screenshot of the game'''
    def grab_screen_shot(self):
        monitor = self.sct.monitors[self.MONITOR]
        sct_img = self.sct.grab(monitor)
        frame = cv2.cvtColor(np.asarray(sct_img), cv2.COLOR_BGRA2RGB)
        frame = frame[46:IMG_HEIGHT + 46, 12:IMG_WIDTH + 12]  # cut the frame to the size of the game
        return frame

    '''Rendering the frame for debugging'''
    @staticmethod
    def render_frame(frame):
        cv2.imshow('debug-render', frame)
        cv2.waitKey(100)
        cv2.destroyAllWindows()

    '''Defining the actions that the agent can take'''
    def take_action(self, action):
        # action = -1 #Uncomment this for emergency block all actions
        if action == 0:
            pydirectinput.keyUp('w')
            pydirectinput.keyUp('s')
            pydirectinput.keyDown('w')
            self.action_name = 'w'
        elif action == 1:
            pydirectinput.keyUp('w')
            pydirectinput.keyUp('s')
            pydirectinput.keyDown('s')
            self.action_name = 's'
        elif action == 2:
            pydirectinput.press('space')
            self.action_name = 'dodge'
        elif action == 3:
            pydirectinput.keyDown('s')
            pydirectinput.press('space')
            self.action_name = 'dodge-backward'
        elif action == 4:
            pydirectinput.press('k')
            self.action_name = 'attack'
        elif action == 5 and (time.time() - self.time_since_charge) > 3:
            pydirectinput.press('h')
            self.action_name = 'heavy'
        elif action == 6:
            pydirectinput.keyDown('h')
            self.time_since_charge = time.time()
            self.action_name = 'charge heavy start'
        elif action == 7 and (time.time() - self.time_since_charge) > 3:
            pydirectinput.keyUp('h')
            self.action_name = 'charge heavy stop'
        elif action == 8 and (time.time() - self.time_since_spell1) > 50:
            pydirectinput.press('1')
            self.time_since_spell1 = time.time()
            self.action_name = 'spell1'
        elif action == 9 and (time.time() - self.time_since_spell2) > 32:
            pydirectinput.press('2')
            self.time_since_spell2 = time.time()
            self.action_name = 'spell2'
        elif action == 10 and (time.time() - self.time_since_spell3) > 120:
            pydirectinput.press('3')
            self.time_since_spell3 = time.time()
            self.action_name = 'spell3'
        elif action == 11 and (time.time() - self.time_since_heal) > 2:  # prevent spamming heal we only allow it to be pressed every 1.5 seconds
            pydirectinput.press('r')  # item
            self.time_since_heal = time.time()
            self.action_name = 'heal'

    def check_for_conclusion_screen(self):
        """Wait until a winner (either the boss or the player) is clear"""
        while True:
            time.sleep(3)

            frame = self.grab_screen_shot()
            # The way we determine if we are in a loading screen is by checking if the text "return" or "vanquished" is in the screen.
            # If it is we are in a conclusion screen. If it is not we are not in a loading screen.
            vanquish_text_image = frame[150:150 + 50,
                                        310:310 + 250]  # Cutting the frame to the location of the text "next" (bottom left corner)
            return_text_image = frame[150:150 + 50,
                                      120:120 + 130]

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
            player_win = "Van" in pytesseract_output1 or "ed" in pytesseract_output1
            boss_win = "Ret" in pytesseract_output2 or "turn" in pytesseract_output2

            if player_win or boss_win:
                break

            # cv2.rectangle(frame, (120, 150), (120 + 130, 150 + 50), (0, 255, 0), 2)  # Drawing rectangle on the frame
            # Debugging output if enabled
            if self.DEBUG_MODE:
                matches = np.argwhere(mask1 == 255)
                percent_match = len(matches) / (mask1.shape[0] * mask1.shape[1])
                print(f"vanquish match percentage: {percent_match * 100:.2f}%")

                matches = np.argwhere(mask2 == 255)
                percent_match = len(matches) / (mask2.shape[0] * mask2.shape[1])
                print(f"return match percentage: {percent_match * 100:.2f}%")

        return player_win, boss_win

    '''Step function that is called by train1.py'''
    def step(self, action):
        # 1. Collect the current observation
        # 2. Collect the reward based on the observation (reward of previous step)
        # 3. Check if the game is done (player died, boss died, 10minute time limit reached)
        # 4. Take the next action (based on the decision of the agent)
        # 5. Ending the step
        # 6. Returning the observation, the reward, if we are done, and the info
        # 7*. train1.py decides the next action and calls step again

        if self.first_step:
            print("#1: Collecting the current observation...")

        '''Grabbing variables'''
        t_start = time.time()  # Start time of this step
        '''1. Collect the current observation'''
        frame = self.grab_screen_shot()
        '''2. Collect the reward based on the observation (reward of previous step)'''
        self.reward, self.death, self.boss_death, self.duel_won = self.rewardGen.update(frame,self.first_step)
        '''3. Checking if the game is done, adding awards here'''
        player_win, boss_win = False, False
        if self.death or self.boss_death:
            self.done = True
            print('###Game Is Finished!###')
            player_win, boss_win = self.check_for_conclusion_screen()
            self.death, self.boss_death = not player_win, not boss_win
            if self.death:
                print("Result: Player dead")
            else:
                print("Result: boss dead")
        elif (time.time() - self.t_start) > 600:
            self.done = True
            #self.take_action(99)  # Need to re-challenge the boss
            print('Step done (due to time limit)')
            player_win, boss_win = self.check_for_conclusion_screen()
            self.death, self.boss_death = not player_win, not boss_win


        if player_win:
            print("Adding player win reward...")
            self.reward += 2000
        elif boss_win:
            print("Minus player dead reward...")
            self.reward -= 2000

        if self.DEBUG_MODE:
            print('@@@ Reward: ', self.reward)
            print('@@@ player wins: ', player_win)
            print('@@@ boss wins: ', boss_win)


        '''4. Taking the action'''
        if not self.done:
            self.take_action(action)

        '''5. Ending the step'''

        '''Return values'''
        info = {}  # Empty info for gym
        observation = cv2.resize(frame, (MODEL_WIDTH,
                                         MODEL_HEIGHT))  # We resize the frame so the agent doesn't have to deal with a 1920x1080 image (400x225)
        if self.DEBUG_MODE: self.render_frame(observation)  # If we are in debug mode we render the frame
        if self.max_reward is None:  # Max reward
            self.max_reward = self.reward
        elif self.max_reward < self.reward:
            self.max_reward = self.reward
        self.reward_history.append(self.reward)  # Reward history
        spaces_dict = {  # Combining the observations into one dictionary like gym wants it
            'img': observation,
            'prev_actions': self.oneHotPrevActions(self.action_history),
            'state': np.asarray([self.rewardGen.curr_hp, self.rewardGen.curr_boss_hp])
        }

        '''Other variables that need to be updated'''
        self.first_step = False
        self.step_iteration += 1
        self.action_history.append(int(action))  # Appending the action to the action history

        '''FPS LIMITER'''
        t_end = time.time()
        desired_fps = (1 / self.DESIRED_FPS) # Adjust DESIRED_FPS if your CPU is slower
        time_to_sleep = desired_fps - (t_end - t_start)
        if time_to_sleep > 0:
            time.sleep(time_to_sleep)
        '''END FPS LIMITER'''

        current_fps = str(round(((1 / ((t_end - t_start) * 10)) * 10), 1))  # Current SPS (steps per second)

        '''Console output of the step'''
        if not self.done:  # Lots of python string formatting to make the console output look nice
            self.reward = round(self.reward, 0)
            reward_with_spaces = str(self.reward)
            for i in range(5 - len(reward_with_spaces)):
                reward_with_spaces = ' ' + reward_with_spaces
            max_reward_with_spaces = str(self.max_reward)
            for i in range(5 - len(max_reward_with_spaces)):
                max_reward_with_spaces = ' ' + max_reward_with_spaces
            for i in range(18 - len(str(self.action_name))):
                self.action_name = ' ' + self.action_name
            for i in range(5 - len(current_fps)):
                current_fps = ' ' + current_fps
            print('~~ Iteration: ' + str(
                self.step_iteration) + '| FPS: ' + current_fps + '| Reward: ' + reward_with_spaces + '| Max Reward: ' + max_reward_with_spaces + '| Action: ' + str(
                self.action_name))
        else:  # If the game is done (Logging Reward for dying or winning)
            print('~~~~ Reward: ' + str(self.reward) + '| Max Reward: ' + str(self.max_reward))

        # 6. Returning the observation, the reward, if we are done, and the info
        return spaces_dict, self.reward, self.done, info

    '''Reset function that is called if the game is done'''

    def reset(self):
        # 1. Clear any held down keys
        # 2. Print the average reward for the last run
        # 3. Proceeding back to the boss
        # 4. Reset all variables
        # 5. Create the first observation for the first step and return it

        print('% Reset called...')

        ''' 1.Clear any held down keys'''
        self.take_action(0)
        print('% Unholding keys...')

        ''' 2. Print the average reward for the last run'''
        if len(self.reward_history) > 0:
            total_r = 0
            for r in self.reward_history:
                total_r += r
            avg_r = total_r / len(self.reward_history)
            print('% Average reward for last run:', avg_r)

            # Save avg_r to a file for later use
            if avg_r is not None:
                with open("average_rewards.txt", "a") as file:
                    file.write(f"{avg_r}\n")

        ''' 3. Proceeding to the boss'''
        print("% Proceeding to boss")
        self.proceed_to_boss.perform()  # This is hard coded in proceedToBoss.py

        if self.death:  # Death counter in txt file
            f = open("deathCounter.txt", "r")
            deathCounter = int(f.read())
            f.close()
            deathCounter += 1
            f = open("deathCounter.txt", "w")
            f.write(str(deathCounter))
            f.close()

        '''4. Reset all variables'''
        self.step_iteration = 0
        self.reward_history = []
        self.done = False
        self.first_step = True
        self.max_reward = None
        self.rewardGen.prev_hp = 1.0
        self.rewardGen.curr_hp = 1.0
        self.rewardGen.previous_charge = 0
        self.rewardGen.curr_charge = 0
        self.rewardGen.time_since_dmg_taken = time.time()
        self.rewardGen.curr_boss_hp = 1
        self.rewardGen.prev_boss_hp = 1
        self.action_history = []
        self.t_start = time.time()

        '''5. Return the first observation'''
        frame = self.grab_screen_shot()
        observation = cv2.resize(frame,
                                 (MODEL_WIDTH, MODEL_HEIGHT))  # Reset also returns the first observation for the agent
        spaces_dict = {
            'img': observation,  # The image
            'prev_actions': self.oneHotPrevActions(self.action_history),  # The last 10 actions (empty)
            'state': np.asarray([1.0, 1.0])  # Full hp and zero charge
        }

        print('% Reset done.')
        return spaces_dict  # return the new observation

    '''No render function implemented (just look at the game)'''
    def render(self, mode='human'):
        pass

    '''Closing the environment (not used)'''
    def close(self):
        self.cap.release()
