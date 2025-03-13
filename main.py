import train1

if __name__ == '__main__':
    '''User Settings'''
    env_config = {
        "PYTESSERACT_PATH": r'C:\Program Files\Tesseract-OCR\tesseract.exe', # Set the path to your tesseract.exe
        "MONITOR": 1,           # Set the monitor to use (1,2,3)
        "DEBUG_MODE": False,    # Debug Mode (more things(variables) get printed)
        "GAME_MODE": "PVE",     # PVe for elite (shorter hp bars) / PVE for boss
        "BOSS": 2,              # Only implemented one Configuration for now: challenge mode bosses
        "DESIRED_FPS": 24       # Set the desired fps (used for actions per second) (24 = 2.4 actions per second)
    }
    # Create a new model or resume training for an existing model
    CREATE_NEW_MODEL = False

    '''Start Training'''
    print("##### Welcome to BlackMythRL #####")
    train1.train(CREATE_NEW_MODEL, env_config)
