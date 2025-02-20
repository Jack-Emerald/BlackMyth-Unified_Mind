import train1

if __name__ == '__main__':
    '''User Settings'''
    env_config = {
        "PYTESSERACT_PATH": r'C:\Program Files\Tesseract-OCR\tesseract.exe',    # Set the path to PyTesseract
        "MONITOR": 1,           #Set the monitor to use (1,2,3)e
        "DEBUG_MODE": False, #renders the AI vision (pretty scuffed)
        "GAME_MODE": "PVE",     #PVe for elite or PVE for boss
        "BOSS": 1,              #1-6 for PVE (look at walkToBoss.py for boss names) | Is ignored for GAME_MODE PVP
        "DESIRED_FPS": 24       #Set the desired fps (used for actions per second) (24 = 2.4 actions per second) #not implemented yet       #My CPU (i9-13900k) can run the training at about 2.4SPS (steps per secons)
    }
    CREATE_NEW_MODEL = False
         #Create a new model or resume training for an existing model


    '''Start Training'''
    print("💍 BlackMythRL 💍")
    train1.train(CREATE_NEW_MODEL, env_config)