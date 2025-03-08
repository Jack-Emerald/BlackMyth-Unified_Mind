# BlackMyth_RL
Reinforcement learning designed for Black Myth: Wukong

Black Myth: Wukong is a Chinese Action RPG released in 2024.8.20. 
In this project, we are trying to train an AI agent to completely defeat bosses in Challenge mode by itself. 
The AI agent should be able to beat the bosses by choosing series of best actions from our provided action poll. 
We can promise this is one of the coolest projects we have ever decided to make.

NOTE:
1. Install all the prerequisite packages listed in requirements.txt.
2. Pytesseract is not just a simple pip install;
Please also download it from https://tesseract-ocr.github.io/tessdoc/Downloads.html;
Make sure the "PYTESSERACT_PATH" setting is your path in main.py.
3. Change env_config in main.py for your customization.
4. Your models and logs will be stored in separate folders by gym and our code.
5. You want to make sure the windowed game is on the left corner of your screen. Do not block your game screen.
