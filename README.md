# Emoji-Expression-Game

This game is an interactive, camera-driven challenge where players match their facial expressions to falling emojis before they hit the bottom of the screen. After launching, a start screen displays a looping backdrop and animated mini-emojis, plus clear instructions and a “START” button. Once playing, live video feeds into the background while a stream of larger emojis descends at increasing speed; the Keras model continuously predicts the player’s expression and, if it matches the current emoji with sufficient confidence, the emoji “pops” with a satisfying sound effect and the player earns a point. Missed emojis deduct a life and play a failure sound, and the game ends when lives run out or all emojis have been popped—triggering a “You Win!” or “You Lose!” overlay complete with celebratory or consoling graphics and their own audio cues. Throughout, a polished GUI layer anchors lives and score displays, background music loops, and retry functionality invites immediate replay.

![image alt](https://github.com/haliluddin/Emoji-Expression-Game/blob/2b8b2de97dc1ca7dd5dec1df276ece0bca526b64/sample.png)

Instructions on How to Run:
1. Clone this repository.
2. Open the folder in your preferred Python IDE (e.g., PyCharm, VS Code).
3. In your terminal, run this "pip install wheel pillow numpy tensorflow==2.15 opencv-python teachable-machine pygame".
4. Execute the main.py file
