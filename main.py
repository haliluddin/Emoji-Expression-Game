import cv2
import numpy as np
import pygame
import random
import time
from keras.models import load_model

WINDOW_WIDTH, WINDOW_HEIGHT = 640, 480
FPS = 30
FALL_SPEED = 3
LIVES = 3
EMOJI_FOLDER = "images/emojis/"
EMOJI_FILES = [
    "grin.png", "angry.png", "shush.png", "peek.png", "kiss.png", "tongue.png", "scream.png",
]
MODEL_PATH = "keras_Model.h5"
LABELS_PATH = "labels.txt"
CONFIDENCE_THRESHOLD = 0.85
POP_DURATION = 0.2

model = load_model(MODEL_PATH, compile=False)
with open(LABELS_PATH, 'r') as f:
    class_names = [line.strip().split()[1] for line in f.readlines()]

camera = cv2.VideoCapture(0)
time.sleep(1)

pygame.init()
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Emoji Expression Game")
clock = pygame.time.Clock()

emoji_surfaces = {}
for fname in EMOJI_FILES:
    surf = pygame.image.load(EMOJI_FOLDER + fname).convert_alpha()
    emoji_surfaces[fname.split('.')[0]] = pygame.transform.smoothscale(surf, (80, 80))

lives = LIVES
score = 0
emojis = EMOJI_FILES.copy()
random.shuffle(emojis)
current = None
x_pos = y_pos = 0
current_label = None
pop_timer = 0
pop_pos = (0, 0)

running = True
while running:
    dt = clock.tick(FPS) / 1000.0
    for event in pygame.event.get():
        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
            running = False

    ret, frame = camera.read()
    if not ret:
        break
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_rgb = cv2.resize(frame_rgb, (WINDOW_WIDTH, WINDOW_HEIGHT), interpolation=cv2.INTER_AREA)
    bg_surf = pygame.image.frombuffer(frame_rgb.tobytes(), (WINDOW_WIDTH, WINDOW_HEIGHT), 'RGB')

    img = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
    arr = ((img.astype(np.float32).reshape(1,224,224,3)) / 127.5) - 1
    preds = model.predict(arr)
    idx = np.argmax(preds[0])
    label = class_names[idx]
    confidence = preds[0][idx]
    print(f"Predicted: {label} ({confidence*100:.1f}%)")

    if current is None and emojis:
        fname = emojis.pop()
        current_label = fname.split('.')[0]
        current = emoji_surfaces[current_label]
        x_pos = random.randint(0, WINDOW_WIDTH - 80)
        y_pos = -80
    elif current is None and not emojis and pop_timer <= 0:
        break

    screen.blit(bg_surf, (0, 0))

    if pop_timer > 0:
        pop_timer -= dt
        radius = int((POP_DURATION - pop_timer) / POP_DURATION * 50)
        pygame.draw.circle(screen, (255, 255, 0), pop_pos, radius)
        if pop_timer <= 0:
            current = None
    elif current:
        y_pos += FALL_SPEED
        screen.blit(current, (x_pos, y_pos))
        if confidence >= CONFIDENCE_THRESHOLD and label == current_label:
            score += 1
            pop_timer = POP_DURATION
            pop_pos = (x_pos + 40, y_pos + 40)
        elif y_pos > WINDOW_HEIGHT:
            lives -= 1
            current = None

    font = pygame.font.SysFont(None, 30)
    screen.blit(font.render(f"Lives: {lives}", True, (255,255,255)), (10,10))
    screen.blit(font.render(f"Score: {score}", True, (255,255,255)), (10,40))
    pygame.display.flip()

    if lives <= 0:
        break

screen.fill((0,0,0))
font = pygame.font.SysFont(None, 50)
msg = "Game Over" if lives <= 0 else "You Win!"
surf = font.render(msg, True, (255,0,0))
screen.blit(surf, (WINDOW_WIDTH//2 - surf.get_width()//2, WINDOW_HEIGHT//2 - surf.get_height()//2))
pygame.display.flip()
time.sleep(2)

camera.release()
pygame.quit()
