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

pygame.mixer.init()
pygame.init()
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Emoji Expression Game")
clock = pygame.time.Clock()

pygame.mixer.music.load("images/music/background.mp3")
pygame.mixer.music.play(-1)
pop_sound = pygame.mixer.Sound("images/music/pop.mp3")
fail_sound = pygame.mixer.Sound("images/music/fail.mp3")

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

state = 'start'
start_button = pygame.Rect(WINDOW_WIDTH//2 - 60, WINDOW_HEIGHT - 60, 120, 40)
retry_button = pygame.Rect(WINDOW_WIDTH//2 - 60, WINDOW_HEIGHT - 60, 120, 40)
font = pygame.font.SysFont(None, 30)
large_font = pygame.font.SysFont(None, 50)

running = True
while running:
    dt = clock.tick(FPS) / 1000.0
    for event in pygame.event.get():
        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mx, my = event.pos
            if state == 'start' and start_button.collidepoint(mx, my):
                state = 'playing'
            elif state == 'end' and retry_button.collidepoint(mx, my):
                lives = LIVES
                score = 0
                emojis = EMOJI_FILES.copy()
                random.shuffle(emojis)
                current = None
                pop_timer = 0
                state = 'playing'

    ret, frame = camera.read()
    if not ret:
        break
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_rgb = cv2.resize(frame_rgb, (WINDOW_WIDTH, WINDOW_HEIGHT), interpolation=cv2.INTER_AREA)
    bg_surf = pygame.image.frombuffer(frame_rgb.tobytes(), (WINDOW_WIDTH, WINDOW_HEIGHT), 'RGB')

    screen.blit(bg_surf, (0, 0))

    if state == 'start':
        instr = font.render("Click START to begin the Emoji Expression Game", True, (255, 255, 255))
        screen.blit(instr, (WINDOW_WIDTH//2 - instr.get_width()//2, 10))
        pygame.draw.rect(screen, (0, 128, 255), start_button)
        txt = font.render("START", True, (255, 255, 255))
        screen.blit(txt, (start_button.x + 30, start_button.y + 10))
    elif state == 'playing':
        img = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
        arr = ((img.astype(np.float32).reshape(1,224,224,3)) / 127.5) - 1
        preds = model.predict(arr)
        idx = np.argmax(preds[0]); label = class_names[idx]; confidence = preds[0][idx]

        if current is None and emojis:
            fname = emojis.pop()
            current_label = fname.split('.')[0]
            current = emoji_surfaces[current_label]
            x_pos = random.randint(0, WINDOW_WIDTH - 80)
            y_pos = -80
        elif current is None and not emojis and pop_timer <= 0:
            state = 'end'

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
                pop_sound.play()
                pop_timer = POP_DURATION
                pop_pos = (x_pos + 40, y_pos + 40)
            elif y_pos > WINDOW_HEIGHT:
                lives -= 1
                fail_sound.play()
                current = None
                if lives <= 0:
                    state = 'end'

        screen.blit(font.render(f"Lives: {lives}", True, (255,255,255)), (10,10))
        screen.blit(font.render(f"Score: {score}", True, (255,255,255)), (10,40))
    elif state == 'end':
        msg_text = "You Win!" if lives > 0 else "Game Over"
        msg_surf = large_font.render(msg_text, True, (255, 0, 0))
        screen.blit(msg_surf, (WINDOW_WIDTH//2 - msg_surf.get_width()//2, 10))
        pygame.draw.rect(screen, (0, 128, 0), retry_button)
        rt = font.render("Try Again", True, (255, 255, 255))
        screen.blit(rt, (retry_button.x + 10, retry_button.y + 10))

    pygame.display.flip()

camera.release()
pygame.quit()
