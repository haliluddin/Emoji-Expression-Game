import cv2
import numpy as np
import pygame
import random
import time
from keras.models import load_model

WINDOW_WIDTH, WINDOW_HEIGHT = 360, 640
FPS = 30
FALL_SPEED = 8
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
win_sound = pygame.mixer.Sound("images/music/win.mp3")
lose_sound = pygame.mixer.Sound("images/music/lose.mp3")
start_sound = pygame.mixer.Sound("images/music/start.mp3")

gui_bg = pygame.image.load("images/graphics/background.png").convert_alpha()
gui_bg = pygame.transform.smoothscale(gui_bg, (WINDOW_WIDTH, gui_bg.get_height() * WINDOW_WIDTH // gui_bg.get_width()))
win_img = pygame.image.load("images/graphics/win.png").convert_alpha()
lose_img = pygame.image.load("images/graphics/lose.png").convert_alpha()
img_scale = 200
win_img = pygame.transform.smoothscale(win_img, (img_scale, img_scale))
lose_img = pygame.transform.smoothscale(lose_img, (img_scale, img_scale))

gui_bg_y = WINDOW_HEIGHT - gui_bg.get_height()

lives_icon = pygame.image.load("images/graphics/lives.png").convert_alpha()
lives_icon = pygame.transform.smoothscale(lives_icon, (18, 18))

emoji_surfaces = {}
for fname in EMOJI_FILES:
    surf = pygame.image.load(EMOJI_FOLDER + fname).convert_alpha()
    emoji_surfaces[fname.split('.')[0]] = pygame.transform.smoothscale(surf, (80, 80))

gui_surfaces = [pygame.transform.smoothscale(pygame.image.load(EMOJI_FOLDER + f).convert_alpha(), (40,40)) for f in EMOJI_FILES]
gui_emojis = []
for _ in range(12):
    surf = random.choice(gui_surfaces)
    gui_emojis.append({'surf': surf,
                       'x': random.randint(0, WINDOW_WIDTH - surf.get_width()),
                       'y': random.randint(-WINDOW_HEIGHT, 0),
                       'speed': random.uniform(50, 150)})

lives = LIVES
score = 0
emojis = EMOJI_FILES.copy()
random.shuffle(emojis)
current = None
x_pos = y_pos = 0
current_label = None
pop_timer = 0
pop_pos = (0,0)

state = 'start'
btn_w, btn_h = 150, 50
btn_x = WINDOW_WIDTH//2 - btn_w//2
btn_y = WINDOW_HEIGHT - 190
start_button = pygame.Rect(btn_x, btn_y, btn_w, btn_h)
retry_button = pygame.Rect(btn_x, btn_y, btn_w, btn_h)

font = pygame.font.SysFont(None, 28)
large_font = pygame.font.SysFont(None, 34)
score_font = pygame.font.SysFont(None, 66)

while True:
    dt = clock.tick(FPS) / 1000.0
    for event in pygame.event.get():
        if event.type == pygame.QUIT or (event.type==pygame.KEYDOWN and event.key==pygame.K_ESCAPE):
            pygame.quit()
            camera.release()
            exit()
        if event.type==pygame.MOUSEBUTTONDOWN and event.button==1:
            mx,my = event.pos
            if state=='start' and start_button.collidepoint(mx,my):
                start_sound.play()
                state='playing'
            elif state=='end' and retry_button.collidepoint(mx,my):
                start_sound.play()
                lives=LIVES
                score=0
                emojis=EMOJI_FILES.copy()
                random.shuffle(emojis)
                current=None
                pop_timer=0
                state='playing'

    ret, frame = camera.read()
    if not ret:
        break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (WINDOW_WIDTH,WINDOW_HEIGHT), interpolation=cv2.INTER_AREA)
    bg = pygame.image.frombuffer(frame.tobytes(), (WINDOW_WIDTH,WINDOW_HEIGHT),'RGB')

    screen.blit(bg,(0,0))
    screen.blit(gui_bg,(0,gui_bg_y))

    if state=='start':
        for e in gui_emojis:
            e['y']+=e['speed']*dt
            if e['y']>WINDOW_HEIGHT:
                e['y']=-e['surf'].get_height()
                e['x']=random.randint(0,WINDOW_WIDTH-e['surf'].get_width())
            screen.blit(e['surf'],(e['x'],e['y']))
        lines=["Match your facial expression","to the falling emoji before it","hits the ground."]
        y=40
        for l in lines:
            s=large_font.render(l,True,(255,255,255))
            screen.blit(s,(WINDOW_WIDTH//2-s.get_width()//2,y))
            y+=s.get_height()+8
        pygame.draw.rect(screen,(229,154,163),start_button,border_radius=20)
        s=font.render("START",True,(255,255,255))
        screen.blit(s,s.get_rect(center=start_button.center))

    elif state=='playing':
        img = cv2.resize(frame,(224,224),interpolation=cv2.INTER_AREA)
        arr = ((np.float32(img).reshape(1,224,224,3)) / 127.5) - 1
        preds = model.predict(arr)
        idx = np.argmax(preds[0])
        label = class_names[idx]
        conf = preds[0][idx]
        print(f"Detected: {label} ({conf*100:.1f}%)")
        if current is None and emojis:
            fname = emojis.pop()
            current_label = fname.split('.')[0]
            current = emoji_surfaces[current_label]
            x_pos = random.randint(0,WINDOW_WIDTH-current.get_width())
            y_pos = -current.get_height()
        elif current is None and not emojis and pop_timer<=0:
            state='end'
            (win_sound if lives>0 else lose_sound).play()
        if pop_timer>0:
            pop_timer-=dt
            r=int((POP_DURATION-pop_timer)/POP_DURATION*40)
            pygame.draw.circle(screen,(255,255,0),pop_pos,r)
            if pop_timer<=0:
                current=None
        elif current:
            y_pos+=FALL_SPEED
            screen.blit(current,(x_pos,y_pos))
            if conf>=CONFIDENCE_THRESHOLD and label==current_label:
                score+=1
                pop_sound.play()
                pop_timer=POP_DURATION
                pop_pos=(x_pos+current.get_width()//2,y_pos+current.get_height()//2)
            elif y_pos>WINDOW_HEIGHT:
                lives-=1
                fail_sound.play()
                current=None
                if lives<=0:
                    state='end'
                    lose_sound.play()
        ix,iy=20,20
        screen.blit(lives_icon,(ix,iy))
        lt=font.render(f"X {lives}",True,(255,255,255))
        tw,th=lt.get_size()
        screen.blit(lt,(ix+lives_icon.get_width()+5,iy+(lives_icon.get_height()-th)//2))
        ss=score_font.render(str(score),True,(255,255,255))
        screen.blit(ss,(WINDOW_WIDTH//2-ss.get_width()//2,40))

    elif state=='end':
        text="You Win!" if lives>0 else "You Lose!"
        es=score_font.render(text,True,(255,255,255))
        screen.blit(es,(WINDOW_WIDTH//2-es.get_width()//2,40))
        img = win_img if lives>0 else lose_img
        x=WINDOW_WIDTH//2-img.get_width()//2
        y=40+es.get_height()+10
        screen.blit(img,(x,y))
        pygame.draw.rect(screen,(229,154,163),retry_button,border_radius=20)
        rs=font.render("Try Again",True,(255,255,255))
        screen.blit(rs,rs.get_rect(center=retry_button.center))

    pygame.display.flip()

camera.release()
pygame.quit()
