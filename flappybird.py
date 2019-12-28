import pygame
import os
import random
import numpy as np
import pickle
from matplotlib import style
import time

pygame.font.init()
pygame.init()

style.use("ggplot")

WIN_WIDTH = 288
WIN_HEIGHT = 512
HmEpisodes = 25000
HitPenalty = -1000
PipeReward = 10
kill = False
PipeInitialReward = 5
Epsilon = 0.9
EpsDecay = 0.9998
ShowEvery = 1
StartQTable = True
LearningRate = 0.1
Discount = 0.95
Move = 0.01

BIRD_IMGS = [pygame.image.load(os.path.join("data", "bird1.png")),
             pygame.image.load(os.path.join("data", "bird2.png")),
             pygame.image.load(os.path.join("data", "bird3.png"))]
PIPE_IMG = pygame.image.load(os.path.join("data", "pipe.png"))
BASE_IMG = pygame.image.load(os.path.join("data", "base.png"))
BG_IMG = pygame.image.load(os.path.join("data", "bg.png"))
GM_OVER = pygame.image.load(os.path.join("data", "gameover.png"))
SOUNDS = [pygame.mixer.Sound("data/hit.ogg"),
          pygame.mixer.Sound("data/point.ogg"),
          pygame.mixer.Sound("data/wing.ogg")]

STAT_FONT = pygame.font.SysFont("comicsans", 30)
PIPE_DIST = 150

class Bird:
    IMGS = BIRD_IMGS
    MAX_ROTATION = 25
    ROT_VEL = 12
    ANIMATION_TIME = 5

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.tilt = 0
        self.tick_count = 0
        self.vel = 0
        self.height = self.y
        self.img_count = 0
        self.img = self.IMGS[0]

    def jump(self):
        self.vel = - 1.8
        self.tick_count = 0
        self.height = self.y

    def move(self):
        self.tick_count += 1

        d = self.vel * self.tick_count + 0.3 * self.tick_count ** 1.8

        if d >= 16:
            d = 16
        if d < 0:
            d -= 2

        self.y += d

        if d < 0 or self.y < self.height + 50:
            if self.tilt < self.MAX_ROTATION:
                self.tilt = self.MAX_ROTATION
        else:
            if self.tilt > - 90:
                self.tilt -= self.ROT_VEL

    def draw(self, win):
        self.img_count += 1

        if self.img_count < self.ANIMATION_TIME:
            self.img = self.IMGS[0]
        elif self.img_count < self.ANIMATION_TIME * 2:
            self.img = self.IMGS[1]
        elif self.img_count < self.ANIMATION_TIME * 3:
            self.img = self.IMGS[2]
        elif self.img_count < self.ANIMATION_TIME * 4:
            self.img = self.IMGS[1]
        elif self.img_count < self.ANIMATION_TIME * 4 + 1:
            self.img = self.IMGS[0]
            self.img_count = 0

        if self.tilt <= -80:
            self.img = self.IMGS[1]
            self.img_count = self.ANIMATION_TIME * 2

        rotated_image = pygame.transform.rotate(self.img, self.tilt)
        new_rect = rotated_image.get_rect(center = self.img.get_rect(topleft = (self.x, self. y)).center)
        win.blit(rotated_image, new_rect.topleft)

    def get_mask(self):
        return pygame.mask.from_surface(self.img)

class Pipe:
    GAP = 100
    VEL = 3.5

    def __init__(self, x):
        self.x = x
        self.height = 0
        self.top = 0
        self.bottom = 0
        self.PIPE_TOP = pygame.transform.flip(PIPE_IMG, False, True)
        self.PIPE_BOTTOM = PIPE_IMG
        self.passed = False
        self.set_height()

    def set_height(self):
        self.height = random.randrange(50, 251)
        self.top = self.height - self.PIPE_TOP.get_height()
        self.bottom = self.height + self.GAP

    def move(self):
        self.x -= self.VEL

    def draw(self, win):
        win.blit(self.PIPE_TOP, (self.x, self.top))
        win.blit(self.PIPE_BOTTOM, (self.x, self.bottom))

    def collide(self, bird):
        bird_mask = bird.get_mask()
        top_mask = pygame.mask.from_surface(self.PIPE_TOP)
        bottom_mask = pygame.mask.from_surface(self.PIPE_BOTTOM)

        top_offset = (int(self.x - bird.x), int(self.top - bird.y))
        bottom_offset = (int(self.x - bird.x), int(self.bottom - bird.y))

        b_point = bird_mask.overlap(bottom_mask, bottom_offset)
        t_point = bird_mask.overlap(top_mask, top_offset)

        if t_point or b_point:
            return True

        return False

class Base:
    VEL = 3.5
    WIDTH = BASE_IMG.get_width()
    IMG = BASE_IMG

    def __init__(self, y):
        self.y = y
        self.x1 = 0
        self.x2 = self.WIDTH

    def move(self):
        self.x1 -= self.VEL
        self.x2 -= self.VEL

        if self.x1 + self.WIDTH < 0:
            self.x1 = self.x2 + self.WIDTH

        if self.x2 + self.WIDTH < 0:
            self.x2 = self.x1 + self.WIDTH

    def draw(self, win):
        win.blit(self.IMG, (self.x1, self.y))
        win.blit(self.IMG, (self.x2, self.y))

def draw_window(win, bird, pipes, base, score):
    win.blit(BG_IMG, (0, 0))

    for pipe in pipes:
        pipe.draw(win)

    text = STAT_FONT.render("SCORE : " + str(score), 1, (255, 255, 255))
    win.blit(text, (WIN_WIDTH - 20 - text.get_width(), 20))
    base.draw(win)
    bird.draw(win)

    pygame.display.update()

if StartQTable:
    QTable = {}
    for x1 in range(0, 361, 2):
        for y1 in range(0, 361, 2):
            for y in range(0, 420, 2):
                QTable[(x1, y1, y)] = [np.random.uniform(-10, 0) for i in range(2)]
    print()
else:
    f = open(StartQTable, "rb")
    QTable = pickle.load(f)

EpisodeRewards = []

for episode in range(HmEpisodes):
    bird = Bird(80, 250)
    base = Base(400)
    pipes = [Pipe(200), Pipe(200 + PIPE_DIST)]
    win = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT), pygame.NOFRAME)
    clock = pygame.time.Clock()
    score = 0
    run = True
    EpisodeReward = 0
    Reward = 0
    Primary = pipes[0]
    while run:
        clock.tick(30)
        Reward = Move
        DIST = abs(int(bird.y))
        if DIST % 2 == 1:
            DIST -= 1
        TOPDISTANCE = abs(int(Primary.height - bird.y))
        if TOPDISTANCE % 2 == 1:
            TOPDISTANCE -= 1
        BOTTOMDISTANCE = abs(int(Primary.bottom - bird.y))
        if BOTTOMDISTANCE % 2 == 1:
            BOTTOMDISTANCE -= 1
        Obs = (TOPDISTANCE, BOTTOMDISTANCE, DIST)

        if np.random.rand() > Epsilon :
            Action = np.argmax(QTable[Obs])
        else :
            Action = np.random.randint(0, 2)
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                run = False
                kill = True
        if Action == 1:
            if episode % ShowEvery == 0 :
                SOUNDS[2].play()
            bird.jump()

        add_pipe = False
        rem = []

        for r in rem:
            pipes.remove(r)

        if Primary.x < bird.x < Primary.x + Primary.PIPE_TOP.get_width() :
            Reward = PipeInitialReward

        for pipe in pipes:
            if pipe.collide(bird):
                Reward = HitPenalty
                if episode % ShowEvery == 0 :
                    SOUNDS[0].play()
                    win.blit(GM_OVER, (int(WIN_WIDTH - GM_OVER.get_width()) / 2, 200))
                    pygame.display.update()
                    pygame.time.delay(500)

            if not pipe.passed and pipe.x + pipe.PIPE_TOP.get_width() < 0:
                rem.append(pipe)
                add_pipe = True
                pipe.passed = True
                if len(rem) == 1:
                    Primary = pipes[1]
                if len(rem) == 2:
                    Primary = pipes[0]
                if episode % ShowEvery == 0 :
                    SOUNDS[1].play()
                score += 1
                Reward = PipeReward
            pipe.move()

        if add_pipe:
            pipes.append(Pipe(pipe.x + PIPE_DIST))

        if bird.y + bird.img.get_height() >= 400:
            bird.y = 400
            Reward = HitPenalty
            if episode % ShowEvery == 0 :
                SOUNDS[0].play()
                win.blit(GM_OVER, (int(WIN_WIDTH - GM_OVER.get_width()) / 2 , 200))
                pygame.display.update()
                pygame.time.delay(500)

        DIST = abs(int(bird.y))
        if DIST % 2 == 1 :
            DIST -= 1
        TOPDISTANCE = abs(int(Primary.height - bird.y))
        if TOPDISTANCE % 2 == 1 :
            TOPDISTANCE -= 1
        BOTTOMDISTANCE = abs(int(Primary.bottom - bird.y))
        if BOTTOMDISTANCE % 2 == 1 :
            BOTTOMDISTANCE -= 1

        NewObs = (TOPDISTANCE, BOTTOMDISTANCE, DIST)
        MaxFutureQ = np.max(QTable[NewObs])
        CurrentQ = QTable[Obs][Action]

        if Reward == PipeReward:
            NewQ = PipeReward
        elif Reward == HitPenalty:
            NewQ = HitPenalty
        elif Reward == PipeInitialReward:
            NewQ = PipeInitialReward
        elif Reward == Move:
            NewQ = Move
        else:
            NewQ = (1 - LearningRate) * CurrentQ + LearningRate * (Reward + Discount * MaxFutureQ)
        QTable[Obs][Action] = NewQ

        EpisodeReward += Reward

        base.move()
        bird.move()
        if episode % ShowEvery == 0 :
            draw_window(win, bird, pipes, base, score)

        if Reward == HitPenalty:
            break

        if score == 200:
            kill = True
            break

    EpisodeRewards.append(EpisodeReward)
    if episode % ShowEvery == 0:
        print(f"Episode #{episode}   ---   Reward : {EpisodeReward}   ---   Score : {score}")

    Epsilon *= EpsDecay
    if kill:
        break

f = open(f"QTable - {int(time.time())}.pickle", "wb")
pickle.dump(QTable, f)

f.close()
pygame.quit()
quit()