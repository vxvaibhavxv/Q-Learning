import pygame
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
import time

style.use("ggplot")
pygame.init()

Size = 210
HmEpisodes = 35000
MovePenalty = 1
EndReward = 2500
WallPenalty = 1000
Epsilon = 0.9
EpsDecay = 0.9998
ShowEvery = 1
StartQTable = None
LearningRate = 0.1
Discount = 0.95

win = pygame.display.set_mode((210, 210), pygame.NOFRAME)
Maze = pygame.image.load("data/maze.png")
Bob = pygame.image.load("data/bob.png")
Maze = pygame.transform.scale(Maze, (210, 210))

class MAZE:
    def __init__(self, x, y):
        self.img = Maze
        self.x = x
        self.y = y

    def collision(self, b):
        bob_mask = b.get_mask()
        maze_mask = pygame.mask.from_surface(self.img)
        offset = (int(self.x - b.x), int(self.y - b.y))
        point = bob_mask.overlap(maze_mask, offset)
        if point:
            return True
        return False

class BOB:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.img = Bob

    def get_mask(self):
        return pygame.mask.from_surface(self.img)

if StartQTable is None:
    QTable = {}
    for x1 in range(-Size * 2, Size * 2 + 1, 10):
        for y1 in range(-Size * 2, Size * 2 + 1, 10):
            QTable[(x1, y1)] = [np.random.uniform(-5, 0) for i in range(4)]
else:
    with open(StartQTable, "rb") as f:
        QTable = pickle.load(f)

EpisodeRewards = []

for episode in range(HmEpisodes):
    run = True
    bob = BOB(0, 0)
    maze = MAZE(0, 0)
    EpisodeReward = 0
    while run:
        Obs = (200 - bob.x, 200 - bob.y)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        if np.random.rand() > Epsilon :
            Action = np.argmax(QTable[Obs])
        else :
            Action = np.random.randint(0, 4)

        if Action == 0:
            bob.y -= 10
        if Action == 1:
            bob.y += 10
        if Action == 2:
            bob.x -= 10
        if Action == 3:
            bob.x += 10

        if bob.x < 0:
            bob.x = 0
        if bob.x > 200:
            bob.x = 200
        if bob.y < 0:
            bob.y = 200
        if bob.x > 200:
            bob.x = 200

        if maze.collision(bob):
            Reward = -WallPenalty
        elif bob.x == 200 and bob.y == 200:
            Reward = EndReward
        else:
            Reward = -MovePenalty

        NewObs = (200 - bob.x, 200 - bob.y)
        MaxFutureQ = np.max(QTable[NewObs])
        CurrentQ = QTable[Obs][Action]

        if Reward == EndReward:
            NewQ = EndReward
        elif Reward == -WallPenalty:
            NewQ = -WallPenalty
        else :
            NewQ = (1 - LearningRate) * CurrentQ + LearningRate * (Reward + Discount * MaxFutureQ)
        QTable[Obs][Action] = NewQ

        EpisodeReward += Reward
        if Reward == -WallPenalty or Reward == EndReward :
            break

    EpisodeRewards.append(EpisodeReward)
    Epsilon *= EpsDecay
    print(f"Episode : {episode + 1}   |   Reward : {EpisodeReward}")
    if episode % ShowEvery == 0:
        pygame.draw.rect(win, (200, 200, 200), (0, 0, 210, 210))
        win.blit(Maze, (maze.x, maze.y))
        win.blit(Bob, (bob.x, bob.y))
        pygame.display.update()

"""MovingAvg = np.convolve(EpisodeRewards, np.ones((ShowEvery, )) / ShowEvery, mode="valid")
plt.plot([i for i in range(len(MovingAvg))], MovingAvg)
plt.ylabel("Rewards")
plt.xlabel("Episodes")
plt.show()"""

x = [i for i in range(1, HmEpisodes + 1)]
y = EpisodeRewards
plt.plot(x, y)
plt.xlabel('Episodes')
plt.ylabel('Rewards')
plt.title('Maze')
plt.show()

with open(f"Maze Solution #1.pickle", "wb") as f:  # QTable - {int(time.time())}.pickle
    pickle.dump(QTable, f)

print(f"Hightest Reward : {max(EpisodeRewards)}")
