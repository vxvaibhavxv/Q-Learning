import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import pickle
from matplotlib import style

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

style.use("ggplot")

RandomMove = ((0, 1), (1, 0), (0, -1), (-1, 0))  # If no input is supplied for the movement
Size = (11, 21)                                  # Grid size
HmEpisodes = 25000                               # No. of Episodes
ShowEvery = 1

# Rewards and Penalties
MovePenalty = 1
FoodReward = 100
MegaFoodReward = 200
FinalPositionReward = 500
EnemyPenalty = 20
WallPenalty = 10

MovesPerEpisode = 1000

Epsilon = 0.9
EpsDecay = 0.9998

StartQTable = None #"last.pickle"

Trained = False

LearningRate = 0.01
Discount = 0.95
PlayerN = 1
FoodN = 2
MegaFoodN = 3
EnemyN = 4
WallN = 5
FinalPositionN = 6
WallsCoordinates = [(0, 5), (1, 5), (2, 5), (3, 5), (5, 0), (5, 1), (5, 2), (5, 10), (6, 10), (7, 10), (8, 10), (10, 10), (0, 15), (1, 15), (2, 15), (3, 15)]

D = {1 : (255, 0, 0), 2 : (0, 255, 0), 3 : (100, 255, 100), 4 : (0, 0, 255), 5 : (55, 55, 55), 6 : (0, 165, 255)}
Frames = []

# Square block
class Blob:
    def __init__(self, x, y):
        self.x = x                      # np.random.randint(0, Size)
        self.y = y                      # np.random.randint(0, Size)

    def __str__(self):
        return f"{self.x}, {self.y}"

    def __sub__(self, other):
        return (self.x - other.x), (self.y - other.y)

    def action(self, choice):
        if choice == 0 :
            self.move(x = 1, y = 0)       # RIGHT Movement
        elif choice == 1 :
            self.move(x = 0, y = 1)       # DOWN Movement
        elif choice == 2 :
            self.move(x = -1, y = 0)      # LEFT Movement
        elif choice == 3 :
            self.move(x = 0, y = -1)      # UP Movement

    def move(self, x = None, y = None):
        # If no values are passed to x and y
        if x is None and y is None:
            a, b = np.random.choice(RandomMove)
            self.x += a
            self.y += b
        # If values are passed to x and y
        else:
            self.x += x
            self.y += y
        # Limitations for the walls
        if self.x < 0:
            self.x = 0
        elif self.x > Size[0] - 1:
            self.x = Size[0] - 1
        if self.y < 0 :
            self.y = 0
        elif self.y > Size[1] - 1 :
            self.y = Size[1] - 1

# If no file is used
if StartQTable is None:
    QTable = {}
    for x1 in range(-Size[0] + 1, Size[0]):
        for y1 in range(-Size[1] + 1, Size[1]) :
            for x2 in range(-Size[0] + 1, Size[0]) :
                for y2 in range(-Size[1] + 1, Size[1]) :
                    QTable[((x1, y1), (x2, y2))] = [np.random.uniform(-5, 0) for i in range(4)]
else:
    with open(StartQTable, "rb") as f:
        Trained = True
        Epsilon = 0.6
        EpsDecay = 0.9998
        QTable = pickle.load(f)

EpisodeRewards = []

for episode in range(HmEpisodes):
    # Objects of the class Blob
    Player = Blob(5, 20)
    Food = Blob(8, 6)
    MegaFood = Blob(9, 14)
    FinalPosition = Blob(2, 2)
    Enemy = Blob(6, 6)
    Walls = [Blob(i[0], i[1]) for i in WallsCoordinates]
    FoodTaken = [False, False]

    if episode % ShowEvery == 0:
        print(f"On #{episode}, Epsilon : {Epsilon}")
        Show = True
    else:
        Show = False

    EpisodeReward = 0
    moves = 0

    for i in range(MovesPerEpisode):
        moves = i
        Obs = (Player - Food, Player - Enemy)

        # Selection of appropriate action for the episode
        if np.random.rand() > Epsilon or Trained is True:
            Action = np.argmax(QTable[Obs])
        else:
            Action = np.random.randint(0, 4)

        Player.action(Action)

        # Penalty and Reward distribution
        if Player.x == Enemy.x and Player.y == Enemy.y:
            Reward = -EnemyPenalty
        elif Player.x == Food.x and Player.y == Food.y and not FoodTaken[0]:
            Reward = FoodReward
            FoodTaken[0] = True
        elif Player.x == MegaFood.x and Player.y == MegaFood.y and not FoodTaken[1]:
            Reward = MegaFoodReward
            FoodTaken[1] = True
        elif Player.x == FinalPosition.x and Player.y == FinalPosition.y:
            Reward = FinalPositionReward
        elif (Player.x, Player.y) in WallsCoordinates:
            Reward = -WallPenalty
        else:
            Reward = -MovePenalty

        if Trained is False:
            NewObs = (Player - Food, Player - Enemy)
            MaxFutureQ = np.max(QTable[NewObs])
            CurrentQ = QTable[Obs][Action]
            # Q Learning Algorithm
            NewQ = (1 - LearningRate) * CurrentQ + LearningRate * (Reward + Discount * MaxFutureQ)
            # Updating values in the Q-Table
            QTable[Obs][Action] = NewQ

        if Show:
            Env = np.zeros((Size[0], Size[1], 3), dtype=np.uint8)

            # Initialising colours to the objects
            if not FoodTaken[0]:
                Env[Food.x][Food.y] = D[FoodN]
            if not FoodTaken[1]:
                Env[MegaFood.x][MegaFood.y] = D[MegaFoodN]

            Env[Player.x][Player.y] = D[PlayerN]
            Env[Enemy.x][Enemy.y] = D[EnemyN]
            Env[FinalPosition.x][FinalPosition.y] = D[FinalPositionN]

            for l in range(len(WallsCoordinates)):
                Env[Walls[l].x][Walls[l].y] = D[WallN]

            Img = Image.fromarray(Env, "RGB")
            Img = np.array(Img)
            Img = image_resize(Img, height = 440) # Change the height of this function to adjust the size of the output screen
            Frames.append(Img)
            cv2.imshow("Fruit Enemy Game", Img)
            cv2.waitKey(1)

        EpisodeReward += Reward

        if Reward == WallPenalty or Reward == -EnemyPenalty or Reward == FinalPositionReward:
            break

    EpisodeRewards.append(EpisodeReward)
    Epsilon *= EpsDecay

    if episode % ShowEvery == 0 :
        print(f"Reward : {EpisodeReward} Moves: {moves}\n")

MovingAvg = np.convolve(EpisodeRewards, np.ones((ShowEvery, )) / ShowEvery, mode = "valid")
plt.plot([i for i in range(len(MovingAvg))], MovingAvg)
plt.ylabel(f"Rewards")
plt.xlabel("Episodes")
plt.show()

with open(f"last.pickle", "wb") as f:
    pickle.dump(QTable, f)

fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter("video.avi", fourcc, 30.0, (840, 440))

for i in range(len(Frames)):
    # writing to a image array
    out.write(Frames[i])

out.release()
