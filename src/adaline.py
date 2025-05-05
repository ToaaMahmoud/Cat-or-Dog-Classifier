import matplotlib.pyplot as plt
import cv2
import numpy as np
from matplotlib import pyplot as plt

# custom imports
from src.utilities import *


# variables

counter = 0
epoch = 0
P = []
T = []
b = []
cost = []
weights = np.array([])
bias = np.array([])
alfa = 0.02
image = None
resize_path = "./data/ALL/Resized"
image_num = int(count_files(resize_path) / 2)
statistics_size = (8, 5.5)


# error
def isError(e):
    if e != 0:
        return True
    return False


# flatten function
def flatten(image):
    new_image = []
    for row in image:
        for el in row:
            new_image.append(el)      
    return new_image


# trining function and calculate the cost
def training():
    global weights, T, P, b, cost, epoch
    S = 1
    for i in range(image_num):
        P.append(
            flatten(cv2.imread(f"{resize_path}/cat.{i}.jpg", cv2.IMREAD_GRAYSCALE))
        )  # p = [[cat], [dog], [cat], ...]
        T.append(
            [1000000000000 for _ in range(S)]
        )  # T = [[1000000000000], [-1000000000000], [1000000000000], [-1000000000000], ...]
        P.append(
            flatten(cv2.imread(f"{resize_path}/dog.{i}.jpg", cv2.IMREAD_GRAYSCALE))
        )
        T.append([-1000000000000 for _ in range(S)])

    P = np.array(P)  # array of lists.([cat], [dog], [cat], ...)
    T = np.array(T)  # array of lists.([1], [0], [1], [0], ...)
    
    
    numP = len(P)  # len of the array p = 20.
    R = len(P[0])  # 90 000

    # Initial w = 0, b = 0.
    weights = np.zeros([S, R], dtype=int)
    b = np.zeros([S, 1], dtype=int)

    count = 0
    index = 0
    error = 0

    while count < numP:
        p = np.array([P[index]]).transpose()
        n = np.dot(weights, p) + b  # Array =  ([Number])
        # print("----------->",n[0][0])
        if n[0][0] > 1000000000000:
            n[0][0] = 1000000000000
        if n[0][0] < -1000000000000:
            n[0][0] = -1000000000000

        e = T[index][0] - n[0][0]

        # Least Mean square error.
        error += (e) ** 2
        cost.append(int(np.log10(error)))
        # print(int(np.log10(error)))
        # print(error)
        # cost.append(error)

        if not (isError(e)):
            count += 1
        else:
            # print('error')
            # print(e)
            count = 0
            weights = weights + (alfa * np.dot(e, p.transpose()))
            b = b + np.dot(alfa, e)
            # print("Updating")
            # print(weights)
            # print(b)

        index += 1
        if index == numP:
            # print(int(np.log10(error)))
            index = 0
            epoch += 1
    # look
    draw(cost, (statistics_size[0], statistics_size[1]))
    return True


# true for cat and false for dog
def neural(img_path):
    global image, weights, b
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Error: Uploaded image not found or invalid.")
        return False

    # Resize the image to 300x300
    resized_img = cv2.resize(img, (300, 300), interpolation=cv2.INTER_AREA)
    p = np.array(flatten(resized_img))  # Flatten the resized image
    p = p.transpose()  # Shape: (90000, 1)

    # Perform the dot product with weights
    n = np.dot(weights, p) + b  # n = (1, 1)
    if n[0][0] >= 0:
        return True  # Cat
    return False  # Dog

def draw(cost, draw_size):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 4))
    # Set canvas size px
    fig.set_figwidth(draw_size[0])
    fig.set_figheight(draw_size[1])
    ax.plot(range(1, len(cost) + 1), cost, marker="o")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("log(Sum-squared-error)")
    ax.set_title("Adaline - Learning rate")
    return fig
