import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
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
resize_path = "./data/Resized"

# Create resize folder if it doesn't exist
if not os.path.exists(resize_path):
    os.makedirs(resize_path)
    print(f"Created directory: {resize_path}")

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


# training function and calculate the cost
def training():
    global weights, T, P, b, cost, epoch
    S = 1
    
    # Reset P and T to avoid appending to existing data on retraining
    P.clear()
    T.clear()
    cost.clear()
    
    # Re-check image_num in case the folder was just created
    current_image_num = int(count_files(resize_path) / 2)
    if current_image_num == 0:
        print(f"Warning: No images found in {resize_path}. Please add images before training.")
        return False
        
    for i in range(current_image_num):
        # Load and normalize cat images
        cat_img = cv2.imread(f"{resize_path}/cat.{i}.jpg", cv2.IMREAD_GRAYSCALE)
        if cat_img is not None:
            cat_data = flatten(cat_img)
            # Normalize pixel values to [0,1] range
            cat_data = [pixel / 255.0 for pixel in cat_data]
            P.append(cat_data)
            T.append([1.0 for _ in range(S)])  # Use 1.0 for cats, simpler than large numbers
        
        # Load and normalize dog images
        dog_img = cv2.imread(f"{resize_path}/dog.{i}.jpg", cv2.IMREAD_GRAYSCALE)
        if dog_img is not None:
            dog_data = flatten(dog_img)
            # Normalize pixel values to [0,1] range
            dog_data = [pixel / 255.0 for pixel in dog_data]
            P.append(dog_data)
            T.append([-1.0 for _ in range(S)])  # Use -1.0 for dogs, simpler than large numbers

    P = np.array(P)  # array of lists.([cat], [dog], [cat], ...)
    T = np.array(T)  # array of lists.([1], [0], [1], [0], ...)

    numP = len(P)  # len of the array p = 20.
    R = len(P[0])  # 90 000    # Initial w = 0, b = 0.
    weights = np.zeros([S, R], dtype=float)  # Changed from int to float for better precision
    b = np.zeros([S, 1], dtype=float)  # Changed from int to float for better precision

    count = 0
    index = 0
    error = 0   
    while count < numP:
        p = np.array([P[index]]).transpose()
        n = np.dot(weights, p) + b  # Array =  ([Number])
        
        # Clamp values within reasonable range (-1 to 1)
        if n[0][0] > 1.0:
            n[0][0] = 1.0
        if n[0][0] < -1.0:
            n[0][0] = -1.0

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
            epoch += 1    # Print final results
    print(f"Training completed in {epoch} epochs")
    print(f"Final error: {error}")
    
    # Plot learning curve
    draw(cost, (statistics_size[0], statistics_size[1]))
    return True


# true for cat and false for dog
def neural(img_path):
    global image, weights, b
    
    # Check if resize folder exists
    if not os.path.exists(resize_path):
        print(f"Error: Resize directory {resize_path} does not exist")
        return None
    
    # Check if file exists and can be loaded
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not load image from {img_path}")
        return None
    
    # Resize the image to match the training data dimensions
    # Get size from first training image
    if not os.path.exists(f"{resize_path}/cat.0.jpg"):
        print(f"Error: No training images found in {resize_path}")
        return None
        
    train_img = cv2.imread(f"{resize_path}/cat.0.jpg", cv2.IMREAD_GRAYSCALE)
    if train_img is not None:
        img = cv2.resize(img, train_img.shape[:2][::-1])
    
    # Process the image
    p = np.array(flatten(img))
    
    # Ensure p is the right shape for transpose operation
    if len(p) == 0:
        print("Error: Image contains no data after flattening")
        return None
    
    # Normalize pixel values (if not already done in training)
    p = p / 255.0
        
    p = p.transpose()
    
    # Make prediction
    try:
        n = np.dot(weights, p) + b
        
        # Debug information
        print(f"Prediction value: {n[0][0]}")
        result = n[0][0] >= 0
        animal_type = "Cat" if result else "Dog"
        print(f"Predicted as: {animal_type}")
        
        return result
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None


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