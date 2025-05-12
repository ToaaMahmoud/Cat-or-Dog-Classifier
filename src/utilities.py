import os
import cv2

# get directories

# cat_dir
cat_dir = "./data/ALL/cats"
cat_files = os.listdir(cat_dir)

# dog dir
dog_dir = "./data/ALL/dogs"
dog_files = os.listdir(dog_dir)


#resized folder
resized_dir = "./data/Resized"

#min_data path
min_TXT="./../data/Training_Data/min_data.txt"
#max_data path
max_data="./../data/Training_Data/max_data.csv"


# get number of images in folder
def count_files(folder_path):
    # List all files in the folder
    files = os.listdir(folder_path)
    # Count the number of files
    num_files = len(files)
    return num_files

#resize all images and rename them
def resize(cat_dir,dog_dir,destination, dim=(300, 300)):
    # convert imgs to jpg
    newCat=""
    newDog=""
    catNumber=0
    dogNumber=0
    for i in cat_files:
        newCat=cat_dir + "/" + i
        if i.split(".")[1] != "jpg":
            oldCat = cat_dir + "/" + i
            newCat = cat_dir + "/" + i.split(".")[0] + "." + "jpg"
            os.rename(oldCat, newCat)
        R_catImg = cv2.imread(f"{newCat}", cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(R_catImg, dim, interpolation=cv2.INTER_AREA)
        cv2.imwrite(f"{destination+'/'}cat.{catNumber}.jpg", resized)
        catNumber= catNumber+1

    # convert imgs to jpg
    for i in dog_files:
        newDog=dog_dir + "/" + i
        if i.split(".")[1] != "jpg":
            oldDog = dog_dir + "/" + i
            newDog = dog_dir + "/" + i.split(".")[0] + "." + "jpg"
            os.rename(oldDog, newDog)
        R_dogImg = cv2.imread(f"{newDog}", cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(R_dogImg, dim, interpolation=cv2.INTER_AREA)
        cv2.imwrite(f"{destination+'/'}dog.{dogNumber}.jpg", resized)
        dogNumber= dogNumber+1

    cv2.destroyAllWindows()



resize(cat_dir, dog_dir,"./data/Resized")


