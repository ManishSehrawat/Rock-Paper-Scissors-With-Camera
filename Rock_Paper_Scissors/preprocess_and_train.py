import os
import cv2
from model import generate_model

TRAINING_DATA_DIR = "C:\\Users\\Manish Sehrawat\\Downloads\\training_data"

# Resizing the images
# Applying Gaussian blur pre-processing technique to reduce the noise in images
def preprocess(img):
    
    #Image Resizing
    width = 225
    height = 225
    dimensions = (width,height)
    img = cv2.resize(img,dimensions,interpolation = cv2.INTER_LINEAR)
    img = cv2.GaussianBlur(img,(5,5),0)
    return img

# NOTE: This method will only work if you have followed the same folder strucutre as I mentioned
# if not update this code below
def get_dataset():
    # load images from the training data directory
    dataset = []
    for label_dir in os.listdir(TRAINING_DATA_DIR):
        # iterating each item in the training data directory
        path = os.path.join(TRAINING_DATA_DIR, label_dir)
        if not os.path.isdir(path):
            continue
        # iterating threw each file in the sub directory
        for image_file in os.listdir(path):
            # loading each image
            img = cv2.imread(os.path.join(path, image_file))
            img = preprocess(img)           
            # adding them to the dataset
            dataset.append([img, label_dir])
    
    return zip(*dataset)

def main():
    X, y = get_dataset()
    generate_model(X, y)
    
if __name__ == "__main__":
    main()

