import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np
import random
from copy import deepcopy

from hog import hog


# Plan

# import data
# verify data integrity
# split data train/test  DONE
# data pre-processing:
#    * image normalisation
#    * image segmentation
#    * hand extraction
# feature extraction
# classification

# confusion matrix = pt colocviu

def detectie_culoare_piele(img):
    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]

    C1 = np.zeros((img.shape[0], img.shape[1]))
    C2 = np.zeros((img.shape[0], img.shape[1]))
    C3 = np.zeros((img.shape[0], img.shape[1]))
    C4 = np.zeros((img.shape[0], img.shape[1]))
    C5 = np.zeros((img.shape[0], img.shape[1]))
    C6 = np.zeros((img.shape[0], img.shape[1]))
    C7 = np.zeros((img.shape[0], img.shape[1]))
    rez = np.zeros((img.shape[0], img.shape[1]))

    C1[np.logical_and(R > 95, G > 40, B > 20)] = 1
    C2[(np.maximum(np.maximum(R, G), B) - np.minimum(np.minimum(R, G), B)) > 15] = 1
    C3[np.absolute(R - G) > 15] = 1
    C4[np.logical_and(R > G, R > B)] = 1

    C5[np.logical_and(R > 220, G > 210, B > 170)] = 1
    C6[np.absolute(R - G) <= 15] = 1
    C7[np.logical_and(R > B, G > B)] = 1

    caz1 = np.logical_and(np.logical_and(C1 == 1, C2 == 1, C3 == 1), C4 == 1)
    caz2 = np.logical_and(C5 == 1, C6 == 1, C7 == 1)

    rez[np.logical_or(caz1 == 1, caz2 == 1)] = 1

    return rez


base_path = "Images/"
file_paths = []
for i in range(4):
    file_paths.append(glob.glob(base_path+str(i)+"/*.jpg"))

print(f"All Images: {file_paths}")
print(f"First Folder: {file_paths[0]}")
print(f"First Image: {file_paths[0][0]}")

random.seed(42)
image_index_arr = random.sample(range(0,9),7)

print(image_index_arr)

train_images_paths = []
test_images_paths = []

for hand_class in file_paths:
    tmp_train = []
    tmp_test = []

    for i in range(10):
        if i in image_index_arr:
            tmp_train.append(hand_class[i])
        else:
            tmp_test.append(hand_class[i])

    train_images_paths.append(tmp_train)
    test_images_paths.append(tmp_test)

print(np.shape(train_images_paths))
print(np.shape(test_images_paths))

train_images = []
test_images = []

for cls in train_images_paths:
    t = []
    for path in cls:
        t.append(cv2.imread(path))
    train_images.append(t)

for cls in test_images_paths:
    t = []
    for path in cls:
        t.append(cv2.imread(path))
    test_images.append(t)

print(np.shape(train_images))
print(np.shape(test_images))

kernel = np.ones((5,5), np.uint8)

copy_train_images = deepcopy(train_images)
cropped_train_images = deepcopy(train_images)
copy_test_images = deepcopy(test_images)
cropped_test_images = deepcopy(test_images)

for i, cls in enumerate(train_images):
    for j, image in enumerate(cls):
        train_images[i][j] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        width, height, channels = image.shape
        train_images[i][j] = cv2.resize(train_images[i][j], (height//10, width//10))
        copy_train_images[i][j] = cv2.resize(copy_train_images[i][j], (height//10, width//10))

        train_images[i][j] = detectie_culoare_piele(train_images[i][j])
        train_images[i][j] = cv2.dilate(train_images[i][j], kernel)
        train_images[i][j] = cv2.erode(train_images[i][j], kernel)


for i, cls in enumerate(test_images):
    for j, image in enumerate(cls):
        test_images[i][j] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        width, height, channels = image.shape
        test_images[i][j] = cv2.resize(test_images[i][j], (height//10, width//10))
        copy_test_images[i][j] = cv2.resize(copy_test_images[i][j], (height//10, width//10))

        test_images[i][j] = detectie_culoare_piele(test_images[i][j])
        test_images[i][j] = cv2.dilate(test_images[i][j], kernel)
        test_images[i][j] = cv2.erode(test_images[i][j], kernel)


for i, cls in enumerate(train_images):
    for j, image in enumerate(cls):
        contours, _ = cv2.findContours(train_images[i][j].astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        area_max, bb_max = 0, 0
        for contour in contours:
            bounding_box = cv2.boundingRect(contour)  # x y width height
            area = bounding_box[2] * bounding_box[3]

            if area_max < area:
                area_max = area
                bb_max = bounding_box

        cropped_train_images[i][j] = copy_train_images[i][j][bb_max[1]:bb_max[1] + bb_max[3], bb_max[0]:bb_max[0] + bb_max[2]]

for i, cls in enumerate(test_images):
    for j, image in enumerate(cls):
        contours, _ = cv2.findContours(test_images[i][j].astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        area_max, bb_max = 0, 0
        for contour in contours:
            bounding_box = cv2.boundingRect(contour)  # x y width height
            area = bounding_box[2] * bounding_box[3]

            if area_max < area:
                area_max = area
                bb_max = bounding_box

        cropped_test_images[i][j] = copy_test_images[i][j][bb_max[1]:bb_max[1] + bb_max[3], bb_max[0]:bb_max[0] + bb_max[2]]

mask = test_images[0][0]
img = copy_test_images[0][0]

img[mask == 0] = 0

plt.figure(), plt.imshow(cropped_train_images[0][0])

plt.show()

first_hog = hog.compute(cropped_train_images[0][0])
print(np.size(first_hog))

