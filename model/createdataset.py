import numpy as np
import matplotlib.pyplot as plt 
import os
import cv2 
import random

training_sets = ["pepsi2coke/trainA", "pepsi2coke/trainB", "pepsi2coke/testA", "pepsi2coke/testB", "pepsi2coke/00000"]
training_data_trainA = []
training_data_trainB = []
training_data_testA = []
training_data_testB = []
stylegan_data = []
n_W, n_H = 128, 128
def create_training_data():
    for i in training_sets:
        if i == "pepsi2coke/00000":
            path = i
            for img in os.listdir(path):
                try:
                    img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR)
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                    new_array = cv2.resize(img_array, (n_H, n_W))
                    stylegan_data.append([new_array])
                except Exception as e:
                    print("There is an issue with: " + img + " in trainA")
        """
        elif i == "pepsi2coke/trainB":
            path = i
            for img in os.listdir(path):
                try:
                    img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR)
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                    new_array = cv2.resize(img_array, (n_H, n_W))
                    training_data_trainB.append([new_array])
                except Exception as e:
                    print("There is an issue with: " + img + " in trainB")
        elif i == "pepsi2coke/testA":
            path = i
            for img in os.listdir(path):
                try:
                    img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR)
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                    new_array = cv2.resize(img_array, (n_H, n_W))
                    training_data_testA.append([new_array])
                except Exception as e:
                    print("There is an issue with: " + img + " in testA")
        elif i == "pepsi2coke/testB":
            path = i
            for img in os.listdir(path):
                try:
                    img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR)
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                    new_array = cv2.resize(img_array, (n_H, n_W))
                    training_data_testB.append([new_array])
                except Exception as e:
                    print("There is an issue with: " + img + " in testB")
        """
    """
    random.shuffle(training_data_trainA)
    random.shuffle(training_data_trainB)
    random.shuffle(training_data_testA)
    random.shuffle(training_data_testB)
    """
    random.shuffle(stylegan_data)

create_training_data()
"""
label_coke = "coke"
label_pepsi = "pepsi"

train_coke = []
train_coke_label = []

train_pepsi = []
train_pepsi_label = []

test_coke = []
test_coke_label = []

test_pepsi = []
test_pepsi_label = []
"""
stylegan_dataset = []
"""
for features in training_data_trainA:
    train_coke.append(features)
    train_coke_label.append(label_coke)

for features in training_data_trainB:
    train_pepsi.append(features)
    train_pepsi_label.append(label_pepsi)

for features in training_data_testA:
    test_coke.append(features)
    test_coke_label.append(label_coke)

for features in training_data_testB:
    test_pepsi.append(features)
    test_pepsi_label.append(label_pepsi)
"""
for features in stylegan_data:
    stylegan_dataset.append(features)
"""
train_coke = np.array(train_coke).reshape(-1, n_W, n_H, 3)
train_pepsi = np.array(train_pepsi).reshape(-1, n_W, n_H, 3)
test_coke = np.array(test_coke).reshape(-1, n_W, n_H, 3)
test_pepsi = np.array(test_pepsi).reshape(-1, n_W, n_H, 3)
"""
stylegan_dataset = np.array(stylegan_dataset).reshape(-1, 128, 128, 3)
"""
print("trainA shape: " + str(train_coke.shape))
print("trainB shape: " + str(train_pepsi.shape))
print("testA shape: " + str(test_coke.shape))
print("testB shape: " + str(test_pepsi.shape))

print(train_coke[0])
print("label: " + train_coke_label[0])

save_trainA = np.save('train_coke', train_coke)
save_trainB = np.save('train_pepsi', train_pepsi)
save_testA = np.save('test_coke', test_coke)
save_testB = np.save('test_pepsi', test_pepsi)
"""
save_stylegan = np.save('faces', stylegan_dataset)
"""
save_trainA_label = np.save('train_coke_label', train_coke_label)
save_trainB_label = np.save('train_pepsi_label', train_pepsi_label)
save_testA_label = np.save('test_coke_label', test_coke_label)
save_testB_label = np.save('test_pepsi_label', test_pepsi_label)
"""
