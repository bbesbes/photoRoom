import pandas as pd
import numpy as np
import cv2
from dataLoader import loadData
from dataset import train_test_split, FaceKeypointDataset
from torch.utils.data import Dataset, DataLoader
import config
from utils import dataset_keypoints_plot
import torch.optim as optim
import torch.nn as nn
from model import FaceKeypointResNet50
from train import fit,validate

trainFilename = '../input/training.csv'
split = 0.1

def printPoints(img,marks):
    for img, marks in zip(images,labels):
        x_points = labels[:: 2]
        y_points = labels[1::2]
        print(img.shape)
        #print(x_points)
        #print(y_points)

if __name__ == "__main__":
    ### Dataloader for images and labels
    df_train = pd.read_csv(trainFilename)
    target_cols = list(df_train.drop('Image', axis=1).columns)
    images, labels = loadData(df_train,target_cols)

    ### Split Data Train and test
    image_training_samples, label_training_samples, image_valid_samples, label_valid_samples = train_test_split(images, labels, split)

    #printPoints(image_valid_samples, image_valid_samples)
    #print(len(image_valid_samples))

    # initialize the dataset - `FaceKeypointDataset()`
    train_data = FaceKeypointDataset(image_training_samples,label_training_samples)
    valid_data = FaceKeypointDataset(image_valid_samples,label_valid_samples)
    # prepare data loaders
    train_loader = DataLoader(train_data,
                              batch_size=config.BATCH_SIZE,
                              shuffle=True)
    valid_loader = DataLoader(valid_data,
                              batch_size=config.BATCH_SIZE,
                              shuffle=False)
    print(f"Training sample instances: {len(image_training_samples)}")
    print(f"Validation sample instances: {len(image_valid_samples)}")

    # whether to show dataset keypoint plots
    if config.SHOW_DATASET_PLOT:
        dataset_keypoints_plot(train_data)

    # model
    model = FaceKeypointResNet50(pretrained=True, requires_grad=True).to(config.DEVICE)
    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.LR)
    # we need a loss function which is good for regression like SmmothL1Loss ...
    # ... or MSELoss
    criterion = nn.SmoothL1Loss()

    train_loss = []
    val_loss = []
    for epoch in range(config.EPOCHS):
        print(f"Epoch {epoch + 1} of {config.EPOCHS}")
        train_epoch_loss = fit(model, train_loader, train_data)
        val_epoch_loss = validate(model, valid_loader, valid_data, epoch)
        train_loss.append(train_epoch_loss)
        val_loss.append(val_epoch_loss)
        print(f"Train Loss: {train_epoch_loss:.4f}")
        print(f'Val Loss: {val_epoch_loss:.4f}')