import torch
import cv2
import pandas as pd
import numpy as np
import config
import utils
from torch.utils.data import Dataset, DataLoader

def train_test_split(images, labels, split):
    len_data = len(images)
    # calculate the validation data sample length
    valid_split = int(len_data * split)
    # calculate the training data samples length
    train_split = int(len_data - valid_split)
    image_training_samples = images[:train_split][:]
    label_training_samples = labels[:train_split][:]
    image_valid_samples = images[-valid_split:][:]
    image_valid_samples = labels[-valid_split:][:]
    return image_training_samples, label_training_samples, image_valid_samples, image_valid_samples

class FaceKeypointDataset(Dataset):
    def __init__(self, image_samples,label_samples):
        self.data = label_samples
        self.image_samples = image_samples
        self.resize = 90

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = np.asarray(self.image_samples[index])
        orig_h, orig_w, channel = image.shape
        keypoints = self.data[::2][index]
        keypoints = np.array(keypoints, dtype='float32')
        # reshape the keypoints
        keypoints = keypoints.reshape(-1, 2)
        # rescale keypoints according to image resize
        #keypoints = keypoints * [self.resize / orig_w, self.resize / orig_h]
        return {
            'image': torch.tensor(image, dtype=torch.float),
            'keypoints': torch.tensor(keypoints, dtype=torch.float),
        }