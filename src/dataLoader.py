import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Image characteristics
IMG_WIDTH = 96
IMG_HEIGHT = 96
IMG_CHANNELS = 1

def loadData(df_train,target_cols):
    #print(df_train.shape)
    feature_col = 'Image'
    # Fill missing values
    df_train[target_cols] = df_train[target_cols].fillna(df_train[target_cols].mean())

    list = df_train[feature_col].str.split().tolist()
    df = pd.DataFrame(list)
    #print(f"DataFrame:\n{df}\n")
    #print(f"column types:\n{df.dtypes}")

    raw_images = np.array(df_train[feature_col].str.split().tolist(), dtype='float')
    images = raw_images.reshape(-1, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

    labels = df_train[target_cols].values

    return images,labels

def show_examples(images, landmarks):
    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(16, 16))

    for img, marks, ax in zip(images, landmarks, axes.ravel()):
        # Keypoints
        x_points = marks[:: 2]
        y_points = marks[1::2]
        print(x_points)
        #ax.imshow(img.squeeze(), cmap='gray')
        #ax.scatter(x_points, y_points, s=10, color='red')

    plt.show()


#idx = np.random.choice(16, 16)
#show_examples(images[idx], labels[idx])

