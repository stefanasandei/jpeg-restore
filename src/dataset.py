import os
import matplotlib.pyplot as plt
from PIL import Image

df2k_path = "/mnt/sda3/Documents/Datasets/DF2K/DF2K_train_HR"

if __name__ == "__main__":
    img = os.listdir(df2k_path)[3]
    img = Image.open(f"{df2k_path}/{img}")

    plt.imshow(img)
    plt.savefig("a.png")
