import os

import tqdm as tqdm
from sklearn.model_selection import train_test_split

if __name__ =="__main__":
    exit(0)
    path = r'C:\Users\skrzy\Downloads\statki_aircraft\MTARSI_Dataset'
    train_path = os.path.join(path, "train")
    val_path = os.path.join(path, "val")
    os.makedirs(val_path)

    for class_name in tqdm.tqdm(os.listdir(train_path)):
        val_class_path = os.path.join(val_path, class_name)
        os.makedirs(val_class_path)
        train_class_path = os.path.join(train_path, class_name)
        train_images = os.listdir(train_class_path)
        train, val = train_test_split(train_images, test_size=1/9, random_state=42)
        for image_name in val:
            os.rename(os.path.join(train_class_path, image_name), os.path.join(val_class_path, image_name))