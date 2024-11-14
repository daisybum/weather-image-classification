import os
import re
import shutil
from glob import glob

import pandas as pd
from PIL import Image
from tqdm import tqdm


def csv_to_dataframe(path_lst):
    df = pd.read_csv(path_lst[0], encoding='utf-8-sig')
    for label_file in tqdm(path_lst[1:]):
        try:
            df_ = pd.read_csv(label_file, encoding='utf-8-sig')
        except UnicodeDecodeError:
            df_ = pd.read_csv(label_file, encoding='cp949', delimiter="\t")
            print(label_file)
        df = pd.concat([df, df_], ignore_index=True)

    return df


def count_class_instances(dataset):
    """Count the number of instances of each class in the dataset."""
    count_dict = {}
    for _, label in tqdm(dataset):
        if label in count_dict:
            count_dict[label] += 1
        else:
            count_dict[label] = 1
    return count_dict


def check_image_compatibility(img_abs_path):
    img_paths = glob(img_abs_path)

    for img_path in tqdm(img_paths):
        try:
            with open(img_path, "rb") as f:
                img = Image.open(f)
                img.convert("RGB")
        except OSError:
            filename = os.path.basename(img_path)
            dst_path = os.path.join("D:\\allbigdat\\allbig_images_예외", filename)
            shutil.move(img_path, dst_path)
            print(filename)


def check_sensor_list(data_dir):
    img_paths = glob(os.path.join(data_dir, "*\\*\\*.jpg"))

    sensor_lst = []
    for path in tqdm(img_paths):
        filename = os.path.basename(path)
        sensor = re.split(r"_0_", filename)[0]
        sensor_lst.append(sensor)

    return set(sensor_lst)


if __name__ == "__main__":
    check_sensor_list("D:\\allbigdat\\allbig_images")
    check_image_compatibility("D:\\allbigdat\\allbig_images\\test\\*\\*.jpg")
