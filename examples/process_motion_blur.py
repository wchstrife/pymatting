from pymatting import *
from tqdm import tqdm
import cv2
import numpy as np
import os
import traceback
from PIL import Image

def process_motion_blur_dataset(img_dir, alpha_dir, save_dir):
    '''
    处理adobe数据集，共有三个文件夹
    '''
    image_lists = os.listdir(img_dir)
    already_process_lists = os.listdir(save_dir)

    bar = tqdm(image_lists)

    for image_name in bar:
        if image_name in already_process_lists:
            continue
        image = load_image(os.path.join(img_dir, image_name), "RGB")
        alpha = load_image(os.path.join(alpha_dir, image_name), "GRAY")
        try:
            print(img_dir + "/" + image_name)
            foreground = estimate_foreground_ml(image, alpha, return_background=False)
            save_image(os.path.join(save_dir, image_name), foreground)
        except Exception as e:
            print(image_name + ' 出现错误')
            traceback.print_exc()
            pass         
        continue

if __name__ == "__main__":
    img_dir = '/home/sensetime/work/mmediting/data/GS_Video/motion_blur/img'
    alpha_dir = '/home/sensetime/work/mmediting/data/GS_Video/motion_blur/alpha'
    save_dir = '/home/sensetime/work/mmediting/data/GS_Video/motion_blur/fg'
    process_motion_blur_dataset(img_dir, alpha_dir, save_dir)