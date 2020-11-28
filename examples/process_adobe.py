from pymatting import *
from tqdm import tqdm
import cv2
import numpy as np
import os
import traceback
from PIL import Image

def generate_show_fg():
    '''
    对比展示两种不同
    '''
    image = load_image("data/test/fg/caterpillar-205204_1920.jpg", "RGB")
    alpha = load_image("data/test/alpha/caterpillar-205204_1920.jpg", "GRAY")
    bg_new = load_image("data/test/bg/organ_bg.png", size=[alpha.shape[1], alpha.shape[0]] , mode="RGB") 

    # Closed-Form
    foreground = estimate_foreground_cf(image, alpha, return_background=False)
    new_image = blend(foreground, bg_new, alpha)
    images = [image, alpha, foreground, new_image]
    grid = make_grid(images)
    save_image("results/caterpillar-205204_1920_closed.png", grid)

    # Fast
    foreground = estimate_foreground_ml(image, alpha, return_background=False)
    new_image = blend(foreground, bg_new, alpha)
    images = [image, alpha, foreground, new_image]
    grid = make_grid(images)
    save_image("results/caterpillar-205204_1920_fast.png", grid)

def generate_show_raw_image():
    '''
    直接根据img合成图片,跟重新估计前景的对比
    '''
    image = load_image("data/test/fg/caterpillar-205204_1920.jpg", "RGB")
    alpha = load_image("data/test/alpha/caterpillar-205204_1920.jpg", "GRAY")
    image = cv2.resize(image, (320,320))
    alpha = cv2.resize(alpha, (320,320))
    bg_new = load_image("data/test/bg/organ_bg.png", size=[alpha.shape[1], alpha.shape[0]] , mode="RGB") 

    foreground = estimate_foreground_ml(image, alpha, return_background=False)
    new_image = blend(foreground, bg_new, alpha)
    old_image = blend(image, bg_new, alpha)
    images = [image, alpha, old_image, new_image]
    grid = make_grid(images)
    save_image("results/caterpillar-205204_1920_raw.png", grid)

def generate_bg():
    '''
    生成纯色背景，注意CV2里面的顺序是BGR
    '''
    bg = np.empty([256,256,3], dtype=int)
    bg[:, :, 0] = 0
    bg[:, :, 1] = 140
    bg[:, :, 2] = 255

    cv2.imwrite('data/test/bg/organ_bg.png', bg)

def process_adobe_dataset(img_dir, alpha_dir, save_dir):
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
            with open(os.path.join(save_dir, 'log.txt'), 'a') as f:
                f.write(img_dir  + '/' + image_name + '/n')
            traceback.print_exc()
            pass         
        continue


if __name__ == "__main__":
    # print('------------process adobe test-----------------')
    # img_dir = 'data/Combined_Dataset/Test_set/Adobe-licensed images/fg'
    # alpha_dir = 'data/Combined_Dataset/Test_set/Adobe-licensed images/alpha'
    # save_dir = 'results/adobe_test'
    # process_adobe_dataset(img_dir, alpha_dir, save_dir)

    # print('------------process adobe train Adobe-licensed Image-----------------')
    # img_dir = 'data/Combined_Dataset/Training_set/Adobe-licensed images/fg'
    # alpha_dir = 'data/Combined_Dataset/Training_set/Adobe-licensed images/alpha'
    # save_dir = 'results/adobe_train_adobe'
    # process_adobe_dataset(img_dir, alpha_dir, save_dir)

    # print('------------process adobe train Others Image-----------------')
    # img_dir = 'data/Combined_Dataset/Training_set/Other/fg'
    # alpha_dir = 'data/Combined_Dataset/Training_set/Other/alpha'
    # save_dir = 'results/adobe_train_others'
    # process_adobe_dataset(img_dir, alpha_dir, save_dir)

    # generate_show_raw_image()