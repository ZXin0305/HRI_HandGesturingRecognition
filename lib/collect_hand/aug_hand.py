import cv2
import ctypes
import numpy as np
from IPython import embed
from matplotlib import pyplot as plt
from path import Path
import math
import random
from skimage import data, exposure, img_as_float

#  ==========================  #
#  给三种方式  #
#  1. depth mask
#  2. cv2.grabCut
#  3. depth mask + cv2.grabCut
#  其中:2的分割效果较好,1因为depth mask会有一定的shallow
#  但是如果1的mask比较好的话,效果也是可以的,同时比较快
rgb_suffix = 'cropped_img_'
depth_suffix = 'cropped_depth_'

def segment_hand(img, depth, bg, mode):
    img_shape = img.shape
    bg_ = cv2.resize(bg, (224,224))
    if mode == 1:
        size_ = 3
        ratio_ = 0.15
        max_val = 1 + ratio_
        min_val = 1 - ratio_
        img_copy = img.copy()
        img_center = [int(img_shape[0] / 2 + 0.5), int(img_shape[1] / 2 + 0.5)]
        depth = depth / 255.0 * 4096.0 / 10 # --> cm
        total_depth = 0.0
        for i in range(img_center[0] - size_, img_center[0] + size_):
            for j in range(img_center[1] - size_, img_center[1] + size_):
                total_depth += depth[i,j]  
        mean_depth_val = total_depth / (pow(size_ * 2, 2.0))
        depth_mask = depth.copy()
        depth_mask = depth_mask[:,:,0]
        depth_mask[depth_mask > mean_depth_val[0] * max_val] = 0
        depth_mask[depth_mask < mean_depth_val[0] * min_val] = 0
        depth_mask[depth_mask != 0] = 1 

        for i in range(3):
            img_copy[:,:,i] = depth_mask * img_copy[:,:,i]

        depth_mask_inverse = depth_mask.copy()
        depth_mask_inverse[depth_mask == 1] = 0
        depth_mask_inverse[depth_mask == 0] = 1
        for i in range(3):
            bg_[:,:,i] = depth_mask_inverse * bg_[:,:,i]
        img_mask = img_copy + bg_       
        cv2.imshow('depth_mask', depth_mask)
        cv2.imshow('img_new', img_mask)
        cv2.imwrite('./depth_mask.jpg', depth_mask * 255)
        cv2.imwrite('./depth_mask_in.jpg', depth_mask_inverse * 255)
        cv2.imwrite('./ori_img.jpg', img)
        cv2.imwrite('./mask_bg.jpg', bg_)
        cv2.imwrite('./mask_img.jpg', img_mask)
        cv2.imwrite('./tmp_img.jpg', img_copy)
 
    elif mode == 2:
        mask = np.zeros(img.shape[:2], np.uint8)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        rect = (1, 1, img.shape[1], img.shape[0])
        cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 50, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")
        img_copy = img * mask2[:, :, np.newaxis]
        depth_mask_inverse = mask2.copy()
        depth_mask_inverse[mask2 == 1] = 0
        depth_mask_inverse[mask2 == 0] = 1
        for i in range(3):
            bg_[:,:,i] = depth_mask_inverse * bg_[:,:,i]  
        img_copy = img_copy + bg_        
        cv2.imshow('mask', mask2 * 255)
        cv2.imshow('img_new', img_copy)
        # cv2.imwrite('./depth_mask_grab.jpg', mask2 * 255)
        # cv2.imwrite('./img_grab.jpg', img_copy)

    elif mode == 3:
        size_ = 3
        ratio_ = 0.15
        max_val = 1 + ratio_
        min_val = 1 - ratio_
        img_copy = img.copy()
        img_center = [int(img_shape[0] / 2 + 0.5), int(img_shape[1] / 2 + 0.5)]
        depth = depth / 255.0 * 4096.0 / 10 # --> cm
        total_depth = 0.0
        for i in range(img_center[0] - size_, img_center[0] + size_):
            for j in range(img_center[1] - size_, img_center[1] + size_):
                total_depth += depth[i,j]  
        mean_depth_val = total_depth / (pow(size_ * 2, 2.0))
        depth_mask = depth.copy()
        depth_mask = depth_mask[:,:,0]
        depth_mask[depth_mask > mean_depth_val[0] * max_val] = 0
        depth_mask[depth_mask < mean_depth_val[0] * min_val] = 0
        depth_mask[depth_mask != 0] = 1 
        for i in range(3):
            img_copy[:,:,i] = depth_mask * img_copy[:,:,i]
        mask = np.zeros(img_copy.shape[:2], np.uint8)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        rect = (1, 1, img_copy.shape[1], img_copy.shape[0])
        cv2.grabCut(img_copy, mask, rect, bgdModel, fgdModel, 100, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")
        img_copy = img_copy * mask2[:, :, np.newaxis]
        cv2.imshow('img_new', img_copy)

    key = cv2.waitKey(0)  
    if key == 27:
        return 'break'  

def process_bg(bg):
    """
    处理背景图
    """
    bg_shape = bg.shape
    crop_size = 350
    crop_bg_x = random.randint(10, int(bg_shape[1]/2))
    crop_bg_y = random.randint(10, int(bg_shape[0]/2))
    cropped_bg = bg[crop_bg_y:crop_bg_y+crop_size, crop_bg_x:crop_bg_x+crop_size, :]
    cv2.imwrite('./cropped_bg.jpg', cropped_bg)
    return cropped_bg

#  其余的离线数据增强的方式
#  高斯模糊
def AugGaussianFilter(img, ksize=[7,7]):
    img_copy = img.copy()
    img_copy = cv2.GaussianBlur(img_copy, ksize, 0, 0)
    return img_copy

# 高斯噪声
def AugGaussianNoise(img, loc=0.0, sigma=0.1):
    img_copy = img.copy()
    img_copy = np.array(img_copy / 255, dtype=np.float)
    noise = np.random.normal(loc, sigma, img_copy.shape)    # 正态分布函数
    gaussian_noise = img_copy + noise
    gaussian_noise = np.clip(gaussian_noise, 0, 1)
    gaussian_noise_img = np.uint8(gaussian_noise * 255)
    return gaussian_noise_img

# 椒盐噪声
def AugSaltNoise(img, s_vs_p=0.5, amount = 0.02):
    """
    :param img: 原图
    :param s_vs_p: 椒盐噪声中椒 ：盐比例
    :param amount: 实施椒盐噪声的元素的数量
    :return:
    """

    img_copy = img.copy()

    # 添加salt噪声
    num_salt = np.ceil(amount * img_copy.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img_copy.shape[:2]]
    img_copy[coords] = 255

    # 添加pepper噪声
    num_pepper = np.ceil(amount * img_copy.size * (1 - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in img_copy.shape[:2]]
    img_copy[coords] = 0
    return img_copy

# 直方图均衡化
def AugHistEqua(img, mask=None, L=256):
    img_copy = img.copy()
    h, w = img_copy.shape[0], img_copy.shape[1]
    hist = cv2.calcHist([img_copy], [0], mask, [256], [0, 255])  # 计算图像的直方图，即存在的每个灰度值的像素点数量
    # plt.plot(hist)
    # plt.show()
    hist[0:255] = hist[0:255] / (h * w)  # 计算灰度值的像素点的概率，除以所有像素点个数，就归一化
    # 设置si
    sum_hist = np.zeros(hist.shape)
    # 开始计算si的一部分值，i每一次增大，si都是对前i个灰度值的分布概率进行累加
    for i in range(256):
        sum_hist[i] = sum(hist[0:i+1])
    equal_hist = np.zeros(sum_hist.shape)
    # si再乘以灰度级，再四舍五入
    for i in range(256):
        equal_hist[i] = int(((L - 1) - 0) * sum_hist[i] + 0.5)
    img_equal = img_copy.copy()
    # 新图片的创建
    for i in range(h):
        for j in range(w):
            img_equal[i, j, 0] = equal_hist[img_copy[i, j, 0]]
            img_equal[i, j, 1] = equal_hist[img_copy[i, j, 1]]
            img_equal[i, j, 2] = equal_hist[img_copy[i, j, 2]]
    return img_equal

    # equal_hist = cv2.calcHist([img_equal], [0], mask, [256], [0, 255])
    # plt.plot(equal_hist)
    # plt.show()
    
# 三通道中方图均衡化
def AugHistEqual2(img):
    img_copy = img.copy()
    (b, g, r) = cv2.split(img_copy)
    bH = cv2.equalizeHist(b)
    gH = cv2.equalizeHist(g)
    rH = cv2.equalizeHist(r)

    result = cv2.merge((bH, gH, rH))
    return result

# 图像亮度增强
def AugBright(img, bright_score=0.3):
    img_copy = img.copy()
    img_copy = exposure.adjust_gamma(img_copy, bright_score)
    return img_copy

# 图像亮度变暗
def AugDark(img, dark_score=2):
    img_copy = img.copy()
    img_copy = exposure.adjust_gamma(img_copy, dark_score)
    return img_copy

if __name__ == "__main__":
    root_img_dir = '/home/xuchengjun/ZXin/smap/depth_and_rgb'
    bg_file_name = '../../bg.jpeg'
    mode = 1
    img_list = Path(root_img_dir).files()

    for i in range(int(len(img_list) / 2)):
        print(f'processing .. {i}')
        rgb_img_name = root_img_dir + '/' + rgb_suffix + str(i) + '.jpg'
        depth_img_name = root_img_dir + '/' + depth_suffix + str(i) + '.png'
        img = cv2.imread(rgb_img_name)
        depth = cv2.imread(depth_img_name) 
        bg = cv2.imread(bg_file_name) 
        cropped_bg = process_bg(bg)
        flag = segment_hand(img, depth, cropped_bg, mode)
        if flag == 'break':
            break