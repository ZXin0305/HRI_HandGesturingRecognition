import cv2
import ctypes
import numpy as np
from IPython import embed
from matplotlib import pyplot as plt

# img = cv2.imread("/home/xuchengjun/ZXin/00_08_00015350.jpg")
# h, w, c = img.shape
# if c > 1:
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# img_data = np.array(img, dtype=np.uint8)
# img_data = img_data.ctypes.data_as(ctypes.c_char_p)

# # cv2.imshow("img", img)
# # cv2.waitKey(0)

# # size_of_initial_crop = (200,200)
# # hand_center = (540, 960)
# test_pyc = ctypes.cdll.LoadLibrary("./lib/process_cpp/test_1.so")
# test_pyc.cdraw_rectangle(50, 55, (540, 960))

# depth = cv2.imread("./depth_and_rgb/cropped_depth_25.png")

# depth_mask = depth.copy()
# depth_mask[depth_mask > 35] = 0
# depth_mask[depth_mask < 10] = 0
# depth_mask[depth_mask != 0] = 1
# img = cv2.imread('./depth_and_rgb/cropped_img_25.jpg')
# # img_new = depth_mask * img
# # cv2.imshow('img_new', img_new)
# cv2.imshow('depth_mask', depth_mask * 255)
# cv2.waitKey(0)
# # embed()

img = cv2.imread('./depth_and_rgb/cropped_img_0.jpg')
img_copy = img.copy()
cv2.imshow('img', img)
img_shape = img.shape
img_center = [int(img_shape[0] / 2 + 0.5), int(img_shape[1] / 2 + 0.5)]
depth = cv2.imread("./depth_and_rgb/cropped_depth_0.png")
depth = depth / 255.0 * 4096.0 / 10  # --> cm
total_depth = 0.0
for i in range(img_center[0] - 3, img_center[0] + 3):
    for j in range(img_center[1] - 3, img_center[1] + 3):
        total_depth += depth[i,j]

mean_depth_val = total_depth / 36
depth_mask = depth.copy()
depth_mask = depth_mask[:,:,0]

depth_mask[depth_mask > mean_depth_val[0] * 1.18] = 0
depth_mask[depth_mask < mean_depth_val[0] * 0.82] = 0
depth_mask[depth_mask != 0] = 1

for i in range(3):
    img_copy[:,:,i] = depth_mask * img_copy[:,:,i]

cv2.imshow('depth_mask', depth_mask)
cv2.imshow('img_new', img_copy)
# cv2.waitKey(0)

# 细化
mask = np.zeros(img.shape[:2], np.uint8)
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)
rect = (1, 1, img.shape[1], img.shape[0])
cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 100, cv2.GC_INIT_WITH_RECT)
mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")
img = img * mask2[:, :, np.newaxis]

cv2.imshow('img_copy', img)
cv2.waitKey(0)

# plt.subplot(121)
# plt.imshow(cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)) 
# plt.title("grabcut")
# plt.xticks([])
# plt.yticks([])
# plt.show()


# if __name__ == "__main__":
#     img = cv2.imread("/home/xuchengjun/Desktop/hand.jpg")
#     OLD_IMG = img.copy()
#     # print(img.shape[0], img.shape[1])
#     mask = np.zeros(img.shape[:2], np.uint8)

#     bgdModel = np.zeros((1, 65), np.float64)
#     fgdModel = np.zeros((1, 65), np.float64)

#     rect = (1, 1, img.shape[1], img.shape[0])
#     print("123 ..")
#     cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 50, cv2.GC_INIT_WITH_RECT)

#     mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")
#     img = img * mask2[:, :, np.newaxis]

#     plt.subplot(121)
#     plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#     plt.title("grabcut")
#     plt.xticks([])
#     plt.yticks([])
#     plt.subplot(122)
#     plt.imshow(cv2.cvtColor(OLD_IMG, cv2.COLOR_BGR2RGB))
#     plt.title("original")
#     plt.xticks([])
#     plt.yticks([])

#     plt.show()