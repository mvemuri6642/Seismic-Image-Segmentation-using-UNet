import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

#####code to remove small and large mask and images from dataset

images=os.listdir('D:/ML_project/test/images/')
masks=os.listdir('D:/ML_project/test/masks/')

count=0
for i in images:
    print(count)
    img = cv2.imread(os.path.join('D:/ML_project/test/images/',i))
    mask = cv2.imread(os.path.join('D:/ML_project/test/masks/',i),0)
    y,x=mask.shape[:2]
    ratio=np.sum(mask==255)/y*x

    # plt.imshow(mask,'gray')
    # plt.title(str(ratio))
    # plt.show()

    if(ratio>=500 and ratio<=7000):

        a='D:/ML_project/test/images_new_selected/'+str(count)+".png"
        b = 'D:/ML_project/test/masks_new_selected/' + str(count) + ".png"

        cv2.imwrite(a,img)
        cv2.imwrite(b, mask)
        count=count+1


############################################################################

# from skimage import exposure
# import cv2
# import matplotlib.pyplot as plt
# import os
# from skimage.filters import unsharp_mask
#
# dirpath = 'C:/Users/Naveen/Desktop/Machine learning/project/test/test/images_new_selected/'
# dirpath2 = 'C:/Users/Naveen/Desktop/Machine learning/project/test/test/masks_new_selected/'
#
# imgpths=[]
#
# print(len(os.listdir(dirpath)))
#
# # for pth in os.listdir(dirpath):
# #     #     print("h")
# #     #     if '.png' in pth[10:]:
# #     #         print("hi")
# #     img = cv2.imread(os.path.join(dirpath, pth), 0)
# #
# #     # gamma_corrected = exposure.adjust_gamma(img, 1.3) #0.7 to 1.3
# #     print(img.shape)
# #     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
# #     gamma_corrected = clahe.apply(img)
# #     print(gamma_corrected.shape)
# #     gamma_corrected = np.reshape(gamma_corrected, img.shape + (1,))
# #     print(gamma_corrected.shape)
# #
# #     # gamma_corrected = unsharp_mask(img, radius=1, amount=1.4)
# #     # gamma_corrected = 255 * gamma_corrected  # Now scale by 255
# #     # gamma_corrected = gamma_corrected.astype(np.uint8)
# #     # print(img.dtype)
# #     # print(gamma_corrected.dtype)
# #
# #
# #     plt.subplot(121), plt.imshow(img, 'gray')
# #     plt.subplot(122), plt.imshow(gamma_corrected, 'gray')
# #     plt.show()
# #
# # import random
# # gamma_values=[0.7,0.8,0.9,1,1.1,1.2,1.3]
# # unsharp_vales=[1.1,1.2,1.3]
# # r1=[0,1]
# # r2=[0,1]
# #
# # if random.choice(r1)==1:
# #     if random.choice(r2)==1:
# #         img = exposure.adjust_gamma(img, random.choice(gamma_values))
# #         if random.choice(r2)==1:
# #             clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
# #             img = clahe.apply(img)
# #     else:
# #         if random.choice(r2)==1:
# #             clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
# #             img = clahe.apply(img)
# #         else:
# #             img = unsharp_mask(img, radius=1, amount=random.choice(unsharp_vales))
# #             img = 255 * img
# #             img = img.astype(np.uint8)
# #
#
# for pth in os.listdir(dirpath)[50:]:
#
#     img = cv2.imread(os.path.join(dirpath, pth), 0)
#     msk = cv2.imread(os.path.join(dirpath2, pth), 0)
#
#     # gamma_corrected = exposure.adjust_gamma(img, 1.3) #0.7 to 1.3
#
#     gamma_corrected = unsharp_mask(img, radius=1, amount=1.3) #1.1,1.2,1.3
#     gamma_corrected = 255 * gamma_corrected
#
#     # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#     # gamma_corrected = clahe.apply(img)
#     # gamma_corrected = gamma_corrected.astype(np.uint8)
#
#     res=np.hstack((msk,img,gamma_corrected))
#     plt.imshow(res,'gray')
#     plt.show()
#
#
