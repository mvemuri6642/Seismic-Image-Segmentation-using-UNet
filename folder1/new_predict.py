import segmentation_models as sm
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import numpy as np
import cv2
import matplotlib.pyplot as plt

BACKBONE = 'resnet101'
model=sm.Unet(BACKBONE,input_shape=(224,224,1),encoder_weights=None,weights='C:/Users/Naveen/Desktop/Machine learning/project/new_17_11_22/train_summary/resnet_101_aug_adam/bestaugresnet101.h5')

# print(model.summary)
# exit()

dirpath = 'C:/Users/Naveen/Desktop/Machine learning/project/test/test/images_new_selected/'

imgpths=[]

for pth in os.listdir(dirpath):
    if '.png' in pth:
        imgpths.append(os.path.join(dirpath,pth))

val_mask = 'C:/Users/Naveen/Desktop/Machine learning/project/test/test/masks_new_selected/'
avg_iou=0
avg_f1=0
avg_acc=0
avg_p=0
avg_r=0

for ipth in imgpths[:]:
    basename=os.path.basename(ipth)
    val_gt=os.path.join(val_mask,basename)
    g_t=cv2.imread(val_gt,0)
    img=cv2.imread(ipth,0)
    rimg=img.copy()
    g_t_img=img.copy()
    y,x=img.shape[:2]
    img=cv2.resize(img,(224,224),interpolation = cv2.INTER_CUBIC)
    img=img/255

    img = np.reshape(img,(1,)+img.shape)
    img = np.reshape(img,img.shape+(1,))
    # print(img.shape)
    seg_res=model.predict(img)[0]
    seg_res=np.array(seg_res*255)
    seg_res=cv2.convertScaleAbs(seg_res)
    seg_res = cv2.resize(seg_res, (x, y), interpolation=cv2.INTER_CUBIC)
    seg_res_copy=seg_res.copy()
    ret, seg_res = cv2.threshold(seg_res, 90, 255, cv2.THRESH_BINARY)


    g_t=np.array(g_t)
    seg_res=np.array(seg_res)

    g_t_copy = g_t.copy()
    seg_res_copy = seg_res.copy()


    def iou_score(g_t,seg_res):
        mask1_area = np.count_nonzero(g_t == 255)
        mask2_area = np.count_nonzero(seg_res == 255)
        intersection = np.count_nonzero( np.logical_and( g_t, seg_res) )
        iou = round(((intersection/(mask1_area+mask2_area-intersection))*100),2)
        return iou

    iou=iou_score(g_t,seg_res)
    avg_iou=avg_iou+iou


    def score(g_t,seg_res):
        g_t[g_t==255]=1
        seg_res[seg_res == 255] = 1
        tp = (g_t * seg_res).sum()
        tn = ((1 - g_t) * (1 - seg_res)).sum()
        fp = ((1 - g_t) * seg_res).sum()
        fn = (g_t * (1 - seg_res)).sum()
        epsilon = 1e-7

        precision = tp / (tp + fp + epsilon)
        recall = tp / (tp + fn + epsilon)
        beta=1
        f1 = (1 + beta ** 2) * (precision * recall) / (beta ** 2 * precision + recall + epsilon)
        return f1,precision,recall

    def accuracy(g_t,seg_res):
        g_t[g_t==255]=1
        seg_res[seg_res == 255] = 1
        tp = (g_t * seg_res).sum()
        tn = ((1 - g_t) * (1 - seg_res)).sum()
        fp = ((1 - g_t) * seg_res).sum()
        fn = (g_t * (1 - seg_res)).sum()

        acc = (tp+tn)/(tp+tn+fn+fp)
        return acc

    acc=accuracy(g_t,seg_res)
    avg_acc=avg_acc+acc

    f1,p,r=score(g_t,seg_res)
    avg_f1=avg_f1+f1
    avg_p = avg_p + p
    avg_r = avg_r + r


    # plt.subplot(131),plt.imshow(g_t_img,'gray')
    # plt.subplot(132), plt.imshow(seg_res, 'gray')
    # plt.subplot(133), plt.imshow(g_t, 'gray')
    # plt.title(str(iou))
    # plt.show()



    contours, hierarchy = cv2.findContours(seg_res, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(g_t_img, contours, -1, (255, 255, 255), 1)
    res_path = 'C:/Users/Naveen/Desktop/Machine learning/project/test/test/res/' + basename
    # stack = np.hstack((rimg, g_t_img,g_t,seg_res))
    stack = np.hstack((rimg, g_t_img, g_t_copy, seg_res_copy))
    cv2.imwrite(res_path,stack)
    # plt.imshow(stack,'gray')
    # plt.show()

print("avg_iou=",avg_iou/len(imgpths))
print("avg_f1=",avg_f1/len(imgpths))
print("avg_acc=",avg_acc/len(imgpths))
print("avg_p=",avg_p/len(imgpths))
print("avg_r=",avg_r/len(imgpths))


