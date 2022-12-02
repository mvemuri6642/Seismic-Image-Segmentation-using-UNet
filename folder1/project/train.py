import segmentation_models as sm
from data import *
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from keras.callbacks import ModelCheckpoint
import keras
import matplotlib.pyplot as plt

data_gen_args = dict(rotation_range=0.3, #0.2 before
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    vertical_flip=True, #new
                    fill_mode='nearest')

BACKBONE = 'resnet101' #'vgg16','resnet23','resnet50','efficientnetb0'
preprocess_input = sm.get_preprocessing(BACKBONE)

myGene = trainGenerator(6,'C:/Users/Naveen/Desktop/Machine learning/project/new_17_11_22/dataset/train/','images_new_selected','masks_new_selected',data_gen_args,save_to_dir = None,preprocess_input=preprocess_input,image_color_mode = "grayscale",mask_color_mode="grayscale",target_size = (224,224)) #rgb,grayscale
validationGene = trainGenerator(6,'C:/Users/Naveen/Desktop/Machine learning/project/new_17_11_22/dataset/train/','images_new_selected','masks_new_selected',data_gen_args,save_to_dir = None,preprocess_input=preprocess_input,image_color_mode = "grayscale",mask_color_mode="grayscale",target_size = (224,224)) #rgb,grayscale

# encoder_weights='imagenet'
# model=sm.Unet(BACKBONE,encoder_weights=encoder_weights,weights=weights)
weights_path=r"C:\Users\Naveen\Desktop\Machine learning\project\new_17_11_22\train_summary\resnet101\bestresnet101.h5"
model=sm.Unet(BACKBONE,input_shape=(224,224,1),encoder_weights=None,weights=weights_path)
# model=sm.Unet(BACKBONE,input_shape=(528,528,1),encoder_weights=None,weights=None)

LR=0.01
optim=keras.optimizers.Adam(LR)
# optim=keras.optimizers.Adagrad(LR)
model.compile(optim,loss=sm.losses.binary_focal_dice_loss,metrics=[sm.metrics.iou_score])
# model.compile(optim,loss=sm.losses.binary_crossentropy,metrics=[sm.metrics.iou_score])
# model.compile(optim,loss=sm.losses.binary_focal_loss,metrics=[sm.metrics.iou_score])

callbacks=[keras.callbacks.ModelCheckpoint(os.path.join('./','tester.h5'),monitor='val_loss',save_weights_only=True,save_best_only=True,mode='min'),
keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=0.1,patience=4,verbose=2,mode='min')]

# callbacks=[keras.callbacks.ModelCheckpoint(os.path.join('./','bestaugresnet101.h5'),monitor='val_loss',save_weights_only=True,save_best_only=True,mode='min')]


# history=model.fit_generator(myGene,callbacks=callbacks,steps_per_epoch=3000,epochs=4)
history=model.fit_generator(myGene,callbacks=callbacks,steps_per_epoch=200,epochs=200,validation_data=validationGene,validation_steps=50) #200,150
print(history.history.keys())

plt.plot(history.history['iou_score'])
plt.plot(history.history['val_iou_score'])
plt.title('model iou score')
plt.ylabel('score')
plt.xlabel('epoch')
# plt.legend(['train','test'],iou='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model  loss')
plt.ylabel('loss')
plt.xlabel('epoch')
# plt.legend(['train','test'],iou='upper left')
plt.show()

plt.plot(history.history['lr'])
plt.title('Reduce lr on plateau')
plt.ylabel('lr')
plt.xlabel('epoch')
# plt.legend(['train','test'],iou='upper left')
plt.show()

# model.save('.../.h5')

import pandas as pd
lr=list(history.history['lr'])
train_loss=list(history.history['loss'])
val_loss=list(history.history['val_loss'])
train_iou=list(history.history['iou_score'])
val_iou=list(history.history['val_iou_score'])

data = {
   'lr': lr,
   'train_loss': train_loss,
   'val_loss': val_loss,
    'train_iou':train_iou,
    'val_iou':val_iou,
}

df = pd.DataFrame.from_dict(data)

df.to_csv('bestaugresnet101.csv')