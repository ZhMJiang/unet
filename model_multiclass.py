import numpy as np 
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras

from complexPadding import *


def unet(pretrained_weights = None,input_size = (256,256,1),use_mirror_padding = False,loss = 'binary_crossentropy'):
    
    # 输入层
    input_s = Input(input_size)
    
    # Padding
    if use_mirror_padding :
        downPadding     = 'valid'
        upPadding       = 'valid'
        input_p         = ComplexPadding2D(padding=110,mode='SYMMETRIC',debug = True)(input_s)
        croppingList    = [5,19,48,105]
    else :
        downPadding     = 'same'
        upPadding       = 'same'
        input_p = input_s
    
    # 向下的卷积层
    conv1 = Conv2D(64, 3, activation = 'relu', padding = downPadding, kernel_initializer = 'he_normal')(input_p)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = downPadding, kernel_initializer = 'he_normal')(conv1)
    print('conv1 :',conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    print('pool1 :',pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = downPadding, kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = downPadding, kernel_initializer = 'he_normal')(conv2)
    print('conv2 :',conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    print('pool2 :',pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = downPadding, kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = downPadding, kernel_initializer = 'he_normal')(conv3)
    print('conv3 :',conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    print('pool3 :',pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = downPadding, kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = downPadding, kernel_initializer = 'he_normal')(conv4)
    print('conv4 :',conv4)
    drop4 = Dropout(0.5)(conv4)
    print('drop4 :',drop4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
    print('pool4 :',pool4)
    
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = downPadding, kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = downPadding, kernel_initializer = 'he_normal')(conv5)
    print('conv5 :',conv5)
    drop5 = Dropout(0.5)(conv5)
    print('drop5 :',drop5)
    
    # 向上的卷积、升采样层
    up6 = Conv2D(512, 3, activation = 'relu', padding = upPadding, kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    print('up6 :',up6)
    # merge6 = merge([drop4,up6], mode = 'concat', concat_axis = 3)
    if not use_mirror_padding :
        merge6 = concatenate([drop4,up6], axis = 3)
    else :
        merge6 = concatenate([Cropping2D(croppingList[0])(drop4),up6], axis = 3)
    print('merge6 :',merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = upPadding, kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = upPadding, kernel_initializer = 'he_normal')(conv6)
    print('conv6 :',conv6)
    
    up7 = Conv2D(256, 3, activation = 'relu', padding = upPadding, kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    print('up7 :',up7)
    # merge7 = merge([conv3,up7], mode = 'concat', concat_axis = 3)
    if not use_mirror_padding :
        merge7 = concatenate([conv3,up7], axis = 3)
    else :
        merge7 = concatenate([Cropping2D(croppingList[1])(conv3),up7], axis = 3)
    print('merge7 :',merge7)
    conv7 = Conv2D(256, 4, activation = 'relu', padding = upPadding, kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = upPadding, kernel_initializer = 'he_normal')(conv7)
    print('conv7 :',conv7)
    
    up8 = Conv2D(128, 3, activation = 'relu', padding = upPadding, kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    # up8 = Conv2D(128, 3, activation = 'relu', padding = upPadding, kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    print('up8 :',up8)
    # merge8 = merge([conv2,up8], mode = 'concat', concat_axis = 3)
    if not use_mirror_padding :
        merge8 = concatenate([conv2,up8], axis = 3)
    else :
        merge8 = concatenate([Cropping2D(croppingList[2])(conv2),up8], axis = 3)
    print('merge8 :',merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = upPadding, kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = upPadding, kernel_initializer = 'he_normal')(conv8)
    print('conv8 :',conv8)
    
    up9 = Conv2D(64, 3, activation = 'relu', padding = upPadding, kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    print('up9 :',up9)
    # merge9 = merge([conv1,up9], mode = 'concat', concat_axis = 3)
    if not use_mirror_padding :
        merge9 = concatenate([conv1,up9], axis = 3)
    else :
        merge9 = concatenate([Cropping2D(croppingList[3])(conv1),up9], axis = 3)
    print('merge9 :',merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = upPadding, kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = upPadding, kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = upPadding, kernel_initializer = 'he_normal')(conv9)
    print('conv9 :',conv9)
    # conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)
    # print('conv10 :',conv10)
    multiClassify = Softmax(axis=-1)(conv9)
    print('multiClassify :',multiClassify)
    
    # model = Model(outputs = [conv10], inputs = [input_s])
    model = Model(outputs = [multiClassify], inputs = [input_s])

    model.compile(optimizer = Adam(lr = 1e-4), loss = loss, metrics = ['accuracy'])
    # model.compile(optimizer = Adam(lr = 1e-4), loss = 'MSE', metrics = ['accuracy'])
    
    model.summary()

    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    return model


