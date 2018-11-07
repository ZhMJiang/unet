from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
import numpy as np 
import os
import glob
import skimage.io as io
import skimage.transform as trans

# Sky = [128,128,128]
# Building = [128,0,0]
# Pole = [192,192,128]
# Road = [128,64,128]
# Pavement = [60,40,222]
# Tree = [128,128,0]
# SignSymbol = [192,128,128]
# Fence = [64,64,128]
# Car = [64,0,128]
# Pedestrian = [64,64,0]
# Bicyclist = [0,128,192]
# Unlabelled = [0,0,0]

# COLOR_DICT = np.array([Sky, Building, Pole, Road, Pavement,
                          # Tree, SignSymbol, Fence, Car, Pedestrian, Bicyclist, Unlabelled])

# 分类标注样式
#   注意目前仅仅考虑第一个颜色通道（RGB的R）
Unlabelled  =   [0,     0,      0	]
A           =   [255,   0,		80	]		# 洋红
B           =   [128,   255,	192 ]		# 浅蓝
C           =   [64,    255,    64  ]		# 草绿
D           =   [192,   96,     255 ]		# 水粉紫

COLOR_DICT  =   [Unlabelled,A,B,C,D]
                          
def getMaskLabelArea(mask,label) :
    '''
    TODO
    把Mask中对应label的范围找出来
    输入
        mask    原始的mask，(batch,row,column,channel)
        label   [R,G,B]
    注意由于数据预处理的位移、缩放等操作，像素取值可能会出现混合
    在这个函数里试图处理这种情况
    '''
    tolerance = 32      # 容差
    mask = mask[:,:,:,0] if(len(mask.shape) == 4) else mask[:,:,0]
    ans = np.abs(mask-label[0]) <= tolerance;
    return(ans)

def adjustData(img,mask,flag_multi_class=False,num_class=2) :
    '''
    调整训练数据，主要功能是
        1.把输入图片img转换为浮点格式（0~1）
        2.根据输入标注mask生成网络输出样本mask',其中若shape(mask) = (batch,row,column,channel)
            a.若为二分类，则简单进行01化，shape(mask') = (batch,row,column,1)
            b.若为多分类，则对比COLOR_DICT，进行独热编码，shape(mask') = (batch,row,column,cntClass)
                *若mask_color_mode = "grayscale"，shape(mask) = (batch,row,column,1)
                *否则，shape(mask) = (batch,row,column,3)
    '''
    if(flag_multi_class) :
        # 多分类
        if(np.max(img) > 1):
            img = img / 255
        # mask = mask[:,:,:,0] if(len(mask.shape) == 4) else mask[:,:,0]
        # print('Mask, Min =',np.min(mask),', Max =',np.max(mask),', between =',np.sum((mask>np.min(mask))&(mask<np.max(mask))),', total =',np.sum(mask>=0))
        new_mask = np.zeros(mask.shape[:-1] + (num_class,))
        residual_mask = np.ones(mask.shape[:-1])==1     # 兜底
        for i in range(num_class):
            t = getMaskLabelArea(mask,COLOR_DICT[i])
            # print('adjustData ',COLOR_DICT[i][0],' with ',np.sum(t),'pixels')
            # new_mask[mask == COLOR_DICT[i][0],i] = 1
            new_mask[t,i] = 1
            residual_mask[t] = False
        # 兜底
        # print('adjustData residual with',np.sum(residual_mask))
        new_mask[residual_mask,0] = 1
        # new_mask = np.reshape(new_mask,(new_mask.shape[0],new_mask.shape[1]*new_mask.shape[2],new_mask.shape[3])) if flag_multi_class else np.reshape(new_mask,(new_mask.shape[0]*new_mask.shape[1],new_mask.shape[2]))
        # mask = new_mask
    else :
        # 二分类
        if(np.max(img) > 1):
            img = img / 255
        if(np.max(mask) > 1):
            mask = mask /255
        # 二分类时，使用固定阈值，进行01分类
        mask = mask[:,:,:,0] if(len(mask.shape) == 4) else mask[:,:,0]  # 取第一个颜色通道，并降维
        new_mask = np.zeros(mask.shape + (1,))                              # 重新升维
        new_mask[mask > 0.5,0] = 1
        # mask[mask > 0.5] = 1
        # mask[mask <= 0.5] = 0
    # return (img,mask)
    return (img,new_mask)



def trainGenerator(batch_size,train_path,image_folder,mask_folder,aug_dict,image_color_mode = "grayscale",
                    mask_color_mode = "grayscale",image_save_prefix  = "image",mask_save_prefix  = "mask",
                    flag_multi_class = False,num_class = 2,save_to_dir = None,target_size = (256,256),seed = 1):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,      # 当使用'grayscale'时图像颜色将只有单通道(batch_size, *target_size, 1)
        target_size = target_size,
        batch_size = batch_size,
        # save_to_dir = save_to_dir,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)
    train_generator = zip(image_generator, mask_generator)
    for (img,mask) in train_generator:
        img,mask = adjustData(img,mask,flag_multi_class,num_class)
        yield (img,mask)

def getTestGeneratorAndNameList(test_path, image_folder, file_filter = '*.png', target_size = None, flag_multi_class = False, as_gray = True):
    name_list = glob.glob(os.path.join(test_path,image_folder,file_filter))
    print('len(name_list)',len(name_list))
    def testGenerator():            # 赞美闭包
        for i in range(len(name_list)) :
            img = io.imread(name_list[i],as_gray = as_gray)
            if (np.max(img) > 1) :
                img = img / 255
            # 默认为可变输入大小
            if target_size : 
                img = trans.resize(img,target_size)
            # 在此时，若 as_gray = False，则img为MxNx3; 否则为 MxN
            # img = np.reshape(img,img.shape+(1,)) if (not flag_multi_class) else img
            img = np.reshape(img,img.shape+(1,)) if (as_gray) else img
            img = np.reshape(img,(1,)+img.shape)
            yield img
    return [name_list,testGenerator()]

def geneTrainNpy(image_path,mask_path,flag_multi_class = False,num_class = 2,image_prefix = "image",mask_prefix = "mask",image_as_gray = True,mask_as_gray = True):
    image_name_arr = glob.glob(os.path.join(image_path,"%s*.png"%image_prefix))
    image_arr = []
    mask_arr = []
    for index,item in enumerate(image_name_arr):
        img = io.imread(item,as_gray = image_as_gray)
        img = np.reshape(img,img.shape + (1,)) if image_as_gray else img
        mask = io.imread(item.replace(image_path,mask_path).replace(image_prefix,mask_prefix),as_gray = mask_as_gray)
        mask = np.reshape(mask,mask.shape + (1,)) if mask_as_gray else mask
        img,mask = adjustData(img,mask,flag_multi_class,num_class)
        image_arr.append(img)
        mask_arr.append(mask)
    image_arr = np.array(image_arr)
    mask_arr = np.array(mask_arr)
    return image_arr,mask_arr


def labelVisualize(num_class,color_dict,img):
    # img = img[:,:,0] if len(img.shape) == 3 else img
    # img_out = np.zeros(img.shape + (3,))
    # for i in range(num_class):
        # img_out[img == i,:] = color_dict[i]
    # return img_out / 255
    '''
    TODO
    '''
    img = img[:,:,1] if len(img.shape) == 3 else img
    img_out = np.zeros(img.shape + (3,))
    for i in range(3) :
        img_out[:,:,i] = img
    if np.max(img_out) > 1 :
        img_out = img_out / 255
    return img_out

# saveResult
#   input:
#       save_path,npyfile,name_list,flag_multi_class = False,num_class = 2
#   默认二值情况下输出,修改以支持任意名称
#   增加了输入 name_list 输入文件地址列表
def saveResult(save_path,npyfile,name_list,flag_multi_class = False,num_class = 2):
    # 从文件地址列表得到去掉后缀名的文件名列表
    file_name_list = list(map(lambda x : '.'.join(x.split('.')[:-1]),       # 去后缀
            list(map(lambda x : x[1],                           # 取文件名
                list(map(os.path.split,name_list))              # [(path,file_name)]
            ))
        ));
    for i,item in enumerate(npyfile):
        img = labelVisualize(num_class,COLOR_DICT,item) if flag_multi_class else item[:,:,0]
        # io.imsave(os.path.join(save_path,"%d_predict.png"%i),img)
        io.imsave(os.path.join(save_path,file_name_list[i]+"_predict.png"),img)