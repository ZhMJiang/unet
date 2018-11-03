# from model import *
from model_fullpad import *
from data import *

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

__UsePretrainedWeights  = False
# __PretrainedWeights     = 'unet_membrane_rgb.hdf5'
__PretrainedWeights     = 'unet_fullpad_membrane_rgb_mse.hdf5'
# __PretrainedWeights     = 'unet_membrane.hdf5'
# __PretrainedWeights     = 'unet_fullpad_membrane.hdf5'
__TrainNetwork          = True
__useMirrorPadding      = True
__NETInputSize          = (None,None,3)
# __NETInputSize            = (None,None,1)

data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')
                    
#======构造网络======
# model = unet(input_size=(256,256,3),pretrained_weights = None)
model = unet(
            input_size=__NETInputSize,
            pretrained_weights = __PretrainedWeights if __UsePretrainedWeights else None,
            use_mirror_padding = __useMirrorPadding,
            loss = 'MSE'
            )

#======训练网络======
if __TrainNetwork :
    # myGene = trainGenerator(2,'data/membrane/train','image','label',data_gen_args,save_to_dir = None)
    myGene = trainGenerator(2, 'data/single_cell_rgb/train', 'image', 'label', data_gen_args, image_color_mode='rgb', save_to_dir = None)
    model_checkpoint = ModelCheckpoint(__PretrainedWeights, monitor='loss',verbose=1, save_best_only=True)
    model.fit_generator(myGene,steps_per_epoch=500,epochs=1,callbacks=[model_checkpoint])

#======进行测试======
# testGene = testGenerator("data/membrane/test")
# [test_name_list,testGene] = getTestGeneratorAndNameList("data/membrane/test", 'image', file_filter = '*.png', as_gray = True, target_size = (256,256))
[test_name_list,testGene] = getTestGeneratorAndNameList("data/single_cell_rgb/test", 'image', file_filter = '*.jpg', as_gray = False)

# results = model.predict_generator(testGene,30,verbose=1)
results = model.predict_generator(testGene,len(test_name_list),verbose=1)

# saveResult("data/membrane/test",results)
# saveResult("data/membrane/test/predict",results,test_name_list)
saveResult("data/single_cell_rgb/test",results,test_name_list)