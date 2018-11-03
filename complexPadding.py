from keras import backend as K
from keras.engine.topology import Layer
# from keras.engine.base_layer import Layer
from keras.engine.base_layer import InputSpec
from keras.utils import conv_utils
# import numpy as np

class ComplexPadding2D(Layer):
    
    def __init__(self,
                padding=(1, 1),
                mode='CONSTANT',
                data_format=None,
                debug=False,
                **kwargs):
        '''
        初始化函数，来自ZeroPadding2D源代码
        '''
        self.debug = debug
        if self.debug :
            print('# ComplexPadding2D # __init__. Now in debug mode')
            # print('**kwargs :',**kwargs)
        super(ComplexPadding2D, self).__init__(**kwargs)    # 调用父类初始化函数
        # 保存填充模式
        self.mode = mode
        if self.debug :
            print('# ComplexPadding2D # Padding mode is', self.mode)
        # 处理数据格式参数
        self.data_format = K.normalize_data_format(data_format)
        if self.debug :
            print('# ComplexPadding2D # data_format is', self.data_format)
        # 处理padding参数
        if isinstance(padding, int):
            self.padding = ((padding, padding), (padding, padding))
        elif hasattr(padding, '__len__'):
            if len(padding) != 2:
                raise ValueError('`padding` should have two elements. '
                                 'Found: ' + str(padding))
            height_padding = conv_utils.normalize_tuple(padding[0], 2,
                                                        '1st entry of padding')
            width_padding = conv_utils.normalize_tuple(padding[1], 2,
                                                       '2nd entry of padding')
            self.padding = (height_padding, width_padding)
        else:
            raise ValueError('`padding` should be either an int, '
                             'a tuple of 2 ints '
                             '(symmetric_height_pad, symmetric_width_pad), '
                             'or a tuple of 2 tuples of 2 ints '
                             '((top_pad, bottom_pad), (left_pad, right_pad)). '
                             'Found: ' + str(padding))
        if self.debug :
            print('# ComplexPadding2D # Padding is',self.padding)
        self.input_spec = InputSpec(ndim=4)
    
    def build(self, input_shape):
        '''
        源码来自https://keras.io/zh/layers/writing-your-own-keras-layers/
        ZeroPadding2D中没有实现该函数
        '''
        if self.debug :
            print('# ComplexPadding2D # build')
        super(ComplexPadding2D, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, input):
        '''
        核心部分
        '''
        # return K.dot(x, self.kernel)
        if self.debug :
            print('# ComplexPadding2D # call')
        [[lenTopPad,lenBottomPad],[lenLeftPad,lenRightPad]] = self.padding
        input_shape = K.int_shape(input)
        if self.debug :
            print('# ComplexPadding2D # input :',input)
            print('# ComplexPadding2D # input_shape :',input_shape)
            print('# ComplexPadding2D # mode :',self.mode,', data_format :',self.data_format)
        if self.data_format == 'channels_first':
            [cntBatch, cntChannel, cntRow, cntColumn] = input_shape
            if self.mode == 'CONSTANT' :
                # 补零
                topPad = K.zeros((cntBatch,cntChannel,lenTopPad,cntColumn))
                bottomPad = K.zeros((cntBatch,cntChannel,lenBottomPad,cntColumn))
                paddedRow = K.concatenate([topPad,input,bottomPad],2)
                cntRowNew = cntRow + lenTopPad + lenBottomPad
                leftPad = K.zeros((cntBatch,cntChannel,cntRowNew,lenLeftPad))
                rightPad = K.zeros((cntBatch,cntChannel,cntRowNew,lenRightPad))
                paddedAll = K.concatenate([leftPad,paddedRow,rightPad],3)
            elif self.mode == 'REFLECT' : 
                # 镜像，不重复边缘
                topPad = K.reverse(input[:,:,1:lenTopPad+1,:],2)            # 前lenTopPad行
                bottomPad = K.reverse(input[:,:,-lenBottomPad-1:-1,:],2)    # 后lenBottomPad行
                paddedRow = K.concatenate([topPad,input,bottomPad],2)
                leftPad = K.reverse(paddedRow[:,:,:,1:lenLeftPad+1],3)      # 前lenLeftPad列
                rightPad = K.reverse(paddedRow[:,:,:,-lenRightPad-1:-1],3)      # 后lenRightPad列
                paddedAll = K.concatenate([leftPad,paddedRow,rightPad],3)
            elif self.mode == 'SYMMETRIC' :
                # 镜像，重复边缘
                topPad = K.reverse(input[:,:,:lenTopPad,:],2)           # 前lenTopPad行
                bottomPad = K.reverse(input[:,:,-lenBottomPad:,:],2)    # 后lenBottomPad行
                paddedRow = K.concatenate([topPad,input,bottomPad],2)
                leftPad = K.reverse(paddedRow[:,:,:,:lenLeftPad],3)     # 前lenLeftPad列
                rightPad = K.reverse(paddedRow[:,:,:,-lenRightPad:],3)      # 后lenRightPad列
                paddedAll = K.concatenate([leftPad,paddedRow,rightPad],3)
        elif self.data_format == 'channels_last':
            [cntBatch, cntRow, cntColumn, cntChannel] = input_shape
            if self.mode == 'CONSTANT' :
                # 补零
                topPad = K.zeros((cntBatch,lenTopPad,cntColumn,cntChannel))
                bottomPad = K.zeros((cntBatch,lenBottomPad,cntColumn,cntChannel))
                paddedRow = K.concatenate([topPad,input,bottomPad],1)
                cntRowNew = cntRow + lenTopPad + lenBottomPad
                leftPad = K.zeros((cntBatch,cntRowNew,lenLeftPad,cntChannel))
                rightPad = K.zeros((cntBatch,cntRowNew,lenRightPad,cntChannel))
                paddedAll = K.concatenate([leftPad,paddedRow,rightPad],2)
            elif self.mode == 'REFLECT' : 
                # 镜像，不重复边缘
                topPad = K.reverse(input[:,1:lenTopPad+1,:,:],1)            # 前lenTopPad行
                bottomPad = K.reverse(input[:,-lenBottomPad-1:-1,:,:],1)    # 后lenBottomPad行
                paddedRow = K.concatenate([topPad,input,bottomPad],1)
                leftPad = K.reverse(paddedRow[:,:,1:lenLeftPad+1,:],2)      # 前lenLeftPad列
                rightPad = K.reverse(paddedRow[:,:,-lenRightPad-1:-1,:],2)      # 后lenRightPad列
                paddedAll = K.concatenate([leftPad,paddedRow,rightPad],2)
            elif self.mode == 'SYMMETRIC' :
                # 镜像，重复边缘
                topPad = K.reverse(input[:,:lenTopPad,:,:],1)           # 前lenTopPad行
                bottomPad = K.reverse(input[:,-lenBottomPad:,:,:],1)    # 后lenBottomPad行
                paddedRow = K.concatenate([topPad,input,bottomPad],1)
                leftPad = K.reverse(paddedRow[:,:,:lenLeftPad,:],2)     # 前lenLeftPad列
                rightPad = K.reverse(paddedRow[:,:,-lenRightPad:,:],2)      # 后lenRightPad列
                paddedAll = K.concatenate([leftPad,paddedRow,rightPad],2)
        if self.debug :
            print('# ComplexPadding2D # output shape :',K.int_shape(paddedAll))
        return(paddedAll)

    def compute_output_shape(self, input_shape):
        '''
        计算输出张量尺寸，来自ZeroPadding2D源代码
        '''
        if self.debug :
            print('# ComplexPadding2D # compute_output_shape')
        if self.data_format == 'channels_first':
            if input_shape[2] is not None:
                rows = input_shape[2] + self.padding[0][0] + self.padding[0][1]
            else:
                rows = None
            if input_shape[3] is not None:
                cols = input_shape[3] + self.padding[1][0] + self.padding[1][1]
            else:
                cols = None
            t = (input_shape[0], input_shape[1], rows, cols)
            if self.debug :
                print('# ComplexPadding2D # output_shape :', t)
            return(t)
        elif self.data_format == 'channels_last':
            if input_shape[1] is not None:
                rows = input_shape[1] + self.padding[0][0] + self.padding[0][1]
            else:
                rows = None
            if input_shape[2] is not None:
                cols = input_shape[2] + self.padding[1][0] + self.padding[1][1]
            else:
                cols = None
            t = (input_shape[0], rows, cols, input_shape[3])
            if self.debug :
                print('# ComplexPadding2D # output_shape :', t)
            return(t)
    
    def get_config(self):
        '''
        修改自ZeroPadding2D源代码
        '''
        if self.debug :
            print('# ComplexPadding2D # get_config')
        config = {'padding': self.padding,
                  'data_format': self.data_format,
                  'mode': self.mode}
        base_config = super(ComplexPadding2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))