import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.layers import (BatchNormalization, Conv2D, MaxPooling2D,
                                     LeakyReLU, Input, concatenate, Reshape,
                                     GlobalAveragePooling2D, Softmax)
from tensorflow.keras.models import Model


def test_for_nan(fun2test):
    def wraper(*args, **keyargs):
        out = fun2test(*args, **keyargs)
        sum = np.sum(out)
        if np.isnan(sum):
            print(F"function, {fun2test}. return nan values.")
        return out
    return wraper


class Loss:
    def __init__(self, anchors, lambda_coord=5, lambda_noobj=1, iou_trehs=0.6):
        self.anchors = anchors
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.priors = self.make_priors()
        self.iou_trehs = iou_trehs

    def center_grid(self, input_tensor):
        '''
        build grid (xi, yi) for fix x_center, y_center in the yolo prediction

        parameters:
        -----------
        input_tensor: ndarray or tensorflow.python.framework.ops.EagerTensor
        return : ndarray shape (batch_size, wg, hg, 1, 1, n_boxs, 2)
        '''
        batch_size, wg, hg, n_boxs = input_tensor.shape[:4]
        xg = np.arange(wg)
        yg = np.arange(hg)
        grid = np.meshgrid(xg, yg)
        grid_x = np.reshape(grid[0], (wg, hg, 1))
        grid_y = np.reshape(grid[1], (wg, hg, 1))
        grid = np.stack((grid_x, grid_y), -1)
        #  tile
        return np.tile(grid, [batch_size, 1, 1, n_boxs, 1])

    def make_priors(self):
        '''
        make a priors  for fix the anchros predictions
        '''
        if len(self.anchors)/2 != len(self.anchors)//2:
            raise ValueError('anchors must be a par number')
        n_boxs2 = 0
        for i in self.anchors:
            n_boxs2 += 1
        n_boxs = n_boxs2//2
        return np.reshape(self.anchors, (1, 1, 1, n_boxs, 2))

    def min_max_boxes(self, tensor_xy, tensor_wh):
        '''
        compute the mins, maxs, form boxes
        '''

        half_wh = tensor_wh / 2.
        boxs_mins = tensor_xy - half_wh
        boxs_maxs = tensor_xy + half_wh

        return boxs_mins, boxs_maxs

    def compute_iou(self, tensor_xy_t, tensor_wh_t, tensor_xy_p, tensor_wh_p):
        '''
        compute the IoU with xy and wh given tensors

        parameters:
        -----------
        *_t : are true values
        *_p : are predicted values

        return : IoU each box tensor (all at once)
        '''
        # intersectio areas
        t_mins, t_maxs = self.min_max_boxes(tensor_xy_t, tensor_wh_t)
        p_mins, p_maxs = self.min_max_boxes(tensor_xy_p, tensor_wh_p)
        intersect_mins = tf.math.maximum(p_mins,  t_mins)
        intersect_maxes = tf.math.minimum(p_maxs, t_maxs)
        intersect_wh = tf.math.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

        # union areas
        true_areas = tensor_wh_t[..., 0] * tensor_wh_t[..., 1]
        pred_areas = tensor_wh_p[..., 0] * tensor_wh_p[..., 1]
        union_areas = pred_areas + true_areas - intersect_areas

        # compute IoU
        iou = tf.math.truediv(intersect_areas, union_areas)

        return iou

    @test_for_nan
    def xy_loss(self, true_xy, predict_xy, true_object):
        '''
        compute x, y cordinate loss like the paper said L2 norm
        '''
        square = tf.math.square(true_xy - predict_xy)
        masqued_square = square*true_object[..., None]
        loss = tf.math.reduce_sum(masqued_square, axis=(1, 2, 3, 4))
        return loss

    @test_for_nan
    def wh_loss(self, true_wh, predict_wh, true_object):
        '''
        compute L2 norm for w, h box cordinates
        '''
        squared_sqrt = tf.square(tf.sqrt(true_wh) - tf.sqrt(predict_wh))
        masqued_squared_sqrt = true_object[..., None]*squared_sqrt
        loss = tf.math.reduce_sum(masqued_squared_sqrt, axis=(1, 2, 3, 4))
        return loss

    @test_for_nan
    def object_loss(self, true_object, predict_object):
        '''
        Explanation:
            our net is a  bounding box predictors, based on anchors, so,
        our metric for object detection must be based on IoU metric. This is
        the nature of yolo bbox predictions based, predict a type based bb.
        '''
        # TODO:
        # build a no_obj, obj loss unifies

        loss = tf.keras.losses.binary_crossentropy(true_object[..., None],
                                                   predict_object[..., None])
        loss = true_object * loss
        loss = tf.math.reduce_sum(loss, axis=(1, 2, 3))
        # OLD (needs IoU)
        # square = tf.math.square(iou - predict_object)
        # masqued_square = true_object * square
        # loss = tf.math.reduce_sum(masqued_square, axis=(1, 2, 3))
        return loss

    @test_for_nan
    def no_obj_loss(self, iou, true_object, predict_object):
        '''
        explanation:
            punish lowest IoU,so wee filter Higths IoU, then wee filter again
         (< 0.6) where. That means, wee said that bbox are a good IoU are good
         candidates are noobj.
         Remember that (1 - true_object) is a mask where 0 are obj and 1 noobj.
        '''
        # definitions first
        higthst_iou = tf.math.reduce_max(iou, axis=-1)
        mask = (tf.cast(higthst_iou < self.iou_trehs,
                        dtype=tf.float32)[..., None]
                * (1 - true_object))  # noobj mask
        loss = mask * tf.keras.losses.binary_crossentropy(
                                                true_object[..., None],
                                                predict_object[..., None])
        loss = tf.reduce_sum(loss, axis=(1, 2, 3))
        # OLD :
        # compute loss
        # squared = tf.math.square(0 - predict_object)  # cero is noobj
        # masqued_square = mask * squared
        # loss = tf.math.reduce_sum(masqued_square, axis=[1, 2, 3])
        return loss

    @test_for_nan
    def class_loss(self, true_logits, predict_logits, true_object):
        # TODO:
        # Test that
        # sparse_softmax_cross_entropy_with_logits may work well
        # tf.reduce_sum(loss_class * class_mask) / (nb_class_box + 1e-6)
        # source: ( https://github.com/experiencor/keras-yolo2/blob/master/
        #           Yolo%20Step-by-Step.ipynb )
        loss = categorical_crossentropy(true_logits[..., None],
                                        predict_logits[..., None])
        loss = true_object[..., None] * loss   # masked
        loss = tf.math.reduce_sum(loss, axis=[1, 2, 3, 4])
        # OLD :
        # squared = tf.square(true_logits - predict_logits)
        # masqued_square = true_object[..., None] * squared
        # loss = tf.math.reduce_sum(masqued_square, axis=[1, 2, 3, 4])
        return loss

    def compute_loss(self,
                     predict_xy, predict_wh, predict_object, predict_logits,
                     true_xy, true_wh, true_object, true_logits):
        iou = self.compute_iou(true_xy, true_wh, predict_xy, predict_wh)
        loss = (
          + self.lambda_coord*self.xy_loss(true_xy, predict_xy, true_object)
          + self.lambda_coord*self.wh_loss(true_wh, predict_wh, true_object)
          + self.object_loss(true_object, predict_object)
          + self.lambda_noobj*self.no_obj_loss(iou, true_object, predict_object
                                               )
          + self.class_loss(true_logits, predict_logits, true_object))

        return loss

    def loss(self, y_true, y_pred):
        if y_pred.shape != y_true.shape:
            raise ValueError('imput shape and output shape must be equal')

        self.wh_grid = self.center_grid(y_pred)

        # predicted x, y, w, h, conf and class box cordinates adjustment
        predict_xy = tf.nn.sigmoid(y_pred[..., :2]) + self.wh_grid
        predict_wh = tf.math.exp(y_pred[..., 2:4]) * self.priors
        predict_object = tf.nn.sigmoid(y_pred[..., 4])
        # predict_logits = y_pred[..., 5:]

        # OLD
        predict_logits = tf.nn.softmax(y_pred[..., 5:])

        # predicted and true x, y, w, h box cordinates
        true_xy = y_true[..., 0:2]
        true_wh = y_true[..., 2:4]
        true_logits = y_true[..., 5:]
        true_object = y_true[..., 4]
        loss = self.compute_loss(predict_xy, predict_wh,
                                 predict_object, predict_logits,
                                 true_xy, true_wh,
                                 true_object, true_logits)
        return loss


class DarkNet19:
    '''DarkNet 19 for detection or clasification task

    Based on the YoloV2 article, this class can build Darknet or yolov2
    Fully Convolitional Neural Network.

    parameters:
    -----------

    classes : list
        list of classes you want to detect

    anchors : list
        list of anchors (anchors are described in the article Yolo9000)

    is_detector : bool
        * True if you want a make a detector
        * False for make a classifier NN

    in_shape : tuple
        tuple with 3 shapes of imputs images. (416, 416, 3) if for detection
        other whay (256, 256, 3)

    outpu_gridshape : tuple
        X, Y of the outpur in the last layer (batch_size, X, Y, ....)
    '''
    def __init__(self, n_classes: list, anchors: list, is_detector: bool,
                 in_shape=(416, 416, 3), output_gridshape=(13, 13)):
        self.output_gridshape = output_gridshape
        self.is_detector = is_detector
        self.anchors = anchors
        self.in_shape = in_shape
        self.n_classes = n_classes
        if self.is_detector is None:
            raise ValueError('Param "detector" must be either True or False.')
        if self.is_detector and not anchors:
            raise ValueError('If you wnat to detect, must give anchors.')
        if self.is_detector:
            self.gridx, self.gridy = self.output_gridshape[:2]
            self.n_anchors = len(self.anchors)//2
            self.input = Input(shape=self.in_shape)
        else:
            self.input = Input(shape=(224, 224, 3))

    # darknet convolutional layer
    def dark_conv(self, x, filters, kernel_shape, strides=1, padding='same',
                  bn=True, name=None):
        if name is None:
            x = Conv2D(filters, kernel_shape, strides=strides,
                       padding=padding, use_bias=(not bn))(x)
            if bn:
                x = BatchNormalization()(x)
                x = LeakyReLU(alpha=0.1)(x)
        else:
            conv_name = '{}_conv'.format(name)
            bn_name = '{}_batch_normalization'.format(name)
            leakyrelu_name = '{}_leakyrelu'.format(name)
            x = Conv2D(filters, kernel_shape, strides=strides,
                       padding=padding, use_bias=(not bn),
                       name=conv_name)(x)
            if bn:
                x = BatchNormalization(name=bn_name)(x)
                x = LeakyReLU(alpha=0.1, name=leakyrelu_name)(x)
        return x

    def passthrough_block(self, sc, x):
        sc = self.dark_conv(sc, filters=64, kernel_shape=(1, 1))
        sc = tf.nn.space_to_depth(sc, block_size=2)
        return concatenate([sc, x])

    def detector_output_block(self, x):
        x = Conv2D((4 + 1 + self.n_classes)*self.n_anchors, (1, 1))(x)
        x = Reshape((self.gridx, self.gridy, self.n_anchors,
                    4 + 1 + self.n_classes))(x)
        return x

    def clasifier_output_block(self, x):
        x = self.dark_conv(x, filters=self.n_classes, kernel_shape=(1, 1))
        x = GlobalAveragePooling2D()(x)
        return Softmax()(x)

    def build(self):
        x = self.dark_conv(self.input, filters=32, kernel_shape=(3, 3))
        x = MaxPooling2D((2, 2), strides=2)(x)
        #
        x = self.dark_conv(x, filters=64, kernel_shape=(3, 3))
        x = MaxPooling2D((2, 2), strides=2)(x)
        #
        x = self.dark_conv(x, filters=128, kernel_shape=(3, 3))
        x = self.dark_conv(x, filters=64, kernel_shape=(1, 1))
        x = self.dark_conv(x, filters=128, kernel_shape=(3, 3))
        x = MaxPooling2D((2, 2,), strides=2)(x)
        #
        x = self.dark_conv(x, filters=256, kernel_shape=(3, 3))
        x = self.dark_conv(x, filters=128, kernel_shape=(1, 1))
        x = self.dark_conv(x, filters=256, kernel_shape=(3, 3))
        x = MaxPooling2D((2, 2,), strides=2)(x)
        #
        x = self.dark_conv(x, filters=512, kernel_shape=(3, 3))
        x = self.dark_conv(x, filters=256, kernel_shape=(1, 1))
        x = self.dark_conv(x, filters=512, kernel_shape=(3, 3))
        x = self.dark_conv(x, filters=256, kernel_shape=(1, 1))
        x = self.dark_conv(x, filters=512, kernel_shape=(3, 3))
        if self.is_detector:
            passthrough = x
        x = MaxPooling2D((2, 2,), strides=2)(x)
        #
        x = self.dark_conv(x, filters=1024, kernel_shape=(3, 3))
        x = self.dark_conv(x, filters=512, kernel_shape=(1, 1))
        x = self.dark_conv(x, filters=1024, kernel_shape=(3, 3))
        x = self.dark_conv(x, filters=512, kernel_shape=(1, 1))
        x = self.dark_conv(x, filters=1024, kernel_shape=(3, 3))
        if self.is_detector:
            # explained in the article
            x = self.dark_conv(x, filters=1024, kernel_shape=(3, 3),
                               name='1_detec')
            x = self.dark_conv(x, filters=1024, kernel_shape=(3, 3),
                               name='2_dete')
            x = self.passthrough_block(passthrough, x)
            x = self.dark_conv(x, filters=1024, kernel_shape=(3, 3),
                               name='3_detec')
            out = self.detector_output_block(x)
        else:
            # Equal to the papper model
            out = self.clasifier_output_block(x)
        return Model(self.input, out)
