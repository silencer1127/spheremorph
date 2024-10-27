"""
tensorflow/keras layers for spheremorph
"""

import keras.backend as K
import keras.layers as KL
import neurite as ne
import numpy as np
import tensorflow as tf
from keras.layers import Layer

from .utils import unpad_2d_image, pad_2d_image_spherically, jacobian_2d, gaussian_filter_2d


class LossEndPoint(Layer):
    """
    Keras Layer: the loss end point that applies loss function to input tensors
    """

    def __init__(self, loss_fn=None, name=None, metric_fn=None, metric_name=None):
        self.loss_fn = loss_fn

        if name is None:
            name = 'lep'

        if isinstance(metric_fn, (list, tuple)):
            self.metric_fn = metric_fn
            self.metric_name = metric_name
        else:
            if metric_fn is not None:
                self.metric_fn = [metric_fn]
                self.metric_name = [metric_name]
            else:
                self.metric_fn = None
                self.metric_name = None

        super(LossEndPoint, self).__init__(name=name)

    def call(self, inputs, **kwargs):
        if self.loss_fn is not None:
            loss = self.loss_fn(*inputs)
        else:
            loss = 0

        if self.metric_fn is not None:
            for m, metric_f in enumerate(self.metric_fn):
                self.add_metric(metric_f(*inputs), name=self.metric_name[m])

        return K.mean(loss)

    def compute_output_shape(self, input_shape):
        return ()


class SphericalLocalParamWithInput(ne.layers.LocalParamWithInput):
    """
    LocalParamWithInput layer with spherical unpadding and padding
    """

    def __init__(self, shape, initializer='RandomNormal', mult=1.0, pad_size=0, **kwargs):
        self.pad_size = pad_size
        super(SphericalLocalParamWithInput, self).__init__(shape, initializer, mult, **kwargs)

    def call(self, x):
        xslice = K.batch_flatten(x)[:, 0:1]
        b = xslice * tf.zeros((1,)) + tf.ones((1,))
        img = K.flatten(self.kernel * self.biasmult)[tf.newaxis, ...]
        y = K.reshape(K.dot(b, img), [-1, *self.shape])
        if self.pad_size > 0:
            y = unpad_2d_image(y, self.pad_size)
            y = pad_2d_image_spherically(y, self.pad_size)
            y.set_shape(self.compute_output_shape(x.shape))
        return y


class ConcatWithPositionalEncoding(KL.Layer):
    """
    concatenate a set of positional encoding tensors to an existing one
    """

    def __init__(self, npos, pad_size=0, **kwargs):
        self.npos = npos
        self.pad_size = pad_size
        super(ConcatWithPositionalEncoding, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'npos': self.npos,
        })
        return config

    def build(self, inshape):  # create the pe tensor
        input_shape = inshape.as_list()[1:-1]
        if self.pad_size > 0:
            input_shape = [x - 2 * self.pad_size for x in input_shape]
        self.pe = _pos_encoding2D(input_shape, self.npos, self.pad_size)
        super().build(inshape)

    def call(self, x):  # concat of x and self.pe
        conc_fn = lambda x: tf.concat([x, self.pe], axis=-1)
        out = tf.map_fn(conc_fn, x)
        return out

    def compute_output_shape(self, input_shape, **kwargs):
        return input_shape[0:-1] + (input_shape[-1] + 2 * self.npos,)


class SphericalConv2D(KL.Conv2D):
    """
    Conv2D layer with spherical unpadding and padding
    """

    def __init__(self, filters, kernel_size, pad_size=0, is_flow=False, **kwargs):
        super(SphericalConv2D, self).__init__(filters, kernel_size, **kwargs)
        self.pad_size = pad_size
        self.is_flow = is_flow

    def call(self, x):
        y = super().call(x)
        if self.pad_size > 0:
            y = unpad_2d_image(y, self.pad_size)
            y = pad_2d_image_spherically(y, self.pad_size, is_flow=self.is_flow)
            y.set_shape(self.compute_output_shape(x.shape))
        return y


class NegativeJacobianFiltering2D(Layer):
    """
    Keras Layer: gaussian filtering only at locations with negative jacobian
    this layer does remove negative jacobian but introduce unnatural flows
    need further test
    """

    def __init__(self, filter_shape=(2, 2), sigma=0.7, pad_size=0, max_iter=3, **kwargs):
        super(NegativeJacobianFiltering2D, self).__init__(**kwargs)
        self.filter_shape = filter_shape
        self.sigma = sigma
        self.pad_size = pad_size
        self.max_iter = max_iter

    def build(self, input_shape):
        if input_shape[-1] != 2:
            raise ValueError(f'The last dimension should be 2, but got {input_shape[-1]}')
        # cache the negative jacobian locations so it will shared by cond and body
        H, W = input_shape[1:3]
        tf_zeros = tf.zeros((1, H, W, 1), dtype=tf.bool)
        self.J_neg = tf.Variable(tf_zeros, shape=(None, H, W, 1), trainable=False, dtype=tf.bool)
        super(NegativeJacobianFiltering2D, self).build(input_shape)

    def call(self, x):
        def _dilate_mask(z, f_size):
            filter = tf.ones((*f_size, 1), dtype=z.dtype)
            return tf.nn.dilation2d(z, filter, strides=[1, 1, 1, 1],
                                    padding='SAME', data_format='NHWC', dilations=[1, 1, 1, 1])

        def _update_J_neg(z):
            # compute jacobian determinant
            J = tf.map_fn(lambda w: jacobian_2d(w, det=True, is_replace_nan=True), z)
            # compute negative jac and expand the last dim because jacobian_2d reduced it
            J_neg = tf.expand_dims(tf.less(J, 0), -1)
            # dilate the mask according to the range of gaussian filter
            # f_size = [int(w * 1.5 // 2 * 2 + 1) for w in self.filter_shape]
            f_size = self.filter_shape
            self.J_neg = tf.cast(_dilate_mask(tf.cast(J_neg, tf.int32), f_size), tf.bool)

        def cond(z):
            num_neg = tf.reduce_sum(tf.cast(self.J_neg, tf.int32))
            return tf.greater(num_neg, 0)

        def body(z):
            # replace the negative jacobian locations (and their neighbors) with gaussian filtered values
            y = tf.where(self.J_neg, gaussian_filter_2d(z, self.filter_shape, self.sigma), z)
            if self.pad_size > 0:
                y = unpad_2d_image(y, self.pad_size)
                y = pad_2d_image_spherically(y, self.pad_size, is_flow=True)
                y.set_shape(z.shape)

            _update_J_neg(y)
            return (y,)

        # pre-update J_neg once at the beginning
        _update_J_neg(x)

        return tf.while_loop(cond, body, [x], maximum_iterations=self.max_iter)[0]

    def compute_output_shape(self, input_shape):
        return input_shape


def Stack(axis=-1, **kwargs):
    def stack_func(x):
        return K.stack(x, axis=axis)

    return KL.Lambda(stack_func, **kwargs)


# functional interface of LossEndPoint layer
def create_loss_end(x, loss_fn=None, **kwargs):
    if not isinstance(x, (list, tuple)):
        x = [x]
    return LossEndPoint(loss_fn, **kwargs)(x)


@tf.function
def _pos_encoding2D(inshape, npos, pad_size=0):
    pe = np.zeros(tuple(inshape) + (2 * npos,))

    for pno in range(npos):
        # x axis / horizontal
        wave_len = inshape[1] / (2 ** pno)
        res = 2 * np.pi / wave_len
        pex_1d = np.cos(np.arange(inshape[1]) * res)
        pex_2d = np.repeat(pex_1d[np.newaxis, ...], inshape[0], axis=0)
        pe[..., pno] = pex_2d

        # y axis / vertical
        wave_len = inshape[0] / (2 ** pno - 0.5)  # -0.5 to avoid same value on two poles
        res = 2 * np.pi / wave_len
        pey_1d = np.cos(np.arange(inshape[0]) * res)
        pey_2d = np.repeat(pey_1d[..., np.newaxis], inshape[1], axis=1)
        pe[..., npos + pno] = pey_2d

    if pad_size > 0:
        pe = pad_2d_image_spherically(pe, pad_size, input_no_batch_dim=True)

    return tf.convert_to_tensor(pe, tf.float32)
