"""
tensorflow/keras utilities for spheremorph
"""

import numpy as np
import tensorflow as tf


def pad_2d_image_spherically(img, pad_size=16, input_no_batch_dim=False, is_flow=False):
    """
    pad parameterized 2d image based on the spherical positions of its vertices
    img: image to pad, whose shape is [batch_size, H, W, ...] or [H, W] for a single image
    """
    is_2d = is_nd(img, 2)
    img = expand_batch_dims_with_cond(img, is_2d, input_no_batch_dim)

    if pad_size > 0:
        # pad the north pole on top
        top = img[:, 1:pad_size + 1, ...]  # get top pad without the first row (reflect)
        top = flip(top, axis=1)  # flip upside down
        top = roll(top, get_shape(top, 2) // 2, axis=2)  # circularly shift by pi

        # very important, if the image represents a flow field, then the sign of the padded region should be flipped
        if is_flow:
            top = -top

        # similarly for the south pole on bottom
        bot = img[:, -pad_size - 1:-1, ...]
        bot = flip(bot, axis=1)
        bot = roll(bot, get_shape(bot, 2) // 2, axis=2)
        if is_flow:
            bot = -bot

        # concatenate top and bottom before padding left and right
        img2 = concat((top, img, bot), axis=1)

        # pad left to right and right to left (wrap)
        left = img2[:, :, 0:pad_size, ...]
        right = img2[:, :, -pad_size:, ...]
        img3 = concat((right, img2, left), axis=2)
    else:
        img3 = img

    img3 = squeeze_with_cond(img3, is_2d, input_no_batch_dim)

    return img3


def unpad_2d_image(img, pad_size=0, input_no_batch_dim=False):
    """
    extract the original image from the padded image
    img: image to unpad, whose shape is [batch_size, H, W, ...]
    """
    is_2d = is_nd(img, 2)
    img = expand_batch_dims_with_cond(img, is_2d, input_no_batch_dim)

    if pad_size > 0:
        img = img[:, pad_size:-pad_size, pad_size:-pad_size, ...]

    img = squeeze_with_cond(img, is_2d, input_no_batch_dim)
    return img


# wrap conditional functions compatible with both numpy and tf only for padding and unpadding functions
def expand_batch_dims_with_cond(data, is_2d, input_no_batch_dim):
    if tf.is_tensor(data):
        is_2d = tf.cast(is_2d, tf.bool)
        input_no_batch_dim = tf.cast(input_no_batch_dim, tf.bool)
        cond = tf.logical_or(is_2d, input_no_batch_dim)
        data = tf.cond(cond, lambda: expand_batch_dims(data), lambda: data)
    else:
        if is_2d or input_no_batch_dim:
            data = expand_batch_dims(data)
    return data


def squeeze_with_cond(data, is_2d, input_no_batch_dim):
    if tf.is_tensor(data):
        is_2d = tf.cast(is_2d, tf.bool)
        input_no_batch_dim = tf.cast(input_no_batch_dim, tf.bool)
        cond = tf.logical_or(is_2d, input_no_batch_dim)
        data = tf.cond(cond, lambda: squeeze(data), lambda: data)
    else:
        if is_2d or input_no_batch_dim:
            data = squeeze(data)
    return data


# wrap some basic functions so they are compatible with both numpy and tf
def get_shape(data, axis=None):
    if tf.is_tensor(data):
        if axis is None:
            return tf.shape(data)
        else:
            return tf.shape(data)[axis]
    else:
        if axis is None:
            return data.shape
        else:
            return data.shape[axis]


def concat(data, axis):
    if tf.is_tensor(data[0]):
        return tf.concat(data, axis)
    else:
        return np.concatenate(data, axis)


def flip(data, axis):
    if tf.is_tensor(data):
        if not (isinstance(axis, list) or isinstance(axis, tuple)):
            axis = [axis]
        return tf.reverse(data, axis)
    else:
        return np.flip(data, axis)


def roll(data, shift, axis):
    if tf.is_tensor(data):
        return tf.roll(data, shift, axis)
    else:
        return np.roll(data, shift, axis)


def squeeze(data):
    if tf.is_tensor(data):
        return tf.squeeze(data)
    else:
        return np.squeeze(data)


def expand_batch_dims(data):
    if tf.is_tensor(data):
        return tf.expand_dims(data, 0)
    else:
        return data[np.newaxis, ...]


def is_nd(data, n):
    if tf.is_tensor(data):
        return tf.cond(tf.cast(tf.rank(data) == n, tf.bool),
                       lambda: tf.constant(True, dtype=tf.bool),
                       lambda: tf.constant(False, dtype=tf.bool))
    else:
        if data.ndim == n:
            return True
        else:
            return False


def to_tensor(data, dtype=tf.float32):
    if tf.is_tensor(data):
        if data.dtype is not dtype:
            data = tf.cast(data, dtype)
        return data
    else:
        return tf.convert_to_tensor(data, dtype)


def spherical_sin(H, W, eps=1e-3):
    # the sin of latitude assumes the first and last element hits the poles
    # i.e. sin(0) and sin(pi), so that it can be padded at the north
    # and south poles properly by "reflection"
    rg = tf.transpose(tf.range(0, H, dtype=tf.float32))
    rg = tf.math.divide_no_nan(rg, to_tensor(H - 1))
    rg = tf.math.multiply(rg, to_tensor(np.pi))
    sin_lat = tf.math.sin(rg)
    # sin_lat = tf.math.sin(to_tensor(np.arange(0, H).T / (H - 1) * np.pi))
    # remove singularity near the two poles by setting the minimum to (positive) eps
    sin_lat = tf.where(sin_lat >= eps, sin_lat, eps)
    # repeat on longitude, forming the sin matrix
    S = tf.expand_dims(sin_lat, 1) * tf.ones((1, W))
    return S


def jacobian_2d(x, det=False, is_use_tf_det=False, is_replace_nan=False):
    """
    this function compute jacobian or its determinant of a 2D vector field
    all lines are written in an explicit and tf compatible way so it can be used for loss evaluation
    i.e. it can be excuted in graph mode
    when used for loss evalution, do NOT use tf.linalg.det to compute the determinant as it will
    cause matrix "not invertible" error during backpropagation
    this function was adapted from nes.utils.gradient/jacobian and vxm.py.utils.jacobian_determinant
    :param x: input tensor of shape [B, H, W, 2]
    :param det: if True, return the determinant of the Jacobian
    :param is_use_tf_det: if True, use tf.linalg.det to compute the determinant
    :param is_replace_nan: if True, replace nan values with 1
    :return: Jacobian tensor of shape [B, H, W, 2, 2] or [B, H, W] if det=True
    """

    def gradient_2d(x):
        beg = (x[1, ...] - x[0, ...])[tf.newaxis, ...]
        mid = 0.5 * (x[2:, ...] - x[:-2, ...])
        end = (x[-1, ...] - x[-2, ...])[tf.newaxis, ...]
        dx = tf.concat((beg, mid, end), axis=0)

        beg = (x[:, 1, ...] - x[:, 0, ...])[:, tf.newaxis, ...]
        mid = 0.5 * (x[:, 2:, ...] - x[:, :-2, ...])
        end = (x[:, -1, ...] - x[:, -2, ...])[:, tf.newaxis, ...]
        dy = tf.concat((beg, mid, end), axis=1)

        return dx, dy

    shape = tf.shape(x)
    gridx = tf.range(shape[0], dtype=tf.float32)
    gridy = tf.range(shape[1], dtype=tf.float32)
    grid = tf.meshgrid(gridx, gridy, indexing='ij')
    x += tf.stack(grid, axis=-1)

    grad = gradient_2d(x)

    if det:
        if is_use_tf_det:
            # using tf.linalg.det will replace nan/inf with 0
            J = tf.stack(grad, axis=-1)
            J = tf.linalg.det(J)
        else:
            dfdx = grad[0]
            dfdy = grad[1]
            J = dfdx[..., 0] * dfdy[..., 1] - dfdy[..., 0] * dfdx[..., 1]
            if is_replace_nan:
                # we replace nan/inf with 1 if we want to use J for loss
                J = tf.where(tf.math.is_finite(J), J, tf.ones_like(J))
    else:
        J = tf.stack(grad, axis=-1)

    return J


def get_ndims(data):
    if tf.is_tensor(data):
        return tf.rank(data)
    else:
        return data.ndim


@tf.function
def gaussian_filter_2d(image, filter_shape=(3, 3), sigma=1.0):
    """Perform Gaussian blur on image(s).
    Args:
      image: Either a 2-D `Tensor` of shape `[height, width]`,
        a 3-D `Tensor` of shape `[height, width, channels]`,
        or a 4-D `Tensor` of shape `[batch_size, height, width, channels]`.
      filter_shape: An `integer` or `tuple`/`list` of 2 integers, specifying
        the height and width of the 2-D gaussian filter. Can be a single
        integer to specify the same value for all spatial dimensions.
      sigma: A `float` or `tuple`/`list` of 2 floats, specifying
        the standard deviation in x and y direction the 2-D gaussian filter.
        Can be a single float to specify the same value for all spatial
        dimensions.
    Returns:
      2-D, 3-D or 4-D `Tensor` of the same dtype as input.
    """
    with tf.name_scope("gaussian_filter_2d"):
        image = to_tensor(image)

        filter_shape = regulate_tuple(filter_shape, n=2, dtype=int, name='filter_shape')
        filter_shape = to_tensor(filter_shape, tf.int32)

        sigma = regulate_tuple(sigma, n=2, dtype=float, name='sigma')
        sigma = to_tensor(sigma)

        original_ndims = get_ndims(image)
        image = to_4D_image(image)

        channels = tf.shape(image)[3]

        kernel_x = compute_1d_gaussian_kernel(sigma[1], filter_shape[1])
        kernel_x = kernel_x[tf.newaxis, :]
        kernel_y = compute_1d_gaussian_kernel(sigma[0], filter_shape[0])
        kernel_y = kernel_y[:, tf.newaxis]
        kernel_2d = tf.matmul(kernel_y, kernel_x)
        kernel_2d = kernel_2d[:, :, tf.newaxis, tf.newaxis]
        kernel_2d = tf.tile(kernel_2d, [1, 1, channels, 1])

        output = tf.nn.depthwise_conv2d(image, kernel_2d, strides=(1, 1, 1, 1), padding='SAME')
        output = from_4D_image(output, original_ndims)
        return output


def regulate_tuple(x, n, dtype=float, name=None):
    if isinstance(x, (list, tuple)):
        if len(x) != n:
            raise ValueError(f'{name} should be a {dtype} or a tuple/list of {n} {dtype}')
        try:
            x = tuple(dtype(s) for s in x)
        except (ValueError, TypeError):
            raise ValueError(f'{name} should be a {dtype} or a tuple/list of {n} {dtype}')
    else:
        try:
            x = (dtype(x),) * n
        except (ValueError, TypeError):
            raise ValueError(f'{name} should be a {dtype} or a tuple/list of {n} {dtype}')
    return x


def to_4D_image(image):
    """Convert 2/3/4D image to 4D image.
    Args:
      image: 2/3/4D `Tensor`.

    Returns:
      4D `Tensor` with the same type.
    """
    with tf.control_dependencies(
            [
                tf.debugging.assert_rank_in(
                    image, [2, 3, 4], message="`image` must be 2/3/4D tensor"
                )
            ]
    ):
        ndims = image.get_shape().ndims
        if ndims is None:
            return _dynamic_to_4D_image(image)
        elif ndims == 2:
            return image[None, :, :, None]
        elif ndims == 3:
            return image[None, :, :, :]
        else:
            return image


def _dynamic_to_4D_image(image):
    shape = tf.shape(image)
    original_rank = tf.rank(image)
    # 4D image => [N, H, W, C] or [N, C, H, W]
    # 3D image => [1, H, W, C] or [1, C, H, W]
    # 2D image => [1, H, W, 1]
    left_pad = tf.cast(tf.less_equal(original_rank, 3), dtype=tf.int32)
    right_pad = tf.cast(tf.equal(original_rank, 2), dtype=tf.int32)
    new_shape = tf.concat(
        [
            tf.ones(shape=left_pad, dtype=tf.int32),
            shape,
            tf.ones(shape=right_pad, dtype=tf.int32),
        ],
        axis=0,
    )
    return tf.reshape(image, new_shape)


def from_4D_image(image, ndims):
    """Convert back to an image with `ndims` rank.

    Args:
      image: 4D `Tensor`.
      ndims: The original rank of the image.

    Returns:
      `ndims`-D `Tensor` with the same type.
    """
    with tf.control_dependencies(
            [tf.debugging.assert_rank(image, 4, message="`image` must be 4D tensor")]
    ):
        if isinstance(ndims, tf.Tensor):
            return _dynamic_from_4D_image(image, ndims)
        elif ndims == 2:
            return tf.squeeze(image, [0, 3])
        elif ndims == 3:
            return tf.squeeze(image, [0])
        else:
            return image


def _dynamic_from_4D_image(image, original_rank):
    shape = tf.shape(image)
    # 4D image <= [N, H, W, C] or [N, C, H, W]
    # 3D image <= [1, H, W, C] or [1, C, H, W]
    # 2D image <= [1, H, W, 1]
    begin = tf.cast(tf.less_equal(original_rank, 3), dtype=tf.int32)
    end = 4 - tf.cast(tf.equal(original_rank, 2), dtype=tf.int32)
    new_shape = shape[begin:end]
    return tf.reshape(image, new_shape)


def compute_1d_gaussian_kernel(sigma, filter_shape):
    """Compute 1D Gaussian kernel."""
    x = tf.cast(tf.range(-filter_shape // 2 + 1, filter_shape // 2 + 1), dtype=tf.float32)
    x = tf.nn.softmax(-tf.pow(x, 2.0) / (2.0 * tf.pow(sigma, 2.0)))
    return x
