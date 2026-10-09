"""
tensorflow/keras utilities for spheremorph
"""

import os
import random

import numpy as np
import scipy.io as sio
import tensorflow as tf
import voxelmorph as vxm
from packaging import version
from tqdm.auto import tqdm


def pad_2d_image_spherically(img, pad_size=16, input_no_batch_dim=False, is_flow=False):
    """
    pad parameterized 2d image based on the spherical positions of its vertices
    img: image to pad, whose shape is [batch_size, H, W, ...] or [H, W] for a single image
    """
    is_2d = is_nd(img, 2)
    img = expand_batch_dims_with_cond(img, is_2d, input_no_batch_dim)

    if pad_size > 0:
        # pad the north pole on top
        top = img[:, 1 : pad_size + 1, ...]  # get top pad without the first row (reflect)
        top = flip(top, axis=1)  # flip upside down
        top = roll(top, get_shape(top, 2) // 2, axis=2)  # circularly shift by pi

        # very important, if the image represents a flow field, then the sign of the padded region should be flipped
        if is_flow:
            top = -top

        # similarly for the south pole on bottom
        bot = img[:, -pad_size - 1 : -1, ...]
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
        return tf.cond(
            tf.cast(tf.rank(data) == n, tf.bool),
            lambda: tf.constant(True, dtype=tf.bool),
            lambda: tf.constant(False, dtype=tf.bool),
        )
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
    grid = tf.meshgrid(gridx, gridy, indexing="ij")
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

        filter_shape = regulate_tuple(filter_shape, n=2, dtype=int, name="filter_shape")
        filter_shape = to_tensor(filter_shape, tf.int32)

        sigma = regulate_tuple(sigma, n=2, dtype=float, name="sigma")
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

        output = tf.nn.depthwise_conv2d(image, kernel_2d, strides=(1, 1, 1, 1), padding="SAME")
        output = from_4D_image(output, original_ndims)
        return output


def regulate_tuple(x, n, dtype=float, name=None):
    if isinstance(x, (list, tuple)):
        if len(x) != n:
            raise ValueError(f"{name} should be a {dtype} or a tuple/list of {n} {dtype}")
        try:
            x = tuple(dtype(s) for s in x)
        except (ValueError, TypeError):
            raise ValueError(f"{name} should be a {dtype} or a tuple/list of {n} {dtype}")
    else:
        try:
            x = (dtype(x),) * n
        except (ValueError, TypeError):
            raise ValueError(f"{name} should be a {dtype} or a tuple/list of {n} {dtype}")
    return x


def to_4D_image(image):
    """Convert 2/3/4D image to 4D image.
    Args:
      image: 2/3/4D `Tensor`.

    Returns:
      4D `Tensor` with the same type.
    """
    with tf.control_dependencies(
        [tf.debugging.assert_rank_in(image, [2, 3, 4], message="`image` must be 2/3/4D tensor")]
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


def read_image_from_mat_file(file_name, var_name, pad_size=0, dtype=None):
    """
    read an image from a .mat file (MATLAB v7.3 via mat73 if available, otherwise scipy.io)
    the image is padded spherically and always returned with a feature dimension [H, W, C]
    """
    s = None
    try:
        import mat73

        s = mat73.loadmat(file_name)
    except (ImportError, TypeError):
        # mat73 is not installed, or the file is not a v7.3 (HDF5) .mat file
        pass

    if s is None:
        try:
            s = sio.loadmat(file_name)
        except NotImplementedError:
            raise TypeError("mat file not readable")

    img = s[var_name]
    # always extend to 3D + so to pad on the correct dimension
    img = img[np.newaxis, ...]
    img = pad_2d_image_spherically(img, pad_size)
    # squeeze back to original one (remove the batch dim)
    img = np.squeeze(img)
    # extend to 3D along the feature dim for any 2D image
    if img.ndim == 2:
        img = img[:, :, np.newaxis]
    if dtype is not None:
        if img.dtype != dtype:
            img = img.astype(dtype)
    return img


def load_mat_data(data_dirs, sub, lr, var_name, pad_size, is_data_cat=True, dtype="float32"):
    data = []
    # data_dirs is a list of lists
    if not isinstance(data_dirs, list):
        raise TypeError("data_dirs must be a list")

    for ft_dir in data_dirs:
        if not isinstance(ft_dir, list):
            file_name = os.path.join(ft_dir, sub + "." + lr + ".mat")
            data.append(read_image_from_mat_file(file_name, var_name, pad_size, dtype))
        else:
            data.append(load_mat_data(ft_dir, sub, lr, var_name, pad_size, True, dtype))

    if is_data_cat:
        # each element in data will be a 3D image [H, W, num_ft]
        return np.concatenate(data, -1)
    else:
        return data


def compute_1d_gaussian_kernel_new(sigma, win_size=None, scalex=1.0, max_win_size=None):
    """Compute 1D Gaussian kernel."""
    if win_size is None:
        win_size = tf.cast(tf.round(sigma / scalex * 3.0) * 2 + 1, tf.int32)
    if max_win_size is not None:
        win_size = tf.minimum(win_size, max_win_size)
    x = tf.cast(tf.range(-win_size // 2 + 1, win_size // 2 + 1), dtype=tf.float32)
    x = tf.nn.softmax(-tf.pow(tf.multiply(x, scalex), 2.0) / (2.0 * tf.pow(sigma, 2.0)))
    return x


class SphericalGaussianFilter:
    """
    create a spherical Gaussian filter that runs a straight conv in 1 dimension, but constructs
    and uses a block-circulant matrix in the other dimension to account for the variable pixel
    spacing in spherical images
    """

    def __init__(self, inshape, sigma, pad_size=16, eps=1e-3):
        sigma = float(sigma)
        self.pad_size = pad_size
        self.sigma = sigma

        H, W = inshape
        S = spherical_sin(H, W, eps)
        S = pad_2d_image_spherically(S, pad_size=pad_size, input_no_batch_dim=True)
        Hp = tf.shape(S)[0]
        Wp = tf.shape(S)[1]
        s = S[:, 0]  # step size for each y position, they are same across x

        # construct kernel for x and y directions, which will be used in depthwise convolution
        kernel = list()
        for m in tf.range(Hp):
            kernel.append(compute_1d_gaussian_kernel_new(sigma, win_size=Wp, scalex=s[m]))
        kernel = tf.stack(kernel)
        kernel = tf.divide(kernel, tf.reduce_sum(kernel, axis=1, keepdims=True))
        self.kernel_x = kernel
        # prepare a reshaped kernel for depthwise convolution (win_size, 1, Hp, 1)
        self.kernel_x_conv = tf.transpose(kernel[..., tf.newaxis, tf.newaxis], (1, 2, 0, 3))

        kernel = compute_1d_gaussian_kernel_new(sigma)
        kernel = tf.divide(kernel, tf.reduce_sum(kernel))
        self.kernel_y = kernel
        # prepare a reshaped kernel for depthwise convolution (win_size, 1, Wp, 1)
        self.kernel_y_conv = tf.tile(kernel[:, tf.newaxis, tf.newaxis, tf.newaxis], [1, 1, Wp, 1])

    # input image should be either 2D (H, W) or 3D (B, H, W)
    def filter_single_channel(self, img, is_padded=True):
        ndim = tf.get_static_value(tf.rank(img))
        # regulate image size (B, H, W, 1)
        if ndim == 2:
            img = img[tf.newaxis, ..., tf.newaxis]
        elif ndim == 3:
            img = img[..., tf.newaxis]
        else:
            raise ValueError("Input image should be either 2D (H, W) or 3D (B, H, W)")

        # pad image if not done yet (B, Hp, Wp, 1)
        if not is_padded:
            img = pad_2d_image_spherically(img, pad_size=self.pad_size)

        # put Hp dim last for depthwise conv (B, Wp, 1, Hp)
        # this does row-wise convolution (along x) with varying kernel shape
        img2 = tf.transpose(img, (0, 2, 3, 1))
        img3 = tf.nn.depthwise_conv2d(img2, self.kernel_x_conv, [1, 1, 1, 1], padding="SAME")

        # put Wp dim last for depthwise conv (B, Hp, 1, Wp)
        # this does column-wise convolution (along y) with fixed kernel shape
        img4 = tf.transpose(img3, (0, 3, 2, 1))
        img5 = tf.nn.depthwise_conv2d(img4, self.kernel_y_conv, [1, 1, 1, 1], padding="SAME")

        # reshape back to the original shape
        if ndim == 2:
            img6 = tf.squeeze(img5)
        else:
            img6 = tf.squeeze(img5, axis=2)

        return img6

    def filter(self, img, is_padded=True):
        ndim = tf.get_static_value(tf.rank(img))
        if ndim == 2 or ndim == 3:
            return self.filter_single_channel(img, is_padded=is_padded)
        elif ndim == 4:
            img2 = tf.unstack(img, axis=3)
            img3 = [
                self.filter_single_channel(img2[m], is_padded=is_padded) for m in range(len(img2))
            ]
            return tf.stack(img3, axis=3)
        else:
            raise ValueError(
                "Input image should be either 2D (H, W), 3D (B, H, W), or 4D (B, H, W, C)"
            )


def spherical_augment(
    img,
    types=[],
    noise_max=1.0,
    deform_max=2.0,
    deform_sigma=8.0,
    deform_smooth=None,
    random_mag=True,
    max_trans=16,
    zero_mean=False,
    pad_size=0,
    eps=1e-1,
    return_warp=False,
):
    """
    data augmentation for spherical images
    :param img: input image of shape [B, H, W, C]
    :param types: a list of augmentation types, current support: 'noise', 'deform', 'rotate'
    :param noise_max: maximum noise magnitude
    :param deform_max: maximum deformation magnitude
    :param deform_sigma: gaussian smoothing sigma for deformation field
    :param deform_smooth: smoothing method, None, or 'gaussian' or a SphericalGaussianFilter instance
    :param random_mag: is the magnitude of augmentation random between 0 and max
    :param max_trans: max translation (rotation) in pixels
    :param zero_mean: is zero mean the deformation field
    :param pad_size: pad size for distortion correction
    :param eps: the epsilon value for spherical sin function to prevent crazy division at poles
    :param return_warp: whether return the warp field
    :return: augmented image
    """
    if isinstance(img, (list, tuple)):
        was_list = True
    else:
        img = [img]
        was_list = False

    img0_shape = tf.shape(img[0])
    B, H, W = img0_shape[0], img0_shape[1], img0_shape[2]
    H_s, W_s = H - pad_size * 2, W - pad_size * 2
    S = spherical_sin(H_s, W_s, eps)
    S = tf.expand_dims(S, axis=0)
    S = pad_2d_image_spherically(S, pad_size=pad_size)

    warp = None

    def random_noise(shape, max_val):
        if random_mag:
            mag = tf.random.uniform([], 0, max_val)
        else:
            mag = max_val
        return mag * tf.random.normal(shape)

    def random_deform(shape):
        # generate random velocity field and convert to tensor
        warp = random_noise(shape, deform_max)
        # smooth velocity field and compensate for magnitude reduction
        # correct for spherical distortion using 1/sin on x axis (horizontal)
        # assume the second dim is y, the third dim is x (indexing is ij)
        warp_y = warp[..., 0]
        warp_x = tf.math.divide_no_nan(warp[..., 1], S)
        warp = tf.stack([warp_y, warp_x], axis=-1)

        if deform_smooth is not None:
            if deform_smooth == "gaussian":
                # determine filter size to be nearest even number to 3 sigma + 1
                # gaussian filter and multiply by filter size
                # the loss of magnitude is due to the average of iid gaussian, where the sigma is
                # reduced by the factor of sqrt(N), where N is the number of samples
                # N = f_size^2, so the sigma should be corrected by f_size
                f_size = tf.cast(tf.round(deform_sigma * 3 // 2) * 2 + 1, tf.int32)
                f_size_float = tf.cast(f_size, tf.float32)
                warp = (
                    gaussian_filter_2d(warp, filter_shape=(f_size, f_size), sigma=deform_sigma)
                    * f_size_float
                )
            elif type(deform_smooth).__name__ == "SphericalGaussianFilter":
                sgf = deform_smooth
                warp = deform_smooth.filter(warp)

                # count the number of non-zero elements in the kernel along x
                # (different for each row), so kx is a vector, expand to 4D tensor for broadcasting
                kx = tf.reduce_sum(tf.cast(tf.abs(sgf.kernel_x) > 0.0001, tf.float32), axis=1)
                kx = kx[tf.newaxis, :, tf.newaxis, tf.newaxis]

                # count the number of non-zero elements in the kernel along y
                # (same for each column), so ky is a scalar
                ky = tf.reduce_sum(tf.cast(tf.abs(sgf.kernel_y) > 0.0001, tf.float32), axis=0)

                # multiply by sqrt(kx) * sqrt(ky) to compensate for the loss of magnitude
                warp = warp * tf.sqrt(kx) * tf.sqrt(ky)
            else:
                raise ValueError(
                    'deform_smooth should be None, "gaussian", or SphericalGaussianFilter instance'
                )

        # integrate to get differomorphic deformation field
        fn = lambda x: vxm.utils.integrate_vec(x, nb_steps=7)
        warp = tf.map_fn(fn, warp)
        return warp

    # assume shape is shape of the padded image
    def random_rotate(shape, max_trans):
        # generate translation in each direction and convert to tensor
        shift = tf.random.uniform((2,), 0, max_trans)
        warp = tf.ones(shape) * shift

        # correct for spherical distortion using 1/sin on x axis (horizontal)
        # assume the second dim is y, the third dim is x (indexing is ij)
        warp_y = warp[..., 0]
        warp_x = tf.math.divide_no_nan(warp[..., 1], S)
        warp = tf.stack([warp_y, warp_x], axis=-1)

        return warp

    if "noise" in types:  # different random noise is applied to different images in the list
        img = [x + random_noise(tf.shape(x), noise_max) for x in img]

    if "deform" in types:  # same random deformation is applied to all images in the list
        img_shape = tf.shape(img[0])
        warp_shape = [img_shape[0], img_shape[1], img_shape[2], 2]
        warp = random_deform(warp_shape)
        if zero_mean:
            warp -= tf.reduce_mean(warp, axis=(1, 2), keepdims=True)
        fn = lambda x: vxm.utils.transform(x[0], x[1], "linear")
        img = [tf.map_fn(fn, (x, warp), fn_output_signature=tf.float32) for x in img]

    if "rotate" in types:
        img_shape = tf.shape(img[0])
        warp_shape = [img_shape[0], img_shape[1], img_shape[2], 2]
        warp = random_rotate(warp_shape, max_trans)
        fn = lambda x: vxm.utils.transform(x[0], x[1], "linear")
        img = [tf.map_fn(fn, (x, warp), fn_output_signature=tf.float32) for x in img]

    if not was_list:
        img = img[0]

    if return_warp:
        return img, warp
    else:
        return img


class SphereGenerator:
    def __init__(
        self,
        subject_list,
        feature_dir,
        batch_size=32,
        pad_size=16,
        file_ext="mat",
        var_name="img",
        hemi="L",
        is_feature_cat=True,
        is_mem_cache=False,
        is_cache_gen_style=True,
        aug_type=None,
        noise_lvl=0.0,
        deform_lvl=0.0,
        deform_sigma=1.0,
        deform_smooth=None,
        max_trans=0.0,
        random_mag=True,
        bi_dir=False,
        dtype="float32",
        is_verbose=True,
    ):
        self.subject_list = subject_list

        if not isinstance(feature_dir, list):
            feature_dir = [feature_dir]
        self.feature_dir = feature_dir

        # feature dir is a list, is_feature_cat indicates whether data are concatenated or kept
        # as a list
        self.is_feature_cat = is_feature_cat

        self.batch_size = batch_size
        self.pad_size = pad_size
        self.file_ext = file_ext
        self.var_name = var_name

        if file_ext == "mat":
            self.load_func = load_mat_data
        else:
            self.load_func = None
            raise ValueError("only mat file is currently supported")

        self.hemi = hemi
        self.is_mem_cache = is_mem_cache
        self.is_cache_gen_style = is_cache_gen_style

        if aug_type is not None:
            if isinstance(aug_type, (list, tuple)):
                self.aug_type = aug_type
            else:
                self.aug_type = [aug_type]
        else:
            self.aug_type = None

        if isinstance(noise_lvl, (list, tuple)):
            self.noise_lvl = noise_lvl
        else:
            self.noise_lvl = [noise_lvl]

        if isinstance(deform_lvl, (list, tuple)):
            self.deform_lvl = deform_lvl
        else:
            self.deform_lvl = [deform_lvl]

        if isinstance(deform_sigma, (list, tuple)):
            self.deform_sigma = deform_sigma
        else:
            self.deform_sigma = [deform_sigma]

        if deform_smooth == "gaussian" or type(deform_smooth).__name__ == "SphericalGaussianFilter":
            self.deform_smooth = deform_smooth
        else:
            self.deform_smooth = None

        if isinstance(max_trans, (list, tuple)):
            self.max_trans = max_trans
        else:
            self.max_trans = [max_trans]

        self.random_mag = random_mag
        self.bi_dir = bi_dir
        self.img_cached = None

        sub = self.subject_list[0]
        img = self.load_func(
            self.feature_dir,
            sub,
            self.hemi,
            self.var_name,
            self.pad_size,
            self.is_feature_cat,
            dtype,
        )

        if self.is_feature_cat:
            self.img_shape = img.shape[0:-1]
            self.num_ft = img.shape[-1]
            self.img_ft_shape = (*self.img_shape, self.num_ft)
            self.zero_img = np.zeros((self.batch_size, *self.img_ft_shape), dtype=dtype)

            self.img_ndim = len(self.img_shape)
            self.grad_shape = (*self.img_shape, self.img_ndim)
            self.zero_grad_batch = np.zeros((self.batch_size, *self.grad_shape), dtype=dtype)
            self.zero_grad = np.zeros(self.grad_shape, dtype=dtype)

            if self.is_mem_cache:
                if is_verbose:
                    print("cache data into memory")
                self.img_cached = np.zeros(
                    (len(self.subject_list), *self.img_shape, self.num_ft), dtype=dtype
                )
                for m, sub in tqdm(
                    enumerate(self.subject_list),
                    total=len(self.subject_list),
                    disable=(not is_verbose),
                ):
                    self.img_cached[m, :, :, :] = self.load_func(
                        self.feature_dir, sub, self.hemi, self.var_name, self.pad_size, True, dtype
                    )
        else:
            num_img = len(img)
            self.img_shape = [x.shape[0:-1] for x in img]
            self.num_ft = [x.shape[-1] for x in img]
            self.img_ft_shape = [(*self.img_shape[m], self.num_ft[m]) for m in range(num_img)]

            self.img_ndim = [len(x) for x in self.img_shape]
            self.grad_shape = [(*self.img_shape[m], self.img_ndim[m]) for m in range(num_img)]

            if self.is_mem_cache:
                if is_verbose:
                    print("cache data into memory")
                self.img_cached = []
                if self.is_cache_gen_style:
                    for m, sub in tqdm(
                        enumerate(self.subject_list),
                        total=len(self.subject_list),
                        disable=(not is_verbose),
                    ):
                        self.img_cached.append(
                            self.load_func(
                                self.feature_dir,
                                sub,
                                self.hemi,
                                self.var_name,
                                self.pad_size,
                                False,
                                dtype,
                            )
                        )
                else:
                    for m, ft_d in enumerate(self.feature_dir):
                        data_temp = np.zeros(
                            (len(self.subject_list), *self.img_ft_shape[m]), dtype=dtype
                        )
                        for n, sub in tqdm(
                            enumerate(self.subject_list),
                            total=len(self.subject_list),
                            disable=(not is_verbose),
                        ):
                            data_temp[n, :, :, :] = self.load_func(
                                ft_d, sub, self.hemi, self.var_name, self.pad_size, True, dtype
                            )
                        self.img_cached.append(data_temp)

    def data_augment(self, img):
        if self.aug_type is not None:
            if isinstance(img, (list, tuple)):
                was_list = True
            else:
                img = [img]
                was_list = False

            if "noise" in self.aug_type:
                if len(self.noise_lvl) == 1:  # allow same noise for image modalities (rarely)
                    img = spherical_augment(
                        img,
                        types=["noise"],
                        noise_max=self.noise_lvl[0],
                        random_mag=self.random_mag,
                    )
                else:  # allow different noise for each image modality (commonly)
                    img = [
                        spherical_augment(
                            x,
                            types=["noise"],
                            noise_max=self.noise_lvl[m],
                            random_mag=self.random_mag,
                        )
                        for m, x in enumerate(img)
                    ]

            if "deform" in self.aug_type:
                # allow same deformation for image modalities (commonly)
                if len(self.deform_lvl) == 1:
                    img = spherical_augment(
                        img,
                        types=["deform"],
                        deform_max=self.deform_lvl[0],
                        deform_sigma=self.deform_sigma[0],
                        pad_size=self.pad_size,
                        deform_smooth=self.deform_smooth,
                        random_mag=self.random_mag,
                    )
                else:  # allow different deformations for each image modality (rarely)
                    img = [
                        spherical_augment(
                            x,
                            types=["deform"],
                            deform_max=self.deform_lvl[m],
                            deform_sigma=self.deform_sigma[m],
                            pad_size=self.pad_size,
                            deform_smooth=self.deform_smooth,
                            random_mag=self.random_mag,
                        )
                        for m, x in enumerate(img)
                    ]

            # always apply the same rotation for all image modalities
            if "rotate" in self.aug_type:
                img = spherical_augment(img, types=["rotate"], max_trans=self.max_trans)

            if not was_list:
                img = img[0]

        return img

    def get_single_input(self):
        img = next(self.gen_single_image())
        if self.is_feature_cat:
            img = img[np.newaxis, ...]
        else:
            img = [x[np.newaxis, ...] for x in img]
        img = self.data_augment(img)
        return img

    def get_single_image(self, squeeze=True):
        img = self.get_single_input()
        if squeeze:
            if self.is_feature_cat:
                return np.squeeze(img)
            else:
                return tuple([np.squeeze(x) for x in img])
        else:
            return img

    def gen_single_image(self):
        while True:
            idx = random.choices(range(len(self.subject_list)), k=1)[0]
            sub = self.subject_list[idx]

            if self.is_feature_cat:
                if self.is_mem_cache:
                    img = self.img_cached[idx, :, :, :]
                else:
                    img = self.load_func(
                        self.feature_dir, sub, self.hemi, self.var_name, self.pad_size, True
                    )
                yield img
            else:
                if self.is_mem_cache:
                    if self.is_cache_gen_style:
                        img = self.img_cached[idx]
                    else:
                        img = [x[idx, ...] for x in self.img_cached]
                else:
                    img = self.load_func(
                        self.feature_dir, sub, self.hemi, self.var_name, self.pad_size, False
                    )
                yield tuple(img)

    def get_single_pair_input(self):
        img_move, img_fixed = next(self.gen_single_pair())
        img_move = self.data_augment(img_move[np.newaxis, ...])
        img_fixed = self.data_augment(img_fixed[np.newaxis, ...])
        return img_move, img_fixed

    def get_single_pair(self, squeeze=True):
        img = self.get_single_pair_input()
        if squeeze:
            return np.squeeze(img[0]), np.squeeze(img[1])
        else:
            return img[0], img[1]

    def gen_single_pair(self):
        while True:
            idx1, idx2 = random.choices(range(len(self.subject_list)), k=2)
            sub1 = self.subject_list[idx1]
            sub2 = self.subject_list[idx2]

            if self.is_feature_cat:
                if self.is_mem_cache:
                    img_move = self.img_cached[idx1, :, :, :]
                    img_fixed = self.img_cached[idx2, :, :, :]
                else:
                    img_move = self.load_func(
                        self.feature_dir, sub1, self.hemi, self.var_name, self.pad_size, True
                    )
                    img_fixed = self.load_func(
                        self.feature_dir, sub2, self.hemi, self.var_name, self.pad_size, True
                    )
            else:
                raise NotImplementedError(
                    "generate image pairs for subject with separate features is not supported"
                )

            yield (img_move, img_fixed)

    # tf dataset generator with end point loss functions, i.e. outputs are scalars (zeros)
    def gen_tf_data_lep(
        self,
        mode="single",
        num_of_outputs=3,
        batch_parallel_size=tf.data.AUTOTUNE,
        prefetch_size=tf.data.AUTOTUNE,
        shuffle_buffer=0,
    ):
        """
        :param shuffle_buffer: if > 0, shuffle the cached subjects with this buffer size every
            epoch (only used when is_mem_cache=True and is_cache_gen_style=False, where subjects
            are otherwise always drawn in the same order); 0 keeps the original behavior
        """

        assert mode in ["single", "pair"], "mode should be either single or pair"

        def gen_loss_end_layer_io(*img_move):
            return img_move, ((np.zeros([1], dtype=np.float32),) * num_of_outputs)

        def augment_map_func(*img_move):
            return self.data_augment(img_move)

        if mode == "single":
            if self.is_cache_gen_style:
                if self.is_feature_cat:
                    output_sig = [tf.TensorSpec(shape=self.img_ft_shape, dtype=tf.float32)]
                else:
                    output_sig = []
                    for shape in self.img_ft_shape:
                        output_sig.append(tf.TensorSpec(shape=shape, dtype=tf.float32))
                output_sig = tuple(output_sig)

                dataset = tf.data.Dataset.from_generator(
                    self.gen_single_image, output_signature=output_sig
                )
            else:
                if self.is_feature_cat:
                    dataset = tf.data.Dataset.from_tensor_slices(self.img_cached)
                else:
                    dataset = tf.data.Dataset.from_tensor_slices(tuple(self.img_cached))
                if shuffle_buffer > 0:
                    dataset = dataset.shuffle(shuffle_buffer, reshuffle_each_iteration=True)
        elif mode == "pair":
            if not self.is_feature_cat:
                raise ValueError("pair mode is only supported for concatenated features")
            # is_feature_cat must be True for this mode
            # there is no is_cache_gen_style option for this mode
            output_sig = (
                tf.TensorSpec(shape=self.img_ft_shape, dtype=tf.float32),
                tf.TensorSpec(shape=self.img_ft_shape, dtype=tf.float32),
            )
            dataset = tf.data.Dataset.from_generator(
                self.gen_single_pair, output_signature=output_sig
            )

        dataset = dataset.repeat()
        if version.parse(tf.__version__) >= version.parse("2.5.0"):
            dataset = dataset.batch(
                self.batch_size, num_parallel_calls=batch_parallel_size, deterministic=False
            )
        else:
            dataset = dataset.batch(self.batch_size)

        dataset = dataset.map(augment_map_func, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.map(gen_loss_end_layer_io, num_parallel_calls=tf.data.AUTOTUNE)

        if prefetch_size != 0:
            dataset = dataset.prefetch(prefetch_size)

        return dataset
