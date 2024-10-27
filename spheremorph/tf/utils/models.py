from os.path import exists

import tensorflow as tf
from spheremorph.tf.networks import SMAB


def spm_geom_model(model_weights_file=None):
    img_shape = (256, 512)
    pad_size = 16
    img_shape_padded = [x + 2 * pad_size for x in img_shape]

    net_factor = 7
    nf = 2 ** net_factor
    unet_struct = [[nf, ] * 5, [nf, ] * 7]

    num_outputs = 5
    losses = [dummy_loss] * num_outputs
    metrics = [None] * num_outputs
    metric_name = [None] * num_outputs
    img_shape_padded_ft = (*img_shape_padded, 3)
    # ==================== hyperparameters ====================

    model = SMAB(input_shape_ft=img_shape_padded_ft, nb_unet_features=unet_struct,
                 loss_fn=losses, metric_fn=metrics, metric_name=metric_name,
                 is_bidir=True, pad_size=pad_size, is_atlas_trainable=False,
                 is_jacobian=True, is_neg_jacobian_filter=True)

    if model_weights_file is not None:
        if exists(model_weights_file):
            model.load_weights(model_weights_file)
            print('Model weights loaded')
        else:
            raise FileNotFoundError(f"Model weights file {model_weights_file} not found")

    return model


def dummy_loss(*args):
    if len(args) == 1:
        y_pred = args[0]
    elif len(args) == 2:
        _, y_pred = args
    else:
        y_pred = 0
    return tf.reduce_mean(y_pred)
