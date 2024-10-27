"""
tensorflow/keras networks for spheremorph
"""

import keras.initializers as KI
import keras.layers as KL
import neurite as ne
import numpy as np
import tensorflow as tf
import voxelmorph as vxm

from . import layers


# Ordinate SphereMorph for pairwise registration
class SphereMorph(ne.modelio.LoadableModel):
    @ne.modelio.store_config_args
    def __init__(self,
                 input_shape_ft=None,
                 input_model=None,
                 nb_unet_features=None,
                 loss_fn=None,
                 metric_fn=None,
                 metric_name=None,
                 is_bidir=True,
                 pad_size=0,
                 int_steps=7,
                 int_method='ss',
                 interp_method='linear',
                 is_neg_jacobian_filter=True,
                 is_jacobian=True,
                 name='spheremorph',
                 **kwargs):
        """
        :param input_shape_ft: the shape of the input [H, W, F] without batch dimension
        :param input_model: if input a model, the input of this model is the output of input_model
        :param nb_unet_features: number of features in the unet, see VxmDense
        :param loss_fn: a tuple of loss functions to be used in loss end layers
        :param metric_fn: a tuple of metric functions to be used in loss end layers
        :param metric_name: a tuple of metric names to be used in loss end layers
        :param is_bidir: whether to use bidirectional flow
        :param pad_size: padding size
        :param int_steps: number of integration steps, default is 7
        :param int_method: type of integration method, default is ss
        :param interp_method: type of interpolation method, default is linear
        :param is_neg_jacobian_filter: whether to use negative jacobian filtering
        :param is_jacobian: whether to use jacobian loss
        :param name: name of the model
        :param kwargs: other arguments for VxmDense
        """
        # config inputs
        if input_model is None:
            if input_shape_ft is None:
                raise ValueError('input_shape_ft must be provided if input_model is None')
            subject = KL.Input(shape=input_shape_ft, name=f'{name}_source_input')
            atlas = KL.Input(shape=input_shape_ft, name=f'{name}_target_input')
            spm_inputs = [subject, atlas]
            model_inputs = spm_inputs
            num_inputs = len(spm_inputs)
        else:
            model_inputs = input_model.inputs
            spm_inputs = input_model.outputs
            [subject, atlas] = spm_inputs
            num_inputs = len(spm_inputs)
            input_shape_ft = []
            for m in range(num_inputs):
                input_shape_ft.append(spm_inputs[m].get_shape().as_list()[1:])

        if num_inputs != 2:
            raise ValueError('this network requires exactly two inputs')

        self.num_inputs = num_inputs

        # positive warp is subject -> atlas
        vxm_model_input = tf.keras.Model(inputs=model_inputs, outputs=spm_inputs)

        input_shape = spm_inputs[0].get_shape().as_list()[1:-1]
        ndims = len(input_shape)

        vxm_model = vxm.networks.VxmDense(
            input_shape, nb_unet_features=nb_unet_features,
            bidir=True, input_model=vxm_model_input, int_resolution=1, **kwargs)

        unet_output = vxm_model.references.unet_model.output
        svf = layers.SphericalConv2D(ndims, kernel_size=3, pad_size=pad_size,
                                     is_flow=True, padding='same',
                                     kernel_initializer=KI.RandomNormal(mean=0.0, stddev=1e-5),
                                     name=f'svf')(unet_output)
        pos_flow = vxm.layers.VecInt(method=int_method, name=f'pos_flow',
                                     int_steps=int_steps)(svf)

        neg_svf = ne.layers.Negate(name=f'neg_svf')(svf)
        neg_flow = vxm.layers.VecInt(method=int_method, name=f'neg_flow',
                                     int_steps=int_steps)(neg_svf)

        if is_neg_jacobian_filter:
            pos_flow = layers.NegativeJacobianFiltering2D(filter_shape=(2, 2), sigma=0.7,
                                                          pad_size=pad_size, name='pos_flow_njf')(pos_flow)
            neg_flow = layers.NegativeJacobianFiltering2D(filter_shape=(2, 2), sigma=0.7,
                                                          pad_size=pad_size, name='neg_flow_njf')(neg_flow)

        warped_subject = vxm.layers.SpatialTransformer(interp_method=interp_method, indexing='ij', fill_value=None,
                                                       name='warped_subject')([subject, pos_flow])

        warped_atlas = vxm.layers.SpatialTransformer(interp_method=interp_method, indexing='ij', fill_value=None,
                                                     name='warped_atlas')([atlas, neg_flow])

        loss_fn_atlas, loss_fn_warp, loss_fn_ms, \
            loss_fn_jacobian, loss_fn_subject = self.parse_loss_metric_function(loss_fn, is_bidir, is_jacobian)

        if metric_fn is not None:
            metric_fn_atlas, metric_fn_warp, metric_fn_ms, metric_fn_jacobian, \
                metric_fn_subject = self.parse_loss_metric_function(metric_fn, is_bidir, is_jacobian)
            mn_atlas, mn_warp, mn_ms, mn_jacobian, \
                mn_subject = self.parse_loss_metric_function(metric_name, is_bidir, is_jacobian)
        else:
            metric_fn_atlas = metric_fn_warp = metric_fn_ms = metric_fn_jacobian = metric_fn_subject = None
            mn_atlas = mn_warp = mn_ms = mn_jacobian = mn_subject = None

        atlas_loss = layers.create_loss_end([atlas, warped_subject], loss_fn_atlas, name=f'loss_atlas',
                                            metric_fn=metric_fn_atlas, metric_name=mn_atlas)

        warp_loss = layers.create_loss_end([pos_flow], loss_fn_warp, name=f'loss_warp',
                                           metric_fn=metric_fn_warp, metric_name=mn_warp)

        mean_stream = ne.layers.MeanStream(name='mean_stream')(pos_flow)
        ms_loss = layers.create_loss_end([mean_stream], loss_fn_ms, name='loss_mean_stream',
                                         metric_fn=metric_fn_ms, metric_name=mn_ms)

        if is_jacobian:
            jacobian_loss = layers.create_loss_end([pos_flow], loss_fn_jacobian, name='loss_jacobian',
                                                   metric_fn=metric_fn_jacobian, metric_name=mn_jacobian)
        else:
            jacobian_loss = None

        if is_bidir:
            subject_loss = layers.create_loss_end([subject, warped_atlas], loss_fn_subject, name=f'loss_subject',
                                                  metric_fn=metric_fn_subject, metric_name=mn_subject)
        else:
            subject_loss = None

        all_losses = [atlas_loss, warp_loss, ms_loss]
        if is_jacobian:
            all_losses.append(jacobian_loss)
        if is_bidir:
            all_losses.append(subject_loss)

        super().__init__(inputs=model_inputs, outputs=all_losses)

        # until this point, the model construction is done
        # below we construct a separate model for generating outputs without loss end points
        model_no_lep = tf.keras.Model(inputs=model_inputs,
                                      outputs=[warped_subject, warped_atlas, pos_flow, neg_flow])

        # cache pointers to important layers and tensors for future reference
        self.references = ne.modelio.LoadableModel.ReferenceContainer()
        self.references.vxm_model = vxm_model
        self.references.model_no_lep = model_no_lep
        self.references.warped_subject = warped_subject
        self.references.warped_atlas = warped_atlas
        self.references.pos_flow = pos_flow
        self.references.neg_flow = neg_flow
        self.references.model_warp_to_atlas = None
        self.references.model_warp_to_subject = None

    def get_model_outputs(self, src, batch_size=8):
        if not isinstance(src, (list, tuple)):
            src = [src]
        return self.references.model_no_lep.predict(src, batch_size=batch_size)

    def get_warped_subject(self, src, batch_size=8):
        return self.get_model_outputs(src, batch_size=batch_size)[0]

    def get_warped_atlas(self, src, batch_size=8):
        return self.get_model_outputs(src, batch_size=batch_size)[1]

    def get_positive_flow(self, src):
        return self.get_model_outputs(src)[2]

    def get_negative_flow(self, src):
        return self.get_model_outputs(src)[3]

    def warp(self, src, img, flow_type, interp_method='linear', fill_value=None, batch_size=8, **kwargs):
        # **kwargs appended last allows extra arguments passed to be compatible with JOSA warp API
        # e.g. the idx argument for JOSA, which is not used in spheremorph

        if flow_type in ['pos', 'to_atlas']:
            model_warp = self.references.model_warp_to_atlas
            flow = self.references.pos_flow
        elif flow_type in ['neg', 'to_subject']:
            model_warp = self.references.model_warp_to_subject
            flow = self.references.neg_flow

        # cache the model if not done before and the input image shape is the same
        # this avoid building the model every time the function is called
        if (model_warp is None) or (model_warp.inputs[1].shape[1:] != img.shape[1:]):
            img_input = tf.keras.Input(shape=img.shape[1:])
            st_layer = vxm.layers.SpatialTransformer(interp_method=interp_method, fill_value=fill_value)
            img_output = st_layer([img_input, flow])
            model_inputs = (*self.inputs, img_input)
            model_outputs = [img_output]

        if isinstance(src, (list, tuple)):
            data_in = [*src, img]
        else:
            data_in = [src, img]

        return model_warp.predict(data_in, batch_size=batch_size, verbose=0)

    def parse_loss_metric_function(self, fn, is_bidir, is_jac):
        fn_atlas, fn_warp, fn_ms = fn[0:3]
        fn_jac = fn[3] if is_jac else None
        fn_sub = fn[4] if is_bidir else None
        return fn_atlas, fn_warp, fn_ms, fn_jac, fn_sub


# SphereMorph with atlas building
class SphereMorphWithAtlasBuilding(ne.modelio.LoadableModel):
    @ne.modelio.store_config_args
    def __init__(self,
                 input_shape_ft=None,
                 input_model=None,
                 nb_unet_features=None,
                 loss_fn=None,
                 metric_fn=None,
                 metric_name=None,
                 is_bidir=True,
                 int_steps=7,
                 int_method='ss',
                 pad_size=0,
                 is_atlas_trainable=True,
                 is_jacobian=True,
                 is_neg_jacobian_filter=True,
                 name='smab',
                 **kwargs):
        """
        :param input_shape_ft: the shape of the input [H, W, F] without batch dimension
        :param input_model: if input a model, the input of this model is the output of input_model
        :param nb_unet_features: number of features in the unet, see VxmDense
        :param loss_fn: a tuple of loss functions to be used in loss end layers
        :param metric_fn: a tuple of metric functions to be used in loss end layers
        :param metric_name: a tuple of metric names to be used in loss end layers
        :param is_bidir: whether to use bidirectional flow
        :param pad_size: padding size
        :param int_steps: number of integration steps, default is 7
        :param int_method: type of integration method, default is ss
        :param interp_method: type of interpolation method, default is linear
        :param is_neg_jacobian_filter: whether to use negative jacobian filtering
        :param is_jacobian: whether to use jacobian loss
        :param name: name of the model
        :param kwargs: other arguments for VxmDense
        """

        # config inputs
        if input_model is None:
            if input_shape_ft is None:
                raise ValueError('input_shape_ft must be provided if input_model is None')
            this_input = KL.Input(shape=input_shape_ft, name='%s_input' % name)
            model_inputs = this_input
        else:
            model_inputs = input_model.inputs
            if len(input_model.outputs) == 1:
                this_input = input_model.outputs[0]
            else:
                this_input = KL.concatenate(input_model.outputs, name=f'{name}_input')
            input_shape_ft = this_input.shape[1:]

        mean_layer = layers.SphericalLocalParamWithInput(
            shape=input_shape_ft,
            mult=1.0,
            initializer=KI.RandomNormal(mean=0.0, stddev=1e-3),
            pad_size=pad_size,
            trainable=is_atlas_trainable,
            name=f'atlas_mean'
        )
        atlas_mean = mean_layer(this_input)

        spm_input_model = tf.keras.Model(inputs=model_inputs, outputs=[this_input, atlas_mean])
        spm_model = SphereMorph(input_model=spm_input_model,
                                nb_unet_features=nb_unet_features,
                                loss_fn=loss_fn,
                                metric_fn=metric_fn,
                                metric_name=metric_name,
                                is_bidir=is_bidir,
                                pad_size=pad_size,
                                int_steps=int_steps,
                                int_method=int_method,
                                is_neg_jacobian_filter=is_neg_jacobian_filter,
                                is_jacobian=is_jacobian,
                                name=name)

        super().__init__(inputs=model_inputs, outputs=spm_model.outputs)

        # cache pointers to important layers and tensors for future reference
        self.references = ne.modelio.LoadableModel.ReferenceContainer()
        self.references.mean_layer = mean_layer
        self.references.atlas_mean = atlas_mean
        self.references.vxm_model = spm_model.references.vxm_model
        self.references.model_no_lep = spm_model.references.model_no_lep
        self.references.warped_subject = spm_model.references.warped_subject
        self.references.warped_atlas = spm_model.references.warped_atlas
        self.references.pos_flow = spm_model.references.pos_flow
        self.references.neg_flow = spm_model.references.neg_flow
        self.references.model_warp_to_atlas = None
        self.references.model_warp_to_subject = None

    def set_atlas_mean(self, data):
        if data.shape[1]:
            data = np.reshape(data, data.shape[1:])
        self.references.mean_layer.set_weights([data])

    def get_atlas_mean(self):
        return self.references.mean_layer.get_weights()[0].squeeze()

    def get_model_outputs(self, src, batch_size=8):
        if not isinstance(src, (list, tuple)):
            src = [src]
        return self.references.model_no_lep.predict(src, batch_size=batch_size)

    def get_warped_subject(self, src, batch_size=8):
        return self.get_model_outputs(src, batch_size=batch_size)[0]

    def get_warped_atlas(self, src, batch_size=8):
        return self.get_model_outputs(src, batch_size=batch_size)[1]

    def get_positive_flow(self, src):
        return self.get_model_outputs(src)[2]

    def get_negative_flow(self, src):
        return self.get_model_outputs(src)[3]

    def warp(self, src, img, flow_type, interp_method='linear', fill_value=None, batch_size=8, **kwargs):
        # **kwargs appended last allows extra arguments passed to be compatible with JOSA warp API
        # e.g. the idx argument for JOSA, which is not used in spheremorph

        if flow_type in ['pos', 'to_atlas']:
            model_warp = self.references.model_warp_to_atlas
            flow = self.references.pos_flow
        elif flow_type in ['neg', 'to_subject']:
            model_warp = self.references.model_warp_to_subject
            flow = self.references.neg_flow

        # cache the model if not done before and the input image shape is the same
        # this avoid building the model every time the function is called
        if (model_warp is None) or (model_warp.inputs[1].shape[1:] != img.shape[1:]):
            img_input = tf.keras.Input(shape=img.shape[1:])
            st_layer = vxm.layers.SpatialTransformer(interp_method=interp_method, fill_value=fill_value)
            img_output = st_layer([img_input, flow])
            model_inputs = (*self.inputs, img_input)
            model_outputs = [img_output]
            model_warp = tf.keras.Model(inputs=model_inputs, outputs=model_outputs)

            if flow_type in ['pos', 'to_atlas']:
                self.references.model_warp_to_atlas = model_warp
            elif flow_type in ['neg', 'to_subject']:
                self.references.model_warp_to_subject = model_warp

        if isinstance(src, (list, tuple)):
            data_in = [*src, img]
        else:
            data_in = [src, img]

        return model_warp.predict(data_in, batch_size=batch_size, verbose=0)


# short name for convienience
SMAB = SphereMorphWithAtlasBuilding


# JOSA version 3, up-to-date with several improvements over original JOSA used in the papers
class JointSphereMorphWithAtlasBuildingV3(ne.modelio.LoadableModel):

    @ne.modelio.store_config_args
    def __init__(self,
                 input_shape_ft=None,
                 input_model=None,
                 nb_unet_features=None,
                 loss_fn=None,
                 metric_fn=None,
                 metric_name=None,
                 is_bidir=True,
                 int_steps=7,
                 int_method='ss',
                 pad_size=0,
                 pos_enc=0,
                 is_semi=False,
                 is_atlas_trainable=True,
                 input_type='float',
                 is_isw=True,
                 is_softmax=True,
                 is_jacobian=False,
                 is_neg_jacobian_filter=True,
                 name='josa',
                 **kwargs):
        """
        :param input_shape_ft: the shape of the input [H, W, F] without batch dimension
        :param input_model: if input a model, the input of this model is the output of input_model
        :param nb_unet_features: number of features in the U-Net, see VxmDense
        :param loss_fn: a tuple of loss functions to be used in the loss end layers
        :param metric_fn: a tuple of metric functions to be used in loss end layers
        :param metric_name:a tuple of metric names to be used in loss end layers
        :param is_bidir: whether to use bidirectional flow
        :param int_steps: integration steps of SVF
        :param pad_size: padding size
        :param pos_enc: whether to concatenate positional encoding into the input
        :param is_semi: whether to use semi-supervised learning, inputs are concatenated if false
        :param is_atlas_trainable: whether the atlas is trainable, can be an array of bools
        :param is_isw: whether use different deformation for each input (intra-subject warping)
        :param is_softmax: whether to use softmax for prob/onehot output
        :param is_jacobian: whether to use jacobian regularization
        :param is_neg_jacobian_filter: whether to filter out negative jacobian values
        :param name: name of the model
        :param kwargs: other arguments for VxmDense

        :return: a model with the scalar loss outputs
        """
        # config inputs
        if input_model is None:
            if input_shape_ft is None:
                raise ValueError('input_shape_ft must be provided if input_model is None')
            num_inputs = len(input_shape_ft)
            tisw_inputs = []
            for m in range(num_inputs):
                tisw_inputs.append(KL.Input(shape=input_shape_ft[m], name=f'{name}_input_{m + 1}'))
            model_inputs = tisw_inputs
        else:
            model_inputs = input_model.inputs
            tisw_inputs = input_model.outputs
            num_inputs = len(tisw_inputs)
            input_shape_ft = []
            for m in range(num_inputs):
                input_shape_ft.append(tisw_inputs[m].get_shape().as_list()[1:])

        if num_inputs < 2:
            raise ValueError('this network requires at least two inputs')

        self.num_inputs = num_inputs

        mean_layers = []
        atlas_means = []
        if not isinstance(is_atlas_trainable, (list, tuple)):
            is_atlas_trainable = [is_atlas_trainable] * num_inputs

        for m in range(num_inputs):
            mean_layer = layers.SphericalLocalParamWithInput(
                shape=input_shape_ft[m],
                mult=1.0,
                initializer=KI.RandomNormal(mean=0.0, stddev=1e-3),
                pad_size=pad_size,
                trainable=is_atlas_trainable[m],
                name=f'atlas_mean_{m + 1}'
            )
            mean_layers.append(mean_layer)
            atlas_mean = mean_layer(tisw_inputs[m])
            atlas_means.append(atlas_mean)

        self.is_semi = is_semi
        if is_semi:
            tisw_input_cat = tisw_inputs[0]
            atlas_mean_cat = atlas_means[0]
        else:
            tisw_input_cat = KL.concatenate(tisw_inputs, axis=-1, name='input_cat')
            atlas_mean_cat = KL.concatenate(atlas_means, axis=-1, name='atlas_mean_cat')

        input_cat_shape = tisw_input_cat.get_shape().as_list()[1:-1]
        vxm_model_input = tf.keras.Model(inputs=model_inputs,
                                         outputs=[tisw_input_cat, atlas_mean_cat])

        if pos_enc > 0:
            vxm_model = vxms.networks.VxmDenseWithPositionalEncoding(input_cat_shape,
                                                                     npos=pos_enc, pad_size=pad_size,
                                                                     nb_unet_features=nb_unet_features,
                                                                     bidir=True, input_model=vxm_model_input,
                                                                     int_resolution=1, **kwargs)
        else:
            vxm_model = vxm.networks.VxmDense(input_cat_shape, nb_unet_features=nb_unet_features,
                                              bidir=True, input_model=vxm_model_input, int_resolution=1, **kwargs)

        ndims = len(input_cat_shape)
        unet_output = vxm_model.references.unet_model.output

        # compute joint svf and flow from the unet output using SphericalConv2D
        svf_0 = layers.SphericalConv2D(ndims, kernel_size=3, pad_size=pad_size, is_flow=True, padding='same',
                                       kernel_initializer=KI.RandomNormal(mean=0.0, stddev=1e-5),
                                       name=f'svf_0')(unet_output)
        pos_flow_0 = vxm.layers.VecInt(method=int_method, name=f'pos_flow_0', int_steps=int_steps)(svf_0)

        neg_svf_0 = ne.layers.Negate(name=f'neg_svf_0')(svf_0)
        neg_flow_0 = vxm.layers.VecInt(method=int_method, name=f'neg_flow_0', int_steps=int_steps)(neg_svf_0)

        loss_fn_atlas, loss_fn_big_warp, loss_fn_small_warp, \
            loss_fn_ms, loss_fn_jacobian, loss_fn_subject = \
            self.parse_loss_metric_function(loss_fn, is_isw, is_jacobian)

        if metric_fn is not None:
            metric_fn_atlas, metric_fn_big_warp, metric_fn_small_warp, \
                metric_fn_ms, metric_fn_jacobian, metric_fn_subject = \
                self.parse_loss_metric_function(metric_fn, is_isw, is_jacobian)
            mn_atlas, mn_big_warp, mn_small_warp, mn_ms, mn_jacobian, mn_subject = \
                self.parse_loss_metric_function(metric_name, is_isw, is_jacobian)
        else:
            metric_fn_atlas = metric_fn_subject = [None] * num_inputs
            metric_fn_big_warp = metric_fn_small_warp = metric_fn_ms = metric_fn_jacobian = None
            mn_atlas = mn_subject = [None] * num_inputs
            mn_big_warp = mn_small_warp = mn_ms = mn_jacobian = None

        small_pos_flows = []
        pos_flow_compos = []
        warped_subjects = []
        atlas_softmaxes = []
        atlas_losses = []

        small_neg_flows = []
        neg_flow_compos = []
        warped_atlases = []
        warped_atlas_softmaxes = []
        subject_losses = []

        if not isinstance(input_type, (list, tuple)):
            input_type = [input_type] * num_inputs
        for m in range(num_inputs):
            assert input_type[m] in ('float', 'label', 'prob'), \
                f'input_type should be in (float, label, prob)'
        self.input_type = input_type

        interp_method = ['nearest' if self.input_type[m] == 'label' else 'linear' \
                         for m in range(num_inputs)]
        self.interp_method = interp_method

        is_softmax_atlas = [True if self.input_type[m] == 'prob' and is_softmax else False \
                            for m in range(num_inputs)]
        self.is_softmax_atlas = is_softmax_atlas

        for m in range(num_inputs):
            if is_isw:
                # individual svf
                svf = layers.SphericalConv2D(ndims, kernel_size=3, pad_size=pad_size, is_flow=True, padding='same',
                                             kernel_initializer=KI.RandomNormal(mean=0.0, stddev=1e-5),
                                             name=f'svf_{m + 1}')(unet_output)
                # individual flow (for regularization)
                flow = vxm.layers.VecInt(method=int_method, name=f'pos_flow_{m + 1}',
                                         int_steps=int_steps)(svf)

                # composite svf with svf_0 first
                pos_svf_compo = KL.add([svf_0, svf], name=f'pos_svf_compo_{m + 1}')

                # then integrate to get the compo flow
                pos_flow_compo = vxm.layers.VecInt(method=int_method, name=f'pos_flow_compo_{m + 1}',
                                                   int_steps=int_steps)(pos_svf_compo)

                if is_neg_jacobian_filter:
                    pos_flow_compo = layers.NegativeJacobianFiltering2D(filter_shape=(2, 2), sigma=0.7,
                                                                        pad_size=pad_size,
                                                                        name=f'pos_flow_compo_njf_{m + 1}')(
                        pos_flow_compo)

                # similarly for neg svf and flow
                neg_svf = ne.layers.Negate(name=f'neg_svf_{m + 1}')(svf)
                neg_flow = vxm.layers.VecInt(method=int_method, name=f'neg_flow_{m + 1}',
                                             int_steps=int_steps)(neg_svf)
                neg_svf_compo = KL.add([neg_svf_0, neg_svf], name=f'neg_svf_compo_{m + 1}')
                neg_flow_compo = vxm.layers.VecInt(method=int_method, name=f'neg_flow_compo_{m + 1}',
                                                   int_steps=int_steps)(neg_svf_compo)
                if is_neg_jacobian_filter:
                    neg_flow_compo = layers.NegativeJacobianFiltering2D(filter_shape=(2, 2), sigma=0.7,
                                                                        pad_size=pad_size,
                                                                        name=f'neg_flow_compo_njf_{m + 1}')(
                        neg_flow_compo)
            else:
                flow = neg_flow = None
                pos_flow_compo = pos_flow_0
                neg_flow_compo = neg_flow_0

            warped_subject = vxm.layers.SpatialTransformer(
                interp_method=interp_method[m], indexing='ij', fill_value=None,
                name=f'warped_subject_{m + 1}')([tisw_inputs[m], pos_flow_compo])

            if is_softmax_atlas[m]:
                atlas_softmax = KL.Activation(
                    'softmax', name=f'atlas_softmax_{m + 1}')(atlas_means[m])
            else:
                atlas_softmax = atlas_means[m]

            atlas_loss = layers.create_loss_end([atlas_softmax, warped_subject],
                                                loss_fn_atlas[m], name=f'loss_atlas_{m + 1}',
                                                metric_fn=metric_fn_atlas[m],
                                                metric_name=mn_atlas[m])
            small_pos_flows.append(flow)
            pos_flow_compos.append(pos_flow_compo)
            warped_subjects.append(warped_subject)
            atlas_softmaxes.append(atlas_softmax)
            atlas_losses.append(atlas_loss)

            warped_atlas = vxm.layers.SpatialTransformer(
                interp_method=interp_method[m], indexing='ij', fill_value=None,
                name=f'warped_atlas_{m + 1}')([atlas_means[m], neg_flow_compo])

            if is_softmax_atlas[m]:
                warped_atlas_softmax = KL.Activation(
                    'softmax', name=f'warped_atlas_softmax_{m + 1}')(warped_atlas)
            else:
                warped_atlas_softmax = warped_atlas

            if is_bidir:
                subject_loss = layers.create_loss_end([tisw_inputs[m], warped_atlas_softmax],
                                                      loss_fn_subject[m], name=f'loss_subject_{m + 1}',
                                                      metric_fn=metric_fn_subject[m],
                                                      metric_name=mn_subject[m])
            else:
                subject_loss = None

            small_neg_flows.append(neg_flow)
            neg_flow_compos.append(neg_flow_compo)
            warped_atlases.append(warped_atlas)
            warped_atlas_softmaxes.append(warped_atlas_softmax)
            subject_losses.append(subject_loss)

        big_warp_loss = layers.create_loss_end([pos_flow_0], loss_fn_big_warp, name='loss_big_warp',
                                               metric_fn=metric_fn_big_warp,
                                               metric_name=mn_big_warp)
        if is_isw:
            small_pos_flows_cat = KL.concatenate(small_pos_flows, axis=-1, name='small_pos_flows_cat')
            small_warp_loss = layers.create_loss_end([small_pos_flows_cat],
                                                     loss_fn_small_warp, name='loss_small_warps',
                                                     metric_fn=metric_fn_small_warp,
                                                     metric_name=mn_small_warp)
        else:
            small_warp_loss = None

        mean_stream = ne.layers.MeanStream(name='mean_stream')(pos_flow_0)
        ms_loss = layers.create_loss_end([mean_stream], loss_fn_ms, name='loss_mean_stream',
                                         metric_fn=metric_fn_ms, metric_name=mn_ms)

        if is_jacobian:
            # positive flows are used, supposed to compute jacobian for flows towards atlas
            # but due to how SpatialTransform layer works (interpolation at displaced locations),
            # i.e. flow direction is reversed. therefore jacobian is negate of what's expected
            # however, when evaluating jacobian loss, we take care of both expansion and shrinkage,
            # the flow direction should not make a difference
            pos_flow_stack = layers.Stack(axis=-1, name='pos_flow_stack')(pos_flow_compos)
            jacobian_loss = layers.create_loss_end([pos_flow_stack],
                                                   loss_fn_jacobian,
                                                   name='loss_jacobian',
                                                   metric_fn=metric_fn_jacobian,
                                                   metric_name=mn_jacobian)
        else:
            jacobian_loss = None

        all_losses = [*atlas_losses, big_warp_loss]
        if is_isw:
            all_losses.append(small_warp_loss)
        all_losses.append(ms_loss)
        if is_jacobian:
            all_losses.append(jacobian_loss)
        if is_bidir:
            all_losses = [*all_losses, *subject_losses]

        super().__init__(inputs=model_inputs, outputs=all_losses)

        # construct models for warp between subject and atlas without additional images
        model_input_to_atlas = tf.keras.Model(inputs=model_inputs, outputs=warped_subjects)

        # NOTE: for semi learning, theoretically only the first model input is required
        # however, because of how the atlas layer was constructed, other aux inputs are still needed
        # at inference, we can insert dummy input for aux data with the same shape as the atlas
        # see comments in the LocalParamWithInput layer for more details
        model_atlas_to_subject = tf.keras.Model(inputs=model_inputs, outputs=warped_atlas_softmaxes)

        self.references = ne.modelio.LoadableModel.ReferenceContainer()
        self.references.mean_layers = mean_layers
        self.references.atlas_means = atlas_means
        self.references.vxm_model = vxm_model
        self.references.pos_flow_0 = pos_flow_0
        self.references.neg_flow_0 = neg_flow_0
        self.references.is_isw = is_isw
        self.references.small_pos_flows = small_pos_flows
        self.references.pos_flow_compos = pos_flow_compos
        self.references.warped_subjects = warped_subjects
        self.references.atlas_softmaxes = atlas_softmaxes
        self.references.small_neg_flows = small_neg_flows
        self.references.neg_flow_compos = neg_flow_compos
        self.references.warped_atlases = warped_atlases
        self.references.warped_atlas_softmaxes = warped_atlas_softmaxes
        self.references.model_input_to_atlas = model_input_to_atlas
        self.references.model_atlas_to_subject = model_atlas_to_subject
        self.references.model_warp_to_atlas = [None] * self.num_inputs
        self.references.model_warp_to_subject = [None] * self.num_inputs
        self.references.model_big_warp_to_atlas = None
        self.references.model_small_warp_to_atlas = [None] * self.num_inputs
        self.references.model_big_warp_to_subject = None
        self.references.model_small_warp_to_subject = [None] * self.num_inputs

    def set_atlas_mean(self, data):
        for m in range(len(data)):
            d = data[m]
            if d.shape[1]:
                d = np.reshape(d, d.shape[1:])
            self.references.mean_layers[m].set_weights([d])

    def get_atlas_mean(self, is_cat=True):
        data = [x.get_weights()[0] for x in self.references.mean_layers]
        data = [x.astype(np.int32) if self.input_type[m] == 'label' else x for m, x in
                enumerate(data)]
        if is_cat:
            data = np.concatenate(data, axis=-1)
        else:
            data = [np.squeeze(x) for x in data]
        return data

    def prebuilt_model_predict(self, a_model, src, is_cat=True, batch_size=8):
        if not isinstance(src, (list, tuple)):
            src = [src]
        warped_src = a_model.predict(src, batch_size=batch_size)

        if is_cat:
            return np.concatenate(warped_src, axis=-1)
        else:
            return warped_src

    def get_warped_inputs(self, src, is_cat=True, batch_size=8):
        a_model = self.references.model_input_to_atlas
        return self.prebuilt_model_predict(a_model, src, is_cat, batch_size)

    def get_warped_atlas(self, src, is_cat=True, batch_size=8):
        a_model = self.references.model_atlas_to_subject
        return self.prebuilt_model_predict(a_model, src, is_cat, batch_size)

    def warp(self, src, img, flow_type, idx=0, interp_method='auto', fill_value=None, batch_size=8):
        assert idx >= 0 and idx < self.num_inputs, 'idx must be between 0 and the number of inputs'
        flow = self.parse_flow_reference(flow_type, idx)

        if flow_type in ['pos', 'to_atlas']:
            model_warp = self.references.model_warp_to_atlas[idx]
        elif flow_type in ['neg', 'to_subject']:
            model_warp = self.references.model_warp_to_subject[idx]
        elif flow_type == 'pos_big':
            model_warp = self.references.model_big_warp_to_atlas
        elif flow_type == 'pos_small':
            model_warp = self.references.model_small_warp_to_atlas[idx]
        elif flow_type == 'neg_big':
            model_warp = self.references.model_big_warp_to_subject
        elif flow_type == 'neg_small':
            model_warp = self.references.model_small_warp_to_subject[idx]

        # cache the model if not done before and the input image shape is the same
        # this avoid building the model every time the function is called
        if (model_warp is None) or (model_warp.inputs[1].shape[1:] != img.shape[1:]):
            img_input = tf.keras.Input(shape=img.shape[1:])
            if interp_method == 'auto':
                if self.input_type[idx] == 'float' or self.input_type[idx] == 'prob':
                    interp_method = 'linear'
                else:
                    interp_method = 'nearest'
            else:
                if not interp_method in ['linear', 'nearest']:
                    raise ValueError('interp_method must be one of: linear, nearest')

            st_layer = vxm.layers.SpatialTransformer(interp_method=interp_method,
                                                     fill_value=fill_value)
            img_output = st_layer([img_input, flow])
            if self.is_semi:
                model_inputs = (self.inputs[0], img_input)
            else:
                model_inputs = (*self.inputs, img_input)
            model_outputs = [img_output]
            model_warp = tf.keras.Model(inputs=model_inputs, outputs=model_outputs)

            if flow_type in ['pos', 'to_atlas']:
                self.references.model_warp_to_atlas[idx] = model_warp
            elif flow_type in ['neg', 'to_subject']:
                self.references.model_warp_to_subject[idx] = model_warp
            elif flow_type == 'pos_big':
                self.references.model_big_warp_to_atlas = model_warp
            elif flow_type == 'pos_small':
                self.references.model_small_warp_to_atlas[idx] = model_warp
            elif flow_type == 'neg_big':
                self.references.model_big_warp_to_subject = model_warp
            elif flow_type == 'neg_small':
                self.references.model_small_warp_to_subject[idx] = model_warp

        if isinstance(src, (list, tuple)):
            data_in = [*src, img]
        else:
            data_in = [src, img]

        return model_warp.predict(data_in, batch_size=batch_size, verbose=0)

    def get_flow(self, src, flow_type, idx):
        flow = self.parse_flow_reference(flow_type, idx)
        if self.is_semi:
            a_model = tf.keras.Model(inputs=self.inputs[0], outputs=flow)
        else:
            a_model = tf.keras.Model(inputs=self.inputs, outputs=flow)

        return self.prebuilt_model_predict(a_model, src, is_cat=False)

    def parse_flow_reference(self, flow_type, idx):
        if flow_type not in ['pos', 'to_atlas', 'neg', 'to_subject',
                             'pos_big', 'pos_small', 'neg_big', 'neg_small']:
            raise ValueError('flow_type must be one of: pos, to_atlas, neg, to_subject')

        if flow_type in ['pos', 'to_atlas']:
            flow = self.references.pos_flow_compos[idx]
        elif flow_type in ['neg', 'to_subject']:
            flow = self.references.neg_flow_compos[idx]
        elif flow_type == 'pos_big':
            flow = self.references.pos_flow_0
        elif flow_type == 'pos_small':
            flow = self.references.small_pos_flows[idx]
        elif flow_type == 'neg_big':
            flow = self.references.neg_flow_0
        elif flow_type == 'neg_small':
            flow = self.references.small_neg_flows[idx]
        return flow

    def parse_loss_metric_function(self, fn, is_isw, is_jacobian):
        num = self.num_inputs

        # first n are data losses in the atlas space
        fn_atlas = fn[0:num]
        c = num

        # must have a gradient loss on the deformation field
        fn_big_warp = fn[c]
        c += 1

        if is_isw:
            fn_small_warp = fn[c]
            c += 1
        else:
            fn_small_warp = None

        # must have a mean stream loss
        fn_ms = fn[c]
        c += 1

        if is_jacobian:
            fn_jacobian = fn[c]
            c += 1
        else:
            fn_jacobian = None

        # last n are data losses in the subject space
        fn_subject = fn[c:]

        return fn_atlas, fn_big_warp, fn_small_warp, fn_ms, fn_jacobian, fn_subject


# short name for convienience
JOSAV3 = JointSphereMorphWithAtlasBuildingV3


# duplicated from vxms, which is not installed with fs dev
class VxmDenseWithPositionalEncoding(ne.modelio.LoadableModel):
    """
    VoxelMorph network with positional encoding.
    """

    @ne.modelio.store_config_args
    def __init__(self, inshape,
                 input_model=None,
                 src_feats=1,
                 trg_feats=1,
                 npos=None,
                 pad_size=0,
                 bidir=False,
                 name='vxm_dense_pe',
                 **kwargs):

        if input_model is None:
            source = tf.keras.Input(shape=(*inshape, src_feats), name='%s_source_input' % name)
            target = tf.keras.Input(shape=(*inshape, trg_feats), name='%s_target_input' % name)
            input_model = tf.keras.Model(inputs=[source, target], outputs=[source, target])
        else:
            source, target = input_model.outputs[:2]

        if npos is not None and npos > 0:
            source_pe = layers.ConcatWithPositionalEncoding(npos, pad_size)(source)
            target_pe = layers.ConcatWithPositionalEncoding(npos, pad_size)(target)
            input_model = tf.keras.Model(inputs=input_model.inputs, outputs=[source_pe, target_pe])

        vxm_model = vxm.networks.VxmDense(inshape, input_model=input_model, bidir=bidir, **kwargs)

        pos_flow = vxm_model.references.pos_flow
        y_source = vxm.layers.SpatialTransformer(interp_method='linear',
                                                 name=f'pos_transformer')([source, pos_flow])
        outputs = [y_source]

        if bidir:
            neg_flow = vxm_model.references.neg_flow
            y_target = vxm.layers.SpatialTransformer(interp_method='linear',
                                                     name=f'neg_transformer')([target, neg_flow])
            outputs.append(y_target)

        super().__init__(inputs=input_model.inputs, outputs=outputs)

        # cache pointers to important layers and tensors for future reference
        self.references = ne.modelio.LoadableModel.ReferenceContainer()
        self.references.vxm_model = vxm_model
        self.references.unet_model = vxm_model.references.unet_model
        self.references.source = source
        self.references.target = target
        self.references.svf = vxm_model.references.svf
        self.references.preint_flow = vxm_model.references.preint_flow
        self.references.postint_flow = vxm_model.references.postint_flow
        self.references.pos_flow = vxm_model.references.pos_flow
        self.references.neg_flow = vxm_model.references.neg_flow
        self.references.y_source = y_source
        self.references.y_target = y_target
        self.references.hyp_input = vxm_model.references.hyp_input
