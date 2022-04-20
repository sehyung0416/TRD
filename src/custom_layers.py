import tensorflow as tf
import numpy as np


def conv_layer(input_conv, num_filters=32, filter_sz=3, stride=1, padding='same', layer_name='conv', relu=False,
               leaky_relu=False, b_norm=False, s_norm=False, layer_reuse=False):
    with tf.variable_scope(layer_name):
        conv = sn_conv3d(input_conv, out_dim=num_filters, k_size=filter_sz, strides=stride, padding=padding,
                         w_init=tf.glorot_uniform_initializer(), use_bias=True, spectral_normed=s_norm, name='conv',
                         layer_reuse=layer_reuse)
        if b_norm:
            if layer_reuse==False:
                conv = tf.layers.batch_normalization(conv, trainable=True)
            else:
                conv = tf.layers.batch_normalization(conv, trainable=True, reuse=tf.AUTO_REUSE)
        if relu:
            conv = tf.nn.relu(conv)       
        if leaky_relu:
            conv = tf.nn.leaky_relu(conv)
    return conv


def upconv_layer(input_conv, num_filters=32, filter_sz=3, stride=2, padding='same', layer_name='upconv',
                 relu=False, leaky_relu=False, b_norm=False, s_norm=False, layer_reuse=False):
    with tf.variable_scope(layer_name):
        upsample = tf.keras.layers.UpSampling3D(size=(stride, stride, stride))(input_conv)
        upconv = sn_conv3d(upsample, out_dim=num_filters, k_size=filter_sz, strides=1, padding=padding,
                           w_init=tf.glorot_uniform_initializer(), use_bias=True, spectral_normed=s_norm, name='conv',
                           layer_reuse=layer_reuse)
        if b_norm:
            if layer_reuse == False:
                upconv = tf.layers.batch_normalization(upconv, trainable=True)
            else:
                upconv = tf.layers.batch_normalization(upconv, trainable=True, reuse = tf.AUTO_REUSE)
        if relu:
            upconv = tf.nn.relu(upconv)
        if leaky_relu:
            upconv = tf.nn.leaky_relu(upconv)
    return upconv


def spectral_normed_weight(w, u=None, num_iters=1, update_collection=None, with_sigma=False):
    # For Power iteration method, usually num_iters = 1 will be enough

    w_shape = w.shape.as_list()
    w_new_shape = [np.prod(w_shape[:-1]), w_shape[-1]]
    w_reshaped = tf.reshape(w, w_new_shape, name='w_reshaped')
    
    if u is None:
        u = tf.get_variable("u_vec", [w_new_shape[0], 1], initializer=tf.glorot_uniform_initializer(), trainable=False)
    
    # power iteration
    u_ = u
    for _ in range(num_iters):
        # ( w_new_shape[1], w_new_shape[0] ) * ( w_new_shape[0], 1 ) -> ( w_new_shape[1], 1 )
        v_ = _l2normalize(tf.matmul(tf.transpose(w_reshaped), u_)) 
        # ( w_new_shape[0], w_new_shape[1] ) * ( w_new_shape[1], 1 ) -> ( w_new_shape[0], 1 )
        u_ = _l2normalize(tf.matmul(w_reshaped, v_))

    u_final = tf.identity(u_, name='u_final')
    v_final = tf.identity(v_, name='v_final')

    u_final = tf.stop_gradient(u_final)
    v_final = tf.stop_gradient(v_final)

    sigma = tf.matmul(tf.matmul(tf.transpose(u_final), w_reshaped), v_final, name="est_sigma")

    update_u_op = tf.assign(u, u_final)

    with tf.control_dependencies([update_u_op]):
        sigma = tf.identity(sigma)
        w_bar = tf.identity(w / sigma, 'w_bar')

    if with_sigma:
        return w_bar, sigma
    else:
        return w_bar


def _l2normalize(v, eps=1e-12):
    with tf.name_scope('l2normalize'):
        return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)


def sn_conv3d(inputs, out_dim, k_size, strides, padding="SAME", w_init=None, use_bias=True, spectral_normed=True,
              name="conv3d", layer_reuse=False):
    
    with tf.variable_scope(name, reuse=layer_reuse):
        w = tf.get_variable("w", shape=[k_size, k_size, k_size, inputs.get_shape()[-1], out_dim], dtype=tf.float32,
                            initializer=w_init)
        
        if spectral_normed:
            w = spectral_normed_weight(w)
        
        conv = tf.nn.conv3d(inputs, w, strides=[1, strides, strides, strides, 1], padding=padding.upper())
        
        if use_bias:
            biases = tf.get_variable("b", [out_dim], initializer=tf.constant_initializer(0.0))
            conv = tf.nn.bias_add(conv, biases, name="conv_add_b")
        
        return conv
