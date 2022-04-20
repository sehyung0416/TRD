import tensorflow as tf
import custom_layers


class RegNet:
    def __init__(self, args):
        self.batch_size = args.batch_size
        self.nf = args.nf
        self.fz = args.fz
        self.max_trs = args.max_trs
        self.max_angle = args.max_angle
        self.max_disp = args.max_disp
        
    def __call__(self, z_view, y_view, x_view, tranformer, name='RegNet'):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            # z view features
            z_feat = custom_layers.conv_layer(z_view, num_filters=self.nf * 1, filter_sz=self.fz, stride=1,
                                              padding='same', layer_name='feat1', leaky_relu=True, s_norm=True)
            z_feat = custom_layers.conv_layer(z_feat, num_filters=self.nf * 2, filter_sz=self.fz, stride=2,
                                              padding='same', layer_name='feat2', leaky_relu=True, s_norm=True)
            z_feat = custom_layers.conv_layer(z_feat, num_filters=self.nf * 4, filter_sz=self.fz, stride=2,
                                              padding='same', layer_name='feat3', leaky_relu=True, s_norm=True)
            z_feat = custom_layers.conv_layer(z_feat, num_filters=self.nf * 8, filter_sz=self.fz, stride=2,
                                              padding='same', layer_name='feat4', leaky_relu=True, s_norm=True)
    
            # y view features
            y_feat = custom_layers.conv_layer(y_view, num_filters=self.nf * 1, filter_sz=self.fz, stride=1,
                                              padding='same', layer_name='feat1', leaky_relu=True, s_norm=True)
            y_feat = custom_layers.conv_layer(y_feat, num_filters=self.nf * 2, filter_sz=self.fz, stride=2,
                                              padding='same', layer_name='feat2', leaky_relu=True, s_norm=True)
            y_feat = custom_layers.conv_layer(y_feat, num_filters=self.nf * 4, filter_sz=self.fz, stride=2,
                                              padding='same', layer_name='feat3', leaky_relu=True, s_norm=True)
            y_feat = custom_layers.conv_layer(y_feat, num_filters=self.nf * 8, filter_sz=self.fz, stride=2,
                                              padding='same', layer_name='feat4', leaky_relu=True, s_norm=True)
            
            # x view features
            x_feat = custom_layers.conv_layer(x_view, num_filters=self.nf * 1, filter_sz=self.fz, stride=1,
                                              padding='same', layer_name='feat1', leaky_relu=True, s_norm=True)
            x_feat = custom_layers.conv_layer(x_feat, num_filters=self.nf * 2, filter_sz=self.fz, stride=2,
                                              padding='same', layer_name='feat2', leaky_relu=True, s_norm=True)
            x_feat = custom_layers.conv_layer(x_feat, num_filters=self.nf * 4, filter_sz=self.fz, stride=2,
                                              padding='same', layer_name='feat3', leaky_relu=True, s_norm=True)
            x_feat = custom_layers.conv_layer(x_feat, num_filters=self.nf * 8, filter_sz=self.fz, stride=2,
                                              padding='same', layer_name='feat4', leaky_relu=True, s_norm=True)
            
            # rigid parameter estimation: y to z
            zy_conv = custom_layers.conv_layer(tf.concat([z_feat, y_feat], 4), num_filters=self.nf * 16,
                                               filter_sz=self.fz, stride=2, padding='same', layer_name='y_down1',
                                               leaky_relu=True, s_norm=True)
            zy_conv = custom_layers.conv_layer(zy_conv, num_filters=self.nf * 16, filter_sz=self.fz, stride=2,
                                               padding='same', layer_name='y_down2', leaky_relu=True, s_norm=True)
            zy_conv = tf.layers.flatten(zy_conv)
            y_rigid_params = tf.layers.dense(zy_conv, 6, name='y_dense')
            y_rigid_params = tf.nn.tanh(y_rigid_params)
            
            # rigid parameter estimation: x to z
            zx_conv = custom_layers.conv_layer(tf.concat([z_feat, x_feat], 4), num_filters=self.nf * 16,
                                               filter_sz=self.fz, stride=2, padding='same', layer_name='x_down1',
                                               leaky_relu=True, s_norm=True)
            zx_conv = custom_layers.conv_layer(zx_conv, num_filters=self.nf * 16, filter_sz=self.fz, stride=2,
                                               padding='same', layer_name='x_down2', leaky_relu=True, s_norm=True)
            zx_conv = tf.layers.flatten(zx_conv)
            x_rigid_params = tf.layers.dense(zx_conv, 6, name='x_dense')
            x_rigid_params = tf.nn.tanh(x_rigid_params) 
            
            # feature warping
            y_motion_fields = tranformer._rigid_to_dense(y_view, y_rigid_params, self.max_trs, self.max_angle)
            x_motion_fields = tranformer._rigid_to_dense(x_view, x_rigid_params, self.max_trs, self.max_angle)
            y_feat_warp = tranformer._feat_warp(y_feat, y_motion_fields, 8.0)
            x_feat_warp = tranformer._feat_warp(x_feat, x_motion_fields, 8.0)
            
            # dense match networks
            y_decode = custom_layers.conv_layer(tf.concat([z_feat, y_feat_warp], 4), num_filters=self.nf * 8,
                                                filter_sz=self.fz, stride=1, padding='same', layer_name='y_decode1',
                                                leaky_relu=True, s_norm=True)
            y_decode = custom_layers.upconv_layer(y_decode, num_filters=self.nf * 4, filter_sz=self.fz, stride=2,
                                                  padding='same', layer_name='y_upconv1', leaky_relu=True, s_norm=True)
            y_decode = custom_layers.upconv_layer(y_decode, num_filters=self.nf * 2, filter_sz=self.fz, stride=2,
                                                  padding='same', layer_name='y_upconv2', leaky_relu=True, s_norm=True)
            y_decode = custom_layers.upconv_layer(y_decode, num_filters=self.nf * 1, filter_sz=self.fz, stride=2,
                                                  padding='same', layer_name='y_upconv3', leaky_relu=True, s_norm=True)
            pred_y_dense = custom_layers.conv_layer(y_decode, num_filters=3, filter_sz=1, stride=1, padding='same',
                                                    layer_name='y_pred', s_norm=True)
            pred_y_dense = tf.nn.tanh(pred_y_dense)*self.max_disp + y_motion_fields
            
            # dense match networks
            x_decode = custom_layers.conv_layer(tf.concat([z_feat, x_feat_warp], 4), num_filters=self.nf * 8,
                                                filter_sz=self.fz, stride=1, padding='same', layer_name='x_decode1',
                                                leaky_relu=True, s_norm=True)
            x_decode = custom_layers.upconv_layer(x_decode, num_filters=self.nf * 4, filter_sz=self.fz, stride=2,
                                                  padding='same', layer_name='x_upconv1', leaky_relu=True, s_norm=True)
            x_decode = custom_layers.upconv_layer(x_decode, num_filters=self.nf * 2, filter_sz=self.fz, stride=2,
                                                  padding='same', layer_name='x_upconv2', leaky_relu=True, s_norm=True)
            x_decode = custom_layers.upconv_layer(x_decode, num_filters=self.nf * 1, filter_sz=self.fz, stride=2,
                                                  padding='same', layer_name='x_upconv3', leaky_relu=True, s_norm=True)
            pred_x_dense = custom_layers.conv_layer(x_decode, num_filters=3, filter_sz=1, stride=1, padding='same',
                                                    layer_name='x_pred', s_norm=True)
            pred_x_dense = tf.nn.tanh(pred_x_dense)*self.max_disp + x_motion_fields            
            
        return pred_y_dense, pred_x_dense
            

class DeblurNet:
    def __init__(self, args):
        self.batch_size = args.batch_size
        self.nf = args.nf
        self.fz = args.fz
    
    def __call__(self, z_view, y_view, x_view, name = 'DeblurNet'):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            # z view features
            z_feat1 = custom_layers.conv_layer(z_view, num_filters=self.nf * 1, filter_sz=self.fz, stride=1,
                                               padding='same', layer_name='feat1', leaky_relu=True, s_norm=True)
            z_feat2 = custom_layers.conv_layer(z_feat1, num_filters=self.nf * 2, filter_sz=self.fz, stride=2,
                                               padding='same', layer_name='feat2', leaky_relu=True, s_norm=True)
            z_feat3 = custom_layers.conv_layer(z_feat2, num_filters=self.nf * 4, filter_sz=self.fz, stride=2,
                                               padding='same', layer_name='feat3', leaky_relu=True, s_norm=True)
            z_feat4 = custom_layers.conv_layer(z_feat3, num_filters=self.nf * 8, filter_sz=self.fz, stride=2,
                                               padding='same', layer_name='feat4', leaky_relu=True, s_norm=True)

            # y view features
            y_feat1 = custom_layers.conv_layer(y_view, num_filters=self.nf * 1, filter_sz=self.fz, stride=1,
                                               padding='same', layer_name='feat1', leaky_relu=True, s_norm=True)
            y_feat2 = custom_layers.conv_layer(y_feat1, num_filters=self.nf * 2, filter_sz=self.fz, stride=2,
                                               padding='same', layer_name='feat2', leaky_relu=True, s_norm=True)
            y_feat3 = custom_layers.conv_layer(y_feat2, num_filters=self.nf * 4, filter_sz=self.fz, stride=2,
                                               padding='same', layer_name='feat3', leaky_relu=True, s_norm=True)
            y_feat4 = custom_layers.conv_layer(y_feat3, num_filters=self.nf * 8, filter_sz=self.fz, stride=2,
                                               padding='same', layer_name='feat4', leaky_relu=True, s_norm=True)

            # x view features
            x_feat1 = custom_layers.conv_layer(x_view, num_filters=self.nf * 1, filter_sz=self.fz, stride=1,
                                               padding='same', layer_name='feat1', leaky_relu=True, s_norm=True)
            x_feat2 = custom_layers.conv_layer(x_feat1, num_filters=self.nf * 2, filter_sz=self.fz, stride=2,
                                               padding='same', layer_name='feat2', leaky_relu=True, s_norm=True)
            x_feat3 = custom_layers.conv_layer(x_feat2, num_filters=self.nf * 4, filter_sz=self.fz, stride=2,
                                               padding='same', layer_name='feat3', leaky_relu=True, s_norm=True)
            x_feat4 = custom_layers.conv_layer(x_feat3, num_filters=self.nf * 8, filter_sz=self.fz, stride=2,
                                               padding='same', layer_name='feat4', leaky_relu=True, s_norm=True)
            
            max_feat1 = tf.maximum(z_feat1, tf.maximum(y_feat1, x_feat1))
            max_feat2 = tf.maximum(z_feat2, tf.maximum(y_feat2, x_feat2))
            max_feat3 = tf.maximum(z_feat3, tf.maximum(y_feat3, x_feat3))
            max_feat4 = tf.maximum(z_feat4, tf.maximum(y_feat4, x_feat4))

            conv = custom_layers.conv_layer(max_feat4, num_filters=self.nf * 8, filter_sz=self.fz, stride=1,
                                            padding='same', layer_name='conv1', leaky_relu=True, s_norm=True)
            conv = custom_layers.upconv_layer(conv, num_filters=self.nf * 4, filter_sz=self.fz, stride=2,
                                              padding='same', layer_name='conv1_up', leaky_relu=True, s_norm=True)

            conv = custom_layers.conv_layer(tf.concat([conv, max_feat3], 4), num_filters=self.nf * 4, filter_sz=self.fz,
                                            stride=1, padding='same', layer_name='conv2', leaky_relu=True, s_norm=True)
            conv = custom_layers.upconv_layer(conv, num_filters=self.nf * 2, filter_sz=self.fz, stride=2,
                                              padding='same', layer_name='conv2_up', leaky_relu=True, s_norm=True)

            conv = custom_layers.conv_layer(tf.concat([conv, max_feat2], 4), num_filters=self.nf * 2, filter_sz=self.fz,
                                            stride=1, padding='same', layer_name='conv3', leaky_relu=True, s_norm=True)
            conv = custom_layers.upconv_layer(conv, num_filters=self.nf * 1, filter_sz=self.fz, stride=2,
                                              padding='same', layer_name='conv3_up', leaky_relu=True, s_norm=True)

            conv = custom_layers.conv_layer(tf.concat([conv, max_feat1], 4), num_filters=self.nf * 1, filter_sz=self.fz,
                                            stride=1, padding='same', layer_name='conv4', leaky_relu=True, s_norm=True)
            pred_fuse = custom_layers.conv_layer(conv, num_filters=1, filter_sz=self.fz, stride=1, padding='same',
                                                 layer_name='pred_fuse', s_norm=True)
            pred_fuse = tf.nn.tanh(pred_fuse)
            
        return pred_fuse
