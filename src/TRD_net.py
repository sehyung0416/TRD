import tensorflow as tf
import numpy as np
import tifffile as tiff
import time
import os
import utils
import sys

import cnn_models
from glob import glob
from spatial_transformer import Dense3DSpatialTransformer

class TRDNet():
    
    def __init__(self, args):    
        
        self.phase = args.phase        
        self.model_name = args.model_name
        self.training_dataset_path = args.training_dataset_path
        self.test_dataset_path = args.test_dataset_path
        self.model_dir = os.path.join(args.model_name, args.save_dir)
        self.sample_dir = os.path.join(args.model_name, args.sample_dir)
        
        self.learning_rate = args.learning_rate
        self.beta1 = args.beta1
        self.beta2 = args.beta2
        
        self.batch_size = args.batch_size
        self.in_patch_size = args.in_patch_size
        self.out_patch_size = args.out_patch_size
        self.margin = int((self.in_patch_size-self.out_patch_size)/2)
        
        self.max_angle = args.max_angle
        self.max_trs = args.max_trs
        self.max_disp = args.max_disp
        
        self.blur_sigma = args.blur_sigma
        
        self.nf = args.nf
        self.fz = args.fz
        self.smooth_weight = args.smooth_weight

        self.training_epochs = args.training_epochs
        self.print_freq = args.print_freq
        self.simulation_interval = args.simulation_interval

        self.train_dataset_list = glob('./{}/*.*'.format(self.training_dataset_path))
        self.dataset_num = len(self.train_dataset_list)
        
        file = open(os.path.join(args.model_name, 'args.txt'), "w")
        file.write("%s = %s\n" %('batch_size', self.batch_size))
        file.write("%s = %s\n" %('in_patch_size', self.in_patch_size))
        file.write("%s = %s\n" %('out_patch_size', self.out_patch_size))
        file.write("%s = %s\n" %('learning_rate', self.learning_rate))
        file.write("%s = %s\n" %('beta1', self.beta1))
        file.write("%s = %s\n" %('beta2', self.beta2))
        file.write("%s = %s\n" %('max_angle', self.max_angle))
        file.write("%s = %s\n" %('max_trs', self.max_trs))
        file.write("%s = %s\n" %('nf', self.nf))
        file.write("%s = %s\n" %('fz', self.fz))
        file.write("%s = %s\n" %('blur_sigma', self.blur_sigma))
        file.write("%s = %s\n" %('smooth_weight', self.smooth_weight))
        file.write("%s = %s\n" %('training_epochs', self.training_epochs))
        file.write("%s = %s\n" %('print_freq', self.print_freq))
        file.close()    
                      
        self.reg_net = cnn_models.RegNet(args)
        self.deblur_net = cnn_models.DeblurNet(args)
        self.tranformer = Dense3DSpatialTransformer()

    def model_setup(self):
        # place holders
        self.misaligned_imgs = tf.placeholder(tf.float32,
                                              shape=[self.batch_size, self.in_patch_size, self.in_patch_size,
                                                     self.in_patch_size, 3], name='misaligned_imgs')
        self.aligned_imgs = tf.placeholder(tf.float32, shape=[self.batch_size, self.in_patch_size, self.in_patch_size,
                                                              self.in_patch_size, 3], name='aligned_imgs')
        self.fused_imgs = tf.placeholder(tf.float32, shape=[self.batch_size, self.in_patch_size, self.in_patch_size,
                                                           self.in_patch_size, 1], name='fused_img')

        self.z_view, self.y_view, self.x_view = tf.split(self.misaligned_imgs, num_or_size_splits=3, axis=4)
        self.z_align, self.y_align, self.x_align = tf.split(self.aligned_imgs, num_or_size_splits=3, axis=4)

        with tf.variable_scope("Model"):
            self.y_pred_motion, self.x_pred_motion = self.reg_net(self.z_view, self.y_view, self.x_view,
                                                                  self.tranformer, name='registration_network')
            
            dx, dy, dz = tf.split(self.y_pred_motion, num_or_size_splits=3, axis=4)
            dx = tf.reshape(dx, [self.batch_size, self.in_patch_size, self.in_patch_size, self.in_patch_size])
            dy = tf.reshape(dy, [self.batch_size, self.in_patch_size, self.in_patch_size, self.in_patch_size])
            dz = tf.reshape(dz, [self.batch_size, self.in_patch_size, self.in_patch_size, self.in_patch_size])
            self.y_dense_warp = self.tranformer._transform(self.y_view, dx, dy, dz)
    
            dx, dy, dz = tf.split(self.x_pred_motion, num_or_size_splits=3, axis=4)
            dx = tf.reshape(dx, [self.batch_size, self.in_patch_size, self.in_patch_size, self.in_patch_size])
            dy = tf.reshape(dy, [self.batch_size, self.in_patch_size, self.in_patch_size, self.in_patch_size])
            dz = tf.reshape(dz, [self.batch_size, self.in_patch_size, self.in_patch_size, self.in_patch_size])
            self.x_dense_warp = self.tranformer._transform(self.x_view, dx, dy, dz)

            self.z_view_slice = self.get_slice_img(self.z_view)
            self.y_view_slice = self.get_slice_img(self.y_dense_warp)
            self.x_view_slice = self.get_slice_img(self.x_dense_warp)
            self.pred_align = tf.concat([self.z_view_slice, self.y_view_slice, self.x_view_slice],4)
            self.pred_fuse = self.deblur_net(self.z_view_slice, self.y_view_slice, self.x_view_slice,
                                             name='deblur_network')
            
    def get_slice_img(self, img):
        return tf.slice(img, [0, self.margin, self.margin, self.margin, 0],
                        [self.batch_size, self.out_patch_size, self.out_patch_size, self.out_patch_size, 1])
    
    def neighbor_loss(self, motion):
        dx = motion[:, 1:, :, :, :] - motion[:, :-1, :, :, :]
        dy = motion[:, :, 1:, :, :] - motion[:, :, :-1, :, :]
        dz = motion[:, :, :, 1:, :] - motion[:, :, :, :-1, :]
        return self.L1_loss(dx, 0) + self.L1_loss(dy, 0) + self.L1_loss(dz, 0)

    def L1_loss(self, a, b):
        return tf.reduce_mean(tf.abs(a-b))

    def loss_setup(self):
        # registration loss
        self.smoothness_loss = self.neighbor_loss(self.get_slice_img(self.y_pred_motion)) \
                               + self.neighbor_loss(self.get_slice_img(self.x_pred_motion))
        self.alignment_loss = self.L1_loss(self.y_view_slice, self.get_slice_img(self.y_align)) \
                              + self.L1_loss(self.x_view_slice, self.get_slice_img(self.x_align))
        self.registration_loss = self.alignment_loss + self.smoothness_loss*self.smooth_weight
        
        # deblurring loss
        self.deblur_loss = self.L1_loss(self.pred_fuse, self.get_slice_img(self.fused_imgs))
        optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1, beta2=self.beta2)        
        self.model_vars = tf.trainable_variables()
        registration_vars = [var for var in self.model_vars if 'registration_network' in var.name]
        deblur_vars = [var for var in self.model_vars if 'deblur_network' in var.name]
        for var in registration_vars: print(var.name)
        print('-------------------------')
        for var in deblur_vars: print(var.name)
        print('-------------------------')
        self.registration_trainer = optimizer.minimize(self.registration_loss, var_list=registration_vars)
        self.deblur_trainer = optimizer.minimize(self.deblur_loss, var_list=deblur_vars)

    def train(self):
        self.model_setup()
        self.loss_setup()        
        saver = tf.train.Saver()
        
        with tf.Session( ) as sess: 
            sess.run(tf.global_variables_initializer())
            tf.get_default_graph().finalize()
            
            original_image_path = os.path.join(self.training_dataset_path, 'samples')
            source_image_path = os.path.join(self.training_dataset_path, 'source')
            align_image_path = os.path.join(self.training_dataset_path, 'aligned')
            misalign_image_path = os.path.join(self.training_dataset_path, 'misaligned')

            if not os.path.exists(source_image_path):
                os.makedirs(source_image_path)
            if not os.path.exists(align_image_path):
                os.makedirs(align_image_path)
            if not os.path.exists(misalign_image_path):
                os.makedirs(misalign_image_path)

            img_list = utils.get_img_list(original_image_path)
            num_img = len(img_list)
            iter_per_epoch = int(num_img/self.batch_size)
            
            for epoch in range(0, self.training_epochs):
                if epoch % self.simulation_interval == 0:
                    print("synthetic image generation")
                    # self.data_simulation(img_list, self.training_dataset_path)
                
                np.random.shuffle(img_list)
                for seq in range(0, iter_per_epoch):
                    source_batch = []
                    aligned_batch = []
                    misaligned_batch = []
                    for i in range(0,self.batch_size):                        
                        source_img = tiff.imread("%s/%s" % (source_image_path, img_list[seq*self.batch_size+i]))
                        aligned_img = tiff.imread("%s/%s" % (align_image_path, img_list[seq*self.batch_size+i]))
                        misaligned_img = tiff.imread("%s/%s" % (misalign_image_path, img_list[seq*self.batch_size+i]))
                        
                        source_batch.append(source_img)
                        aligned_batch.append(aligned_img)
                        misaligned_batch.append(misaligned_img)

                    t = time.time()
                    source_batch = np.reshape(source_batch, [self.batch_size, self.in_patch_size, self.in_patch_size,
                                                             self.in_patch_size, 1])
                    aligned_batch = np.reshape(aligned_batch, [self.batch_size, self.in_patch_size, self.in_patch_size,
                                                               self.in_patch_size, 3])
                    misaligned_batch = np.reshape(misaligned_batch,
                                                  [self.batch_size, self.in_patch_size, self.in_patch_size,
                                                   self.in_patch_size, 3])
                    
                    source_batch = utils.data_normalize(source_batch)
                    aligned_batch = utils.data_normalize(aligned_batch)
                    misaligned_batch = utils.data_normalize(misaligned_batch)

                    _, pred_align, z_view_slice, y_view_slice, x_view_slice, registration_loss = \
                        sess.run([self.registration_trainer, self.pred_align, self.z_view_slice, self.y_view_slice,
                                  self.x_view_slice, self.registration_loss],
                                 feed_dict={self.misaligned_imgs: misaligned_batch, self.aligned_imgs: aligned_batch})

                    _, pred_fuse, deblur_loss = \
                        sess.run([self.deblur_trainer, self.pred_fuse, self.deblur_loss],
                                 feed_dict={self.fused_imgs: source_batch, self.z_view_slice: z_view_slice,
                                            self.y_view_slice: y_view_slice, self.x_view_slice: x_view_slice})

                    if seq%self.print_freq == 0:
                        align_path = './{}/align_{:02d}_{:06d}.tif'.format(self.sample_dir, epoch, seq+1)
                        fuse_path = './{}/fuse_{:02d}_{:06d}.tif'.format(self.sample_dir, epoch, seq+1)
                        
                        align_img = utils.data_denormalize(pred_align[0])
                        fuse_img = utils.data_denormalize(pred_fuse[0])
                        
                        tiff.imsave(align_path, align_img.astype('uint8'))
                        tiff.imsave(fuse_path, fuse_img.astype('uint8'))
                    
                    elapsed = time.time() - t
                    print("epoch %d: step %d/%d, r_loss%g, d_loss=%g, time=%g " % (
                    epoch, seq, iter_per_epoch, registration_loss, deblur_loss, elapsed))
                    
                saver.save(sess, './{}/model'.format(self.model_dir), write_meta_graph=False)

    def data_simulation(self, img_list, dataset_path):
        num_img = len(img_list)
        for i in range(0, num_img):
            img_name = img_list[i]
            original_img = tiff.imread("%s/samples/%s" % (dataset_path, img_name))
            source_img, misaligned_img, aligned_img = utils.registration_simul(original_img, self.in_patch_size,
                                                                               self.max_angle, self.max_trs)
            misaligned_img = utils.lens_simul(misaligned_img, self.blur_sigma)
            aligned_img = utils.lens_simul(aligned_img, self.blur_sigma)
            tiff.imsave("%s/source/%s" % (dataset_path, img_name), source_img)
            tiff.imsave("%s/misaligned/%s" % (dataset_path, img_name), misaligned_img)
            tiff.imsave("%s/aligned/%s" % (dataset_path, img_name), aligned_img)
            self.print_progress_bar(i/num_img*50, 50, "progress")
                
    def load_model(self):
        print('Load trained model')
        self.batch_size = 1
        self.model_setup()
        saver = tf.train.Saver()
        self.sess = tf.Session()
        saver.restore(self.sess, './{}/model'.format(self.model_dir))

    def padding_image(self, img):
        sz = img.shape 
        pad_img = np.random.rand(sz[0]+self.margin*2, sz[1]+self.margin*2, sz[2]+self.margin*2)*10 + 5
        pad_img[self.margin:sz[0]+self.margin, self.margin:sz[1]+self.margin, self.margin:sz[2]+self.margin] = img
        return pad_img
    
    def test(self, z_view, y_view, x_view):
        if z_view.ndim == 4:
            z_view = np.reshape(z_view, (z_view.shape[0], z_view.shape[1], z_view.shape[2]))
            
        if y_view.ndim == 4:
            y_view = np.reshape(y_view, (y_view.shape[0], y_view.shape[1], y_view.shape[2]))
            
        if x_view.ndim == 4:
            x_view = np.reshape(x_view, (x_view.shape[0], x_view.shape[1], x_view.shape[2]))
            
        align_img, fuse_img = self.predict_registration_fusion(z_view, y_view, x_view)
        
        return align_img, fuse_img
    
    def print_progress_bar(self, i, max_bars, post_text):
        n_bar = max_bars
        j= i/max_bars
        sys.stdout.write('\r')
        sys.stdout.write(f"[{'=' * int(n_bar * j):{n_bar}s}] {int(100 * j)}%  {post_text}")
        sys.stdout.flush()
        
    def sliding_convolution(self, z_view, y_view, x_view, z, y, x, align_img, fuse_img, occupancy):
        z_patch = z_view[z:z + self.in_patch_size, y:y + self.in_patch_size, x:x + self.in_patch_size]
        y_patch = y_view[z:z + self.in_patch_size, y:y + self.in_patch_size, x:x + self.in_patch_size]
        x_patch = x_view[z:z + self.in_patch_size, y:y + self.in_patch_size, x:x + self.in_patch_size]

        z_patch = np.reshape(z_patch, [1, self.in_patch_size, self.in_patch_size, self.in_patch_size, 1])
        y_patch = np.reshape(y_patch, [1, self.in_patch_size, self.in_patch_size, self.in_patch_size, 1])
        x_patch = np.reshape(x_patch, [1, self.in_patch_size, self.in_patch_size, self.in_patch_size, 1])

        pred_align, pred_fuse = self.sess.run([self.pred_align, self.pred_fuse],
                                              feed_dict={self.z_view: z_patch, self.y_view: y_patch,
                                                         self.x_view: x_patch})
        pred_align = np.reshape(pred_align, [self.out_patch_size, self.out_patch_size, self.out_patch_size, 3])
        pred_fuse = np.reshape(pred_fuse, [self.out_patch_size, self.out_patch_size, self.out_patch_size])

        pred_align = np.clip(pred_align, -1.0, 1.0)
        pred_fuse = np.clip(pred_fuse, -1.0, 1.0)
        
        m = self.margin
        wz = self.out_patch_size
        align_img[z + m:z + m + wz, y + m:y + m + wz, x + m:x + m + wz, :] += pred_align
        fuse_img[z + m:z + m + wz, y + m:y + m + wz, x + m:x + m + wz] += pred_fuse
        occupancy[z + m:z + m + wz, y + m:y + m + wz, x + m:x + m + wz] += 1
        return align_img, fuse_img, occupancy
    
    def predict_registration_fusion(self, z_view, y_view, x_view):
        z_view = self.padding_image(z_view)
        y_view = self.padding_image(y_view)
        x_view = self.padding_image(x_view)        
        
        img_sz = z_view.shape
        z_view = z_view.astype('float32')/127.5 - 1.0
        y_view = y_view.astype('float32')/127.5 - 1.0
        x_view = x_view.astype('float32')/127.5 - 1.0
        
        align_img = np.zeros(img_sz, dtype=np.float32)
        align_img = np.stack((align_img, align_img, align_img), axis=3)
        fuse_img = np.zeros(img_sz, dtype=np.float32)
        occupancy = np.zeros(img_sz, dtype=np.float32)

        patch_sz = int(self.in_patch_size)
        sampling_step = int(self.out_patch_size-3)
        
        z_samples = list(range(0, img_sz[0]-patch_sz+1, sampling_step))
        z_samples.append(img_sz[0]-patch_sz)
        y_samples = list(range(0, img_sz[1]-patch_sz+1, sampling_step))
        y_samples.append(img_sz[1]-patch_sz)
        x_samples = list(range(0, img_sz[2]-patch_sz+1, sampling_step))
        x_samples.append(img_sz[2]-patch_sz)        
        
        num_total_bars = 50
        num_total_iters = len(z_samples)*len(y_samples)*len(x_samples)

        iter_count = 0
        for i in range(0, len(z_samples)):
            z = z_samples[i]
            for j in range(0, len(y_samples)):
                y = y_samples[j]
                for k in range(0, len(x_samples)):
                    x = x_samples[k]
                    align_img, fuse_img, occupancy = self.sliding_convolution(z_view, y_view, x_view, z, y, x,
                                                                              align_img, fuse_img, occupancy)
                    iter_count = iter_count + 1
                    self.print_progress_bar(iter_count/num_total_iters*num_total_bars, num_total_bars, "progress")
        self.print_progress_bar(iter_count/num_total_iters*num_total_bars, num_total_bars, "complete\n")
        
        epsilon = 1E-10
        occupancy = occupancy + epsilon
        align_img[..., 0] = np.divide(align_img[..., 0], occupancy)
        align_img[..., 1] = np.divide(align_img[..., 1], occupancy)
        align_img[..., 2] = np.divide(align_img[..., 2], occupancy)
        fuse_img = np.divide(fuse_img, occupancy)
        
        align_img = (align_img + 1.0)*127.5
        fuse_img = (fuse_img + 1.0)*127.5
        
        align_img = np.clip(align_img, 0, 255)
        fuse_img = np.clip(fuse_img, 0, 255)

        align_img = align_img[self.margin:fuse_img.shape[0] - self.margin, self.margin:fuse_img.shape[1] - self.margin,
                    self.margin:fuse_img.shape[2] - self.margin, :]
        fuse_img = fuse_img[self.margin:fuse_img.shape[0] - self.margin, self.margin:fuse_img.shape[1] - self.margin,
                   self.margin:fuse_img.shape[2] - self.margin]

        return align_img, fuse_img