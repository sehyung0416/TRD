import numpy as np
import os
import scipy
import fnmatch
import random


def get_img_list(img_dir):
    img_list = fnmatch.filter(os.listdir("%s" % (img_dir)), '*.tif')
    return img_list


def data_denormalize(data):
    data = np.clip(data, -1, 1)
    denorm_data = (data+1.0)*255.0/2.0
    return denorm_data


def data_normalize(data):
    norm_data = data.astype('float32')/255.0*2.0 - 1.0
    norm_data = np.clip(norm_data, -1, 1)
    return norm_data


def rand_scale(source_img, misalign_img, align_img, min_scale, max_scale):
    z_scale = np.random.uniform(min_scale, max_scale)
    y_scale = np.random.uniform(min_scale, max_scale)
    x_scale = np.random.uniform(min_scale, max_scale)
    source_img = source_img*z_scale
    misalign_img[..., 0] = misalign_img[..., 0] * z_scale
    misalign_img[..., 1] = misalign_img[..., 1] * y_scale
    misalign_img[..., 2] = misalign_img[..., 2] * x_scale
    align_img[..., 0] = align_img[..., 0] * z_scale
    align_img[..., 1] = align_img[..., 1] * y_scale
    align_img[..., 2] = align_img[..., 2] * x_scale
    
    return source_img, misalign_img, align_img


def rand_transpose(img):
    seed = np.random.uniform(0, 1)
    if seed < 0.33:
        img = np.transpose(img, (1, 0, 2))
    elif seed < 0.66:
        img = np.transpose(img, (2, 1, 0))
    return img


def random_rotation_3d(img, max_angle):
    # rotate along z-axis
    angle = random.uniform(-max_angle, max_angle)
    rot_img = scipy.ndimage.interpolation.rotate(img, angle, mode='nearest', axes=(0, 1), reshape=False) 

    # rotate along y-axis
    angle = random.uniform(-max_angle, max_angle)
    rot_img = scipy.ndimage.interpolation.rotate(rot_img, angle, mode='nearest', axes=(0, 2), reshape=False) 

    # rotate along x-axis
    angle = random.uniform(-max_angle, max_angle)
    rot_img = scipy.ndimage.interpolation.rotate(rot_img, angle, mode='nearest', axes=(1, 2), reshape=False)     
    
    return rot_img


def lens_simul(img, blur_sigma):
    z_view = img[..., 0]
    y_view = img[..., 1]
    x_view = img[..., 2]

    z_simul = scipy.ndimage.filters.gaussian_filter(z_view, (blur_sigma, 0.1, 0.1))
    y_simul = scipy.ndimage.filters.gaussian_filter(y_view, (0.1, blur_sigma, 0.1))
    x_simul = scipy.ndimage.filters.gaussian_filter(x_view, (0.1, 0.1, blur_sigma))

    z_simul = np.expand_dims(z_simul, axis=3)
    y_simul = np.expand_dims(y_simul, axis=3)
    x_simul = np.expand_dims(x_simul, axis=3)
    blurry_img = np.concatenate((z_simul, y_simul, x_simul), axis=3)
        
    return blurry_img


def registration_simul(img, patch_size, max_rot, max_trs):
    img = rand_transpose(img)
    img_sz = img.shape[0]
    rot_img1 = random_rotation_3d(img, max_rot)
    rot_img2 = random_rotation_3d(img, max_rot)
    
    z0 = int(img_sz/2-patch_size/2)
    y0 = int(img_sz/2-patch_size/2)
    x0 = int(img_sz/2-patch_size/2)
    
    z1 = z0 + random.randint(-3, 3)
    y1 = y0 + random.randint(-3, 3)
    x1 = x0 + random.randint(-3, 3)
    source_img = img[z1:z1+patch_size, y1:y1+patch_size, x1:x1+patch_size]

    while True:
        z2 = z1 + random.randint(-max_trs, max_trs)
        if z2 > 0 and z2 + patch_size < img_sz:
            break            
    while True:
        y2 = y1 + random.randint(-max_trs, max_trs)
        if y2 > 0 and y2 + patch_size < img_sz:
            break            
    while True:
        x2 = x1 + random.randint(-max_trs, max_trs)
        if x2 > 0 and x2 + patch_size < img_sz:
            break
    rot_patch1 = rot_img1[z2:z2+patch_size, y2:y2+patch_size, x2:x2+patch_size]
    
    while True:
        z3 = z1 + random.randint(-max_trs, max_trs)
        if z3 > 0 and z3 + patch_size < img_sz:
            break            
    while True:
        y3 = y1 + random.randint(-max_trs, max_trs)
        if y3 > 0 and y3 + patch_size < img_sz:
            break
    while True:
        x3 = x1 + random.randint(-max_trs, max_trs)
        if x3 > 0 and x3 + patch_size < img_sz:
            break
    rot_patch2 = rot_img2[z3:z3+patch_size, y3:y3+patch_size, x3:x3+patch_size]
    
    z_view = np.expand_dims(source_img,axis=3)
    y_view = np.expand_dims(rot_patch1,axis=3)
    x_view = np.expand_dims(rot_patch2,axis=3)
    misaligned_img = np.concatenate( (z_view, y_view, x_view), axis=3)
    aligned_img = np.concatenate((z_view, z_view, z_view), axis=3)
        
    return source_img, misaligned_img, aligned_img
