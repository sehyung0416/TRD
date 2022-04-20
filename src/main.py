import argparse
import os
import tifffile as tiff
from TRD_net import TRDNet


def parse_args():
    """parsing and configuration"""
    desc = "Tensorflow implementation of TRD-net"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--phase', type=str, default='train', help='train or test')
    parser.add_argument('--training_dataset_path', type=str, default='./', help='training dataset path')
    parser.add_argument('--test_dataset_path', type=str, default='./', help='test dataset path')

    parser.add_argument('--training_epochs', type=int, default=100, help='The number of epochs to run')
    parser.add_argument('--print_freq', type=int, default=500, help='The number of image print frequency')
    
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='The learning rate')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for ADAM optimizer')
    parser.add_argument('--beta2', type=float, default=0.9, help='beta2 for ADAM optimizer')
    
    parser.add_argument('--batch_size', type=int, default=16, help='The batch size')
    parser.add_argument('--in_patch_size', type=int, default=128, help='The input patch size')
    parser.add_argument('--out_patch_size', type=int, default=96, help='The output patch size')
    parser.add_argument('--simulation_interval', type=int, default=5, help='The simulation interval')
    
    parser.add_argument('--max_angle', type=float, default=20, help='maximum rotation of rigid transformation (degree)')
    parser.add_argument('--max_trs', type=float, default=40, help='maximum translation of rigid transformation')
    parser.add_argument('--max_disp', type=float, default=15, help='maximum voxel-wise displacement')
    parser.add_argument('--blur_sigma', type=float, default=8.5, help='Gaussian PSF')
        
    parser.add_argument('--fz', type=int, default=3, help='convolution filter size')
    parser.add_argument('--nf', type=int, default=16, help='base channel number per layer')
    parser.add_argument('--smooth_weight', type=float, default=0.001, help='smoothness weight')

    parser.add_argument('--model_name', type=str, default='model', help='model name used as folder name')
    parser.add_argument('--save_dir', type=str, default='save', help='subfolder name to save checkpoint files')
    parser.add_argument('--sample_dir', type=str, default='samples', help='subfolder name to save the samples')

    parser.add_argument('--test_result', type=str, default='test_result', help='subfolder name to save the test images')

    return check_args(parser.parse_args())


def check_folder(test_dir):
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    return test_dir


def check_args(args):
    """checking arguments"""
    # --model_name
    try:
        assert bool(args.model_name)
    except:
        print('model name must be given')
    
    check_folder(args.model_name)
    
    # --save dir
    check_folder(os.path.join(args.model_name, args.save_dir))

    # --sample_dir
    check_folder(os.path.join(args.model_name, args.sample_dir))
    
    check_folder(os.path.join(args.model_name, args.test_result))

    # --epoch
    try:
        assert args.training_epochs >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')
        
    return args


def main():
    """main"""
    # parse arguments
    args = parse_args()
    if args is None:
        exit()
        
    trd_net = TRDNet(args)

    if args.phase == 'train' :
        trd_net.train()
        print(" [*] Training finished!")
        
    elif args.phase == 'test':
        trd_net.load_model()

        for subfolder in os.listdir(trd_net.test_dataset_path):
            file_path = trd_net.test_dataset_path + subfolder
            print('read images from %s' % file_path)
            z_view = tiff.imread(file_path + '/z_img.tif')
            y_view = tiff.imread(file_path + '/y_img.tif')
            x_view = tiff.imread(file_path + '/x_img.tif')

            aligned_img, fused_img = trd_net.test(z_view, y_view, x_view)
            aligned_img = aligned_img.astype('uint8')
            fused_img = fused_img.astype('uint8')

            output_path = args.model_name + '/' + args.test_result + '/' + subfolder
            check_folder(output_path)
            print('save images to %s' % output_path)
            align_path = output_path + '/aligned_image.tif'
            fuse_path = output_path + '/fused_image.tif'
            tiff.imsave(align_path, aligned_img)
            tiff.imsave(fuse_path, fused_img)

        print(" [*] Test finished!")


if __name__ == '__main__':
    main()
