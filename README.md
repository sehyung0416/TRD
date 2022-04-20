## Reference
* Sehyung Lee, Hideaki Kume, Hidetoshi Urakubo, Haruo Kasai, Shin Ishii, "Tri-view two-photon microscopic image registration and deblurring with convolutional neural networks", Neural Networks 2022. 

# Requirements 
Tensorflow v1.15, python 3.7.6, scipy, and tifffile (3D image read and write)

We verified that RTX 2080ti and V100 GPUs are working with CUDA 10.1 

# Training command
python ./src/main.py --phase train --training_dataset_path DATASET_PATH --model_name MODEL_NAME

In this implementation, we provide sample training images that can be found at https://sites.google.com/view/sehyung/home/projects

Please prepare the training images and set to DATASET_PATH when you run the training command

The training dataset folder tree is made of 

DATASET_PATH --- samples (sample training images)

                  |
                  
                  --- source (synthetic source images)
                  
                  | 
               
                  --- misaligned (synthetic misaligned images)
                  
                  | 
                  
                  --- aligned (synthetic aligned images)

During training, synthetic source, misaligned, aligned images are created based on the real images included in the samples folder

	   
Note that MODEL_NAME is used to save the trained model and monitor sample images generated during training steps


Please refer to the default parameter settings in "def parse_args()" of main.py and paper

# Test command
python main.py --phase test --
Please check the test dataset and refer to the scripts 'test.py' and 'test_demo.py'



