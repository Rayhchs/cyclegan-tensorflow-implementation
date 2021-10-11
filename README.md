# cyclegan-tensorflow-implementation

A tensorflow implementation of [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://junyanz.github.io/CycleGAN/). In this repo, generator with 9 residual block is used. However, instead of using 256 by 256 pixels for input size, 128 by 128 pixels is adopted due to device limintation. This repo implement two tasks. The first one is original implementation - aerial image 2 google map. The other one is implemented for fun - mask 2 non mask.

## Requisite

* python 3.8
* tensorflow 2.5.0
* Cuda 11.1
* CuDNN 8.1.1

	  pip install -r requirements.txt
    
## Getting Started
* Clone this repo

      git clone https://github.com/Rayhchs/cyclegan-tensorflow-implementation.git
      cd cyclegan-tensorflow-implementation
    
* Train

      python -m main train
    
There are two things should be input: directions containing domain A & B images. 
* Train

      python -m main test
    
There are two things should be input: direction of testing image and transform type (A2B or B2A). 
* Arguments

 | Positional arguments | Description |
 | ------------- | ------------- |
 | mode | train or test |
 
 | Optional arguments | prefix_char | Description |
 | ------------- | ------------- |------------- |
 | --epoch | -e | epochs, default=400 |
 | --batch_size | -b | batch size, default=1 |
 | --save_path | -s | path to save testing result, default= .\result |
 | --lambda_cc | -lc | lambda of identity loss, default=10 |
 | --decay | -d | learning rate of discriminator decay from which epoch, default=100 |
 | --do_idt_loss | -id | include identity loss for generator loss or not, default=False |
 | --do_resize | -r | resize to original size or not, default=False |
 
 ## Results
Here is the results generated from this implementation:

* Aerial map (train for 200 epochs):

| Aerial image | Generated map (pix2pix) | Generated map (cycleGAN) | Ground truth |
| ------------- | ------------- | ------------- | ------------- |
| <img src="https://github.com/Rayhchs/Pix2pix-tensorflow-implementation/blob/main/test/3.jpg" width="250"> | <img src="https://github.com/Rayhchs/Pix2pix-tensorflow-implementation/blob/main/result/3.jpg" width="250"> | <img src="https://github.com/Rayhchs/Pix2pix-tensorflow-implementation/blob/main/test/label_3.jpg" width="250">|
| <img src="https://github.com/Rayhchs/Pix2pix-tensorflow-implementation/blob/main/test/6.jpg" width="250"> | <img src="https://github.com/Rayhchs/Pix2pix-tensorflow-implementation/blob/main/result/6.jpg" width="250"> | <img src="https://github.com/Rayhchs/Pix2pix-tensorflow-implementation/blob/main/test/label_6.jpg" width="250">|
| <img src="https://github.com/Rayhchs/Pix2pix-tensorflow-implementation/blob/main/test/24.jpg" width="250"> | <img src="https://github.com/Rayhchs/Pix2pix-tensorflow-implementation/blob/main/result/24.jpg" width="250"> | <img src="https://github.com/Rayhchs/Pix2pix-tensorflow-implementation/blob/main/test/label_24.jpg" width="250">|
| <img src="https://github.com/Rayhchs/Pix2pix-tensorflow-implementation/blob/main/test/33.jpg" width="250"> | <img src="https://github.com/Rayhchs/Pix2pix-tensorflow-implementation/blob/main/result/33.jpg" width="250"> | <img src="https://github.com/Rayhchs/Pix2pix-tensorflow-implementation/blob/main/test/label_33.jpg" width="250"> |
| <img src="https://github.com/Rayhchs/Pix2pix-tensorflow-implementation/blob/main/test/45.jpg" width="250"> | <img src="https://github.com/Rayhchs/Pix2pix-tensorflow-implementation/blob/main/result/45.jpg" width="250"> | <img src="https://github.com/Rayhchs/Pix2pix-tensorflow-implementation/blob/main/test/label_45.jpg" width="250">|
