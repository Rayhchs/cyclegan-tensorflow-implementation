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
