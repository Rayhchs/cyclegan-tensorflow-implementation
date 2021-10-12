# cyclegan-tensorflow-implementation

A tensorflow implementation of [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://junyanz.github.io/CycleGAN/). In this repo, generator with 9 residual block is used. However, instead of using 256 by 256 pixels for input size, 128 by 128 pixels is adopted due to device limintation. This repo implement two tasks. The first one is original implementation - aerial image 2 google map. The other one is implemented for fun - mask 2 non mask (MFR2 dataset from: https://github.com/aqeelanwar/MaskTheFace).

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

* Task1 - aerial map A2B (train for 200 epochs, decay at 100 epoch, -id=True):

| Aerial image | Generated map (pix2pix) | Generated map (cycleGAN) | Ground truth |
| ------------- | ------------- | ------------- | ------------- |
| <img src="https://github.com/Rayhchs/cyclegan-tensorflow-implementation/blob/main/test/3.jpg" width="250" title="3.jpg"> | <img src="https://github.com/Rayhchs/Pix2pix-tensorflow-implementation/blob/main/result/3.jpg" width="250">| <img src="https://github.com/Rayhchs/cyclegan-tensorflow-implementation/blob/main/result/3.jpg" width="250"> | <img src="https://github.com/Rayhchs/cyclegan-tensorflow-implementation/blob/main/test/label_3.jpg" width="250"> |
| <img src="https://github.com/Rayhchs/cyclegan-tensorflow-implementation/blob/main/test/6.jpg" width="250" title="6.jpg"> | <img src="https://github.com/Rayhchs/Pix2pix-tensorflow-implementation/blob/main/result/6.jpg" width="250"> | <img src="https://github.com/Rayhchs/cyclegan-tensorflow-implementation/blob/main/result/6.jpg" width="250"> | <img src="https://github.com/Rayhchs/cyclegan-tensorflow-implementation/blob/main/test/label_6.jpg" width="250"> |
| <img src="https://github.com/Rayhchs/cyclegan-tensorflow-implementation/blob/main/test/24.jpg" width="250" title="24.jpg"> | <img src="https://github.com/Rayhchs/Pix2pix-tensorflow-implementation/blob/main/result/24.jpg" width="250"> | <img src="https://github.com/Rayhchs/cyclegan-tensorflow-implementation/blob/main/result/24.jpg" width="250"> | <img src="https://github.com/Rayhchs/cyclegan-tensorflow-implementation/blob/main/test/label_24.jpg" width="250"> |
| <img src="https://github.com/Rayhchs/cyclegan-tensorflow-implementation/blob/main/test/33.jpg" width="250" title="33.jpg"> | <img src="https://github.com/Rayhchs/Pix2pix-tensorflow-implementation/blob/main/result/33.jpg" width="250"> | <img src="https://github.com/Rayhchs/cyclegan-tensorflow-implementation/blob/main/result/33.jpg" width="250"> | <img src="https://github.com/Rayhchs/cyclegan-tensorflow-implementation/blob/main/test/label_33.jpg" width="250"> |
| <img src="https://github.com/Rayhchs/cyclegan-tensorflow-implementation/blob/main/test/45.jpg" width="250" title="45.jpg"> | <img src="https://github.com/Rayhchs/Pix2pix-tensorflow-implementation/blob/main/result/45.jpg" width="250"> | <img src="https://github.com/Rayhchs/cyclegan-tensorflow-implementation/blob/main/result/45.jpg" width="250"> | <img src="https://github.com/Rayhchs/cyclegan-tensorflow-implementation/blob/main/test/label_45.jpg" width="250"> |

* Task1 - aerial map B2A:

| Google map | Generated map (cycleGAN) | Ground truth |
| ------------- | ------------- | ------------- |
| <img src="https://github.com/Rayhchs/cyclegan-tensorflow-implementation/blob/main/test/label_3.jpg" width="250" title="3.jpg"> | <img src="https://github.com/Rayhchs/cyclegan-tensorflow-implementation/blob/main/result/label_3.jpg" width="250">| <img src="https://github.com/Rayhchs/cyclegan-tensorflow-implementation/blob/main/test/3.jpg" width="250"> |
| <img src="https://github.com/Rayhchs/cyclegan-tensorflow-implementation/blob/main/test/label_6.jpg" width="250" title="3.jpg"> | <img src="https://github.com/Rayhchs/cyclegan-tensorflow-implementation/blob/main/result/label_6.jpg" width="250">| <img src="https://github.com/Rayhchs/cyclegan-tensorflow-implementation/blob/main/test/6.jpg" width="250"> |
| <img src="https://github.com/Rayhchs/cyclegan-tensorflow-implementation/blob/main/test/label_24.jpg" width="250" title="3.jpg"> | <img src="https://github.com/Rayhchs/cyclegan-tensorflow-implementation/blob/main/result/label_24.jpg" width="250">| <img src="https://github.com/Rayhchs/cyclegan-tensorflow-implementation/blob/main/test/24.jpg" width="250"> |
| <img src="https://github.com/Rayhchs/cyclegan-tensorflow-implementation/blob/main/test/label_33.jpg" width="250" title="3.jpg"> | <img src="https://github.com/Rayhchs/cyclegan-tensorflow-implementation/blob/main/result/label_33.jpg" width="250">| <img src="https://github.com/Rayhchs/cyclegan-tensorflow-implementation/blob/main/test/33.jpg" width="250"> |
| <img src="https://github.com/Rayhchs/cyclegan-tensorflow-implementation/blob/main/test/label_45.jpg" width="250" title="3.jpg"> | <img src="https://github.com/Rayhchs/cyclegan-tensorflow-implementation/blob/main/result/label_45.jpg" width="250">| <img src="https://github.com/Rayhchs/cyclegan-tensorflow-implementation/blob/main/test/45.jpg" width="250"> |

* Task2 - mask 2 non mask:
In task2, TsaiIngwen and XiJinping are used for testing, others are used for training (aprox. 167 images). Training setting: 1000 epochs, discriminator learning rate start decaying at epochs=500, -id=True.

| mask (President Tsai) | non mask (President Tsai) | mask (Mr. Xi) | non mask (Mr. Xi) |
| ------------- | ------------- | ------------- | ------------- |
| <img src="https://github.com/Rayhchs/cyclegan-tensorflow-implementation/blob/main/result/TsaiIngwen_0001.png" width="160"> | <img src="https://github.com/Rayhchs/cyclegan-tensorflow-implementation/blob/main/test/TsaiIngwen_0001.png" width="160">| <img src="https://github.com/Rayhchs/cyclegan-tensorflow-implementation/blob/main/test/XiJinping_0001.png" width="160" title="Pooh">| <img src="https://github.com/Rayhchs/cyclegan-tensorflow-implementation/blob/main/result/XiJinping_0001.png" width="160" title="Pooh"> |
| <img src="https://github.com/Rayhchs/cyclegan-tensorflow-implementation/blob/main/test/TsaiIngwen_0002.png" width="160"> | <img src="https://github.com/Rayhchs/cyclegan-tensorflow-implementation/blob/main/result/TsaiIngwen_0002.png" width="160">|<img src="https://github.com/Rayhchs/cyclegan-tensorflow-implementation/blob/main/result/XiJinping_0002.png" width="160" title="Pooh">| <img src="https://github.com/Rayhchs/cyclegan-tensorflow-implementation/blob/main/test/XiJinping_0002.png" width="160" title="Pooh"> |
| <img src="https://github.com/Rayhchs/cyclegan-tensorflow-implementation/blob/main/test/TsaiIngwen_0003.png" width="160"> | <img src="https://github.com/Rayhchs/cyclegan-tensorflow-implementation/blob/main/result/TsaiIngwen_0003.png" width="160">|<img src="https://github.com/Rayhchs/cyclegan-tensorflow-implementation/blob/main/test/XiJinping_0003.png" width="160" title="Pooh">| <img src="https://github.com/Rayhchs/cyclegan-tensorflow-implementation/blob/main/result/XiJinping_0003.png" width="160" title="Pooh"> |
| <img src="https://github.com/Rayhchs/cyclegan-tensorflow-implementation/blob/main/test/TsaiIngwen_0004.png" width="160"> | <img src="https://github.com/Rayhchs/cyclegan-tensorflow-implementation/blob/main/result/TsaiIngwen_0004.png" width="160">|<img src="https://github.com/Rayhchs/cyclegan-tensorflow-implementation/blob/main/test/XiJinping_0004.png" width="160" title="Pooh">| <img src="https://github.com/Rayhchs/cyclegan-tensorflow-implementation/blob/main/result/XiJinping_0004.png" width="160" title="Pooh"> |
| <img src="https://github.com/Rayhchs/cyclegan-tensorflow-implementation/blob/main/result/TsaiIngwen_0006.png" width="160"> | <img src="https://github.com/Rayhchs/cyclegan-tensorflow-implementation/blob/main/test/TsaiIngwen_0006.png" width="160">|<img src="https://github.com/Rayhchs/cyclegan-tensorflow-implementation/blob/main/result/XiJinping_0005.png" width="160" title="Pooh">| <img src="https://github.com/Rayhchs/cyclegan-tensorflow-implementation/blob/main/test/XiJinping_0005.png" width="160" title="Pooh"> |

## Discussion
* In task1 A2B, the generated results of cycleGAN are worse than results of pix2pix. This is intuitive because cycleGAN uses unpaired images. In contrast, generated results of cycleGAN seems more creative than pix2pix (see second(6.jpg) and last(45.jpg) generated map).

* In task2, the generated results are suck, however, it can be notice that the generators realize "mask" is crucial issue to deceive discriminators. In my opinion, the reason causing terrible results might be: (1) insufficient data, (2) insufficient training epoch (not sure) and (3) inappropriate decaying epoch (not sure). 

There is a better generated result from training data shown in below:
| mask | non mask |
| ------------- | ------------- |
| <img src="https://github.com/Rayhchs/cyclegan-tensorflow-implementation/blob/main/testA2B.jpg" width="160"> | <img src="https://github.com/Rayhchs/cyclegan-tensorflow-implementation/blob/main/o.jpg" width="160">|

## Acknowledgements
Code heavily borrows from [pix2pix](https://github.com/phillipi/pix2pix) and [pix2pix-tensorflow](https://github.com/yenchenlin/pix2pix-tensorflow). Thanks for the excellent work!

If you use this code for your research, please cite the original pix2pix paper:

	@inproceedings{CycleGAN2017,
	  title={Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks},
	  author={Zhu, Jun-Yan and Park, Taesung and Isola, Phillip and Efros, Alexei A},
	  booktitle={Computer Vision (ICCV), 2017 IEEE International Conference on},
	  year={2017}
	}


	@inproceedings{isola2017image,
	  title={Image-to-Image Translation with Conditional Adversarial Networks},
	  author={Isola, Phillip and Zhu, Jun-Yan and Zhou, Tinghui and Efros, Alexei A},
	  booktitle={Computer Vision and Pattern Recognition (CVPR), 2017 IEEE Conference on},
	  year={2017}
	}

## Reference
 
 https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
 
 https://junyanz.github.io/CycleGAN/
