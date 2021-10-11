"""
Created on Moon Festival + 1 2021

model: utils

@author: Ray
"""
from argparse import ArgumentParser, RawTextHelpFormatter
import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from cyclegan import *
from utils import *

parser = ArgumentParser(usage=None, formatter_class=RawTextHelpFormatter, description="Image translation using CycleGAN: \n \n"
    "This code provides a cycleGAN model from training to testing. "
    "Users can apply it for image translation tasks. \n"
    "------------------------------ Parameters ------------------------------ \n"
    "Epoch: default=400 \n"
    "Batch size: default=1 \n"
    "Save path: Path to save testing result, default='.\result' \n")

parser.add_argument("mode", help="train or test")
parser.add_argument("-e", "--epoch", type=int, default=400, dest="epoch")
parser.add_argument("-b", "--batch_size", type=int, default=1, dest="batch_size")
parser.add_argument("-s", "--save_path", type=str, default=None, dest="save_path")
parser.add_argument("-lc", "--lambda_cc", type=float, default=10.0, dest="lambda_c")
parser.add_argument("-d", "--decay", type=int, default=100, dest="decay")
parser.add_argument("-id", "--do_idt_loss", type=bool, default=False, dest="do_idt")
parser.add_argument("-r", "--do_resize", type=bool, default=False, dest="do_resize")

args = parser.parse_args()

def main():
    if args.mode.lower() == 'train':
        A_folder = input("Please input dir containing A domain images: ")
        A_paths = load_path(A_folder)
        sys.exit("There is no image in this dir") if A_paths == [] else None

        B_folder = input("Please input dir containing B domain images: ")
        B_paths = load_path(B_folder)
        sys.exit("There is no image in this dir") if B_paths == [] else None

        # Make A and B with same number of images
        A_paths, B_paths = fill_data(A_paths, B_paths)

        with tf.Session() as sess:
            model = cyclegan(sess, args)
            model.train(A_paths, B_paths)

    elif args.mode.lower() == 'test':
        folder = input("Please input dir containing images: ")
        paths = load_path(folder)
        sys.exit("There is no image in this dir") if paths == [] else None

        style = input("B2A or A2B: ")
        sys.exit("There is no image in this dir") if style.lower() not in ["b2a", "a2b"] else None

        with tf.Session() as sess:
            model = cyclegan(sess, args)
            g_imgs = model.test(paths, style)
        output_path = save_data(g_imgs, paths, args.save_path)
        print('Saved in {}'.format(output_path))

    else:
        sys.exit("Incorrect mode")

if __name__ == '__main__':
    main()