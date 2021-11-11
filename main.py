import CoReD
import FT_Standard
import KD
import TGD
import argparse
from Function_common import Eval
import Logger
import os
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CONTINUAL LEARNING')
    parser.add_argument('--model', '-m', type=str, default='CoReD',choices=['CoReD','KD','TG','FT'], help='Choose the one of [\'CoReD\',\'KD\',\'TG\',\'FT\']')
    parser.add_argument('--test', '-te', help="turn on test mode", action="store_true")
    parser.add_argument('--name_sources', '-s', type=str, default='DeepFake', help='name of sources(more than one)(ex.DeepFake / DeepFake_Face2Face / DeepFake_Face2Face_FaceSwap) / used for Testset folder as well')
    parser.add_argument('--name_target', '-t', type=str, default='Face2Face', help='name of target(only one)(ex.DeepFake / Face2Face / FaceSwap) / used for Train only')
    parser.add_argument('--name_saved_folder1', '-folder1', type=str, default='CoReD', help='name of folder that will be made')
    parser.add_argument('--name_saved_folder2', '-folder2', type=str, default='', help='name of folder that will be made in folder1 (just option)')
    parser.add_argument('--path_data', '-d',type=str, default='./data/DeepFake', help='the folder of path must contains real/fake folders that is consists of images')
    parser.add_argument('--path_preweight', '-w', '-path', type=str, default='./weights', help='the folder of path must source(s) folder that have model weights')

    parser.add_argument('--lr', '-l', type=float, default=0.05, help='initial learning rate')
    parser.add_argument('--KD_alpha', '-a', type=float, default=0.5, help='KD alpha')
    parser.add_argument('--num_class', '-nc', type=int, default=2, help='number of classes')
    parser.add_argument('--num_store', '-ns', type=int, default=5, help='number of stores')
    parser.add_argument('--epochs', '-me', type=int, default=100, help='epochs')
    parser.add_argument('--batch_size', '-nb', type=int, default=128, help='batch size')
    parser.add_argument('--num_gpu', '-ng', type=str, default='2', help='excuted gpu number')
    args = parser.parse_args()

    if not args.test:
        if args.model == 'CoReD':
            run_module = CoReD
        elif args.model == 'KD':
            run_module = KD
        elif args.model == 'TG':
            run_module = TGD
        elif args.model == 'FT':
            run_module = FT_Standard
        else:
            print("CAN NOT EXCUTE !")
            exit()
        run_module.Train(args)
    else:
        log = Logger.Logger()
        log_dir = './log'
        os.makedirs(log_dir, exist_ok=True)
        log.open(log_dir + f'/[eval_yt]GAN_script2_v2.txt', mode='a')
        log.write('\n')
        Eval(args, log)