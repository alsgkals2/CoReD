
import CoReD
import FT_Standard
import KD
import TGD
import Vanilla

import argparse
from Function_common import Eval
import Logger
import os
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CONTINUAL LEARNING')
    parser.add_argument('--model', '-m', type=str, default='CoReD',choices=['CoReD','KD','TG','FT'], help='Choose the one of [CoReD, KD, TG, FT]')
    # parser.add_argument('--network', '-n', type=str, default='Efficient',choices=['Xception','Efficient'], help='Choose the one of [Xception, Efficient]')
    parser.add_argument('--network', '-n', type=str, default='Xception',choices=['Xception','Efficient'], help='Choose the one of [Xception, Efficient]')
    parser.add_argument('--test', '-te', help="turn on test mode", action="store_true")
    parser.add_argument('--fully_train', '-f', help="False on fully training", action="store_true")
    parser.add_argument('--name_sources', '-s', type=str, default='', help='name of sources(more than one)(ex.DeepFake / DeepFake_Face2Face / DeepFake_Face2Face_FaceSwap) / used for Testset folder as well')
    parser.add_argument('--name_target', '-t', type=str, default='FaceSwap', help='name of target(only one)(ex.DeepFake / Face2Face / FaceSwap) / used for Train only')
    parser.add_argument('--name_folder1', '-folder1', type=str, default='CoReD', help='name of folder that will be made')
    parser.add_argument('--name_folder2', '-folder2', type=str, default='', help='name of folder that will be made in folder1 (just option)')
    parser.add_argument('--data', '-d',type=str, default='/media/data1/mhkim/CLRNet_jpg25', help='the folder of path must contains Sources & Target folder names')
    parser.add_argument('--weigiht', '-w', '-path', type=str, default='', help='You can select the full path or folder path included in the file \'.pth\'')

    parser.add_argument('--lr', '-l', type=float, default=0.05, help='initial learning rate')
    parser.add_argument('--KD_alpha', '-a', type=float, default=0.5, help='KD')
    parser.add_argument('--num_class', '-nc', type=int, default=2, help='number of classes')
    parser.add_argument('--num_store', '-ns', type=int, default=5, help='number of stores')
    parser.add_argument('--epochs', '-me', type=int, default=100, help='epochs')
    parser.add_argument('--batch_size', '-nb', type=int, default=128, help='batch size')
    parser.add_argument('--num_gpu', '-ng', type=str, default='2', help='excuted gpu number')
    args = parser.parse_args()

    log = Logger.Logger()
    args.folder_weight = f'/media/data1/mhkim/CoReD_weights/{args.name_sources}_{args.name_target}/{args.name_folder1}/{args.name_folder2}/'
    log_dir = f'{args.folder_weight}/log'
    os.makedirs(log_dir, exist_ok=True)
    if not args.test:
        if args.model == 'CoReD':
            run_module = CoReD
        elif args.model == 'KD':
            run_module = KD
        elif args.model == 'TG':
            run_module = TGD
        elif args.model == 'FT':
            run_module = FT_Standard
        elif args.model == 'PRETRAIN':
            run_module = Vanilla
        else:
            print("CAN NOT EXCUTE !")
            exit()
        # Train(args)
        name_file = f'[Train]{args.name_sources}_{args.name_target}.txt'
        log.open(os.path.join(log_dir,name_file), mode='w')
        log.write('\n')
        run_module.Train(args, log)
    else:
        name_file = f'[Eval]{args.name_sources}_{args.name_target}.txt'
        log.open(os.path.join(log_dir,name_file), mode='w')
        log.write('\n')
        Eval(args, log)