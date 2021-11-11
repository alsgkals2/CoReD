import os
import os.path
import numpy as np
import torch
import torch.nn as nn
import shutil
import random
from torch.nn import functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.datasets as datasets
from PIL import Image
import xception_origin

def set_seeds(seed=2020):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

class CustumDataset(Dataset):
    def __init__(self, data, target, transform=None):
        self.data = data
        self.target = target
        self.transform = transform
    
    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        path = self.data[idx]
        img = Image.open(path)
        img = img.convert('RGB')

        if self.transform:
            img = self.transform(img)
        return img, self.target[idx]

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
def save_checkpoint(state, checkpoint,
                    best_filename = 'student_model_best_acc.pt',
                    isAcc=False
                    ):
    if isAcc:
        torch.save(state, os.path.join(checkpoint,best_filename))
     
def Make_DataLoader(rootpath_dataset,
                    name_source,
                    name_target,
                    name_mixed_folder='',
                    train_aug=None,
                    val_aug=None,
                    mode_CoReD = False,
                    batch_size=128,
                    TRAIN_MODE=True
                    ):
    dic_CoReD = None
    if TRAIN_MODE:
        val_target_dir_mixed=''
        val_target_loader_mixed=None
        print('param check')
        print(rootpath_dataset,name_source, name_target, name_mixed_folder)
        if name_mixed_folder :
            train_dir = os.path.join(rootpath_dataset+'/TransferLearning',name_mixed_folder+'/train/')
            print(train_dir)
        else :
            train_dir = os.path.join(rootpath_dataset+'/TransferLearning',name_target+'/train/')
        #For Validataion
        source_dataset = os.path.join(rootpath_dataset,name_source)
        target_dataset = os.path.join(rootpath_dataset,name_target)
        val_source_dir = os.path.join(source_dataset, 'val')
        if name_mixed_folder :
            target_dataset_mix = os.path.join(rootpath_dataset.replace('CLRNet_jpg25', ''), name_target)
            val_target_dir = os.path.join(target_dataset, 'val')
            val_target_dir_mixed = os.path.join(target_dataset_mix,'val')
        else:
            val_target_dir = os.path.join(target_dataset, 'val')

        #check the paths
        print("DATASET PATHS")
        print(train_dir)
        print(val_source_dir)
        print(val_target_dir)

        #check existing of folders
        if not(os.path.exists(train_dir) and os.path.exists(val_source_dir) and os.path.exists(val_target_dir)) :
            print("check the paths")
            return -1
        elif name_mixed_folder:
            print('oteher mixed path is : {}'.format(val_target_dir_mixed))
            if not(os.path.exists(val_target_dir_mixed)):
                print("check the paths of mix dataset")
                return -1

        train_target_loader, train_target_loader_forcorrect = None,None
        train_target_dataset = datasets.ImageFolder(train_dir,transform=None)
        train_target_dataset = CustumDataset(np.array(train_target_dataset.samples)[:,0],np.array(train_target_dataset.targets),train_aug)
        train_target_loader = DataLoader(train_target_dataset,
                                        batch_size=batch_size,
                                        shuffle=True,
                                        num_workers=8,
                                        pin_memory=True
                                        )

        if mode_CoReD : train_target_loader_forcorrect = DataLoader(train_target_dataset,
                                                                    batch_size=batch_size,
                                                                    shuffle=False,
                                                                    num_workers=8,
                                                                    pin_memory=True
                                                                    )
        val_target_loader = DataLoader(datasets.ImageFolder(val_target_dir, val_aug),
                                    batch_size=batch_size,
                                    shuffle=True,
                                    num_workers=4,
                                    pin_memory=True
                                    )
        val_source_loader = DataLoader(datasets.ImageFolder(val_source_dir, val_aug),
                                    batch_size=batch_size,
                                    shuffle=True,
                                    num_workers=4,
                                    pin_memory=True
                                    )
        if name_mixed_folder:
            val_target_loader_mixed = DataLoader(datasets.ImageFolder(val_target_dir_mixed, val_aug),
                                                batch_size=batch_size,
                                                shuffle=True,
                                                num_workers=4,
                                                pin_memory=True
                                                )
        dic = {'train_target': train_target_loader, 'val_source': val_source_loader,'val_target': val_target_loader, 'val_target_mix':val_target_loader_mixed}
        dic_CoReD = {'train_target_dataset':train_target_dataset ,'train_target_forCorrect':train_target_loader_forcorrect}
    
    else: #Test mode
        # dir_testset = os.path.join(rootpath_dataset,name_target)
        dir_testset = os.path.join(rootpath_dataset, 'test')
        loader_test = DataLoader(datasets.ImageFolder(dir_testset, val_aug),
                                                batch_size=batch_size,
                                                shuffle=True,
                                                num_workers=4,
                                                pin_memory=True
                                                )
        dic = {'test_dataset':loader_test}
    return dic, dic_CoReD

def Make_DataLoader_continual(rootpath_dataset,
                              name_source,
                              name_target,
                              name_source2='',
                              name_source3='',
                              name_mixed_folder='',
                              train_aug=None,
                              val_aug=None,
                              mode_CoReD = False,
                              batch_size=128,
                              TRAIN_MODE = True
                              ):
    if TRAIN_MODE:
        train_dir = os.path.join(rootpath_dataset+'/TransferLearning', '{}/train/'.format(name_target))
        val_source_loader2,val_source_loader3 = None,None
        #For Validataion
        val_target_loader_mixed=None
        val_target_dir_MIXED = ''
        source_dataset = os.path.join(rootpath_dataset,name_source)
        source_dataset2 = os.path.join(rootpath_dataset,name_source2)
        source_dataset3 = os.path.join(rootpath_dataset,name_source3) #if name_source3 else None
        target_dataset = os.path.join(rootpath_dataset,name_target)
        val_source_dir = os.path.join(source_dataset, 'val')
        val_source_dir2 = os.path.join(source_dataset2, 'val')
        val_source_dir3 = os.path.join(source_dataset3, 'val')
        val_target_dir = os.path.join(target_dataset, 'val')
        #check the paths
        if name_mixed_folder :
            target_dataset_mix = os.path.join(rootpath_dataset.replace('CLRNet_jpg25', ''), name_target)
            val_target_dir = os.path.join(target_dataset, 'val')
            val_target_dir_MIXED = os.path.join(target_dataset_mix,'val')

        if not(os.path.exists(train_dir) and os.path.exists(val_source_dir) and os.path.exists(val_target_dir)) :
            print("check the paths")

        print("DATASET PATHS")
        print('val_source_dir ' ,val_source_dir)
        print('val_source_dir2 ',val_source_dir2)
        print('val_source_dir3 ',val_source_dir3)
        print('val_target_dir ' ,val_target_dir)
        print('train_dir ' ,train_dir)
                    
        train_target_loader, train_target_loader_forcorrect = None,None
        train_target_dataset = datasets.ImageFolder(train_dir,transform=None)
        train_target_dataset = CustumDataset(np.array(train_target_dataset.samples)[:,0],np.array(train_target_dataset.targets),train_aug)
        train_target_loader = DataLoader(train_target_dataset,
                                        batch_size=batch_size,
                                        shuffle=True,
                                        num_workers=4,
                                        pin_memory=True
                                        )
        if mode_CoReD : train_target_loader_forcorrect = DataLoader(train_target_dataset,
                                                                    batch_size=batch_size,
                                                                    shuffle=False,
                                                                    num_workers=4,
                                                                    pin_memory=True
                                                                    )

        val_target_loader = DataLoader(datasets.ImageFolder(val_target_dir, val_aug),
                                    batch_size=batch_size,
                                    shuffle=True,
                                    num_workers=4,
                                    pin_memory=True
                                    )
        val_source_loader = DataLoader(datasets.ImageFolder(val_source_dir, val_aug),
                                    batch_size=batch_size,
                                    shuffle=True,
                                    num_workers=4,
                                    pin_memory=True
                                    )

        if name_source2:
            val_source_loader2 = DataLoader(datasets.ImageFolder(val_source_dir2, val_aug),
                                            batch_size=batch_size,
                                            shuffle=True,
                                            num_workers=4,
                                            pin_memory=True
                                            )
        if name_source3:
            val_source_loader3 = DataLoader(datasets.ImageFolder(val_source_dir3, val_aug),
                                            batch_size=batch_size,
                                            shuffle=True,
                                            num_workers=4,
                                            pin_memory=True
                                            )

            print(rootpath_dataset,name_source, name_target, name_mixed_folder)
        if name_mixed_folder :
            train_dir = os.path.join(rootpath_dataset+'/TransferLearning',name_mixed_folder+'/train/')
            print(train_dir)
        else :
            train_dir = os.path.join(rootpath_dataset+'/TransferLearning',name_target+'/train/')
            
        if name_mixed_folder:
            val_target_loader_mixed = DataLoader(datasets.ImageFolder(val_target_dir_MIXED, val_aug),
                                                batch_size=batch_size,
                                                shuffle=True,
                                                num_workers=4,
                                                pin_memory=True
                                                )

        dic = {'train_target':train_target_loader,'val_source':val_source_loader,'val_source2':val_source_loader2,'val_source3':val_source_loader3,'val_target':val_target_loader,'val_target_mix':val_target_loader_mixed}
        dic_CoReD = {'train_target_dataset':train_target_dataset ,'train_target_forCorrect':train_target_loader_forcorrect}
    else: 
        dic_CoReD = None
        dir_testset = os.path.join(rootpath_dataset,'test')
        loader_test = DataLoader(datasets.ImageFolder(dir_testset, val_aug),
                                                batch_size=batch_size,
                                                shuffle=False,
                                                num_workers=4,
                                                pin_memory=True
                                                )

        dic = {'test_dataset':loader_test}
    return dic, dic_CoReD

def Make_DataLoader_togeter(rootpath_dataset,
                            name_source,
                            name_target,
                            name_target2,
                            train_aug=None,
                            val_aug=None,
                            mode_CoReD=False,
                            batch_size=128
                            ):
    train_dir = os.path.join(rootpath_dataset ,'TransferLearning', name_target , 'train/')
    train_dir2 = os.path.join(rootpath_dataset ,'TransferLearning', name_target2 , 'train/')

    # For Validataion
    source_dataset = os.path.join(rootpath_dataset, name_source)
    target_dataset = os.path.join(rootpath_dataset, name_target)
    target_dataset2 = os.path.join(rootpath_dataset, name_target2)
    val_source_dir = os.path.join(source_dataset, 'val')
    val_target_dir = os.path.join(target_dataset, 'val')
    val_target_dir2 = os.path.join(target_dataset2, 'val')

    # check the paths
    if not (os.path.exists(train_dir) and os.path.exists(train_dir2) and os.path.exists(val_source_dir) and os.path.exists(val_target_dir)):
        print("check the paths")
        return

    print("DATASET PATHS")
    print('val_source_dir ', val_source_dir)
    print('val_target_dir ', val_target_dir)
    print('val_target_dir2 ', val_target_dir2)
    print('train_dir ', train_dir)
    print('train_dir2 ', train_dir2)

    train_target_loader, train_target_loader_forcorrect = None, None
    train_target_dataset = datasets.ImageFolder(train_dir, transform=None)
    train_target_dataset2 = datasets.ImageFolder(train_dir2, transform=None)

    train_target_dataset = CustumDataset(np.concatenate((np.array(train_target_dataset.samples)[:, 0],np.array(train_target_dataset2.samples)[:, 0])),
                                         np.concatenate((np.array(train_target_dataset.targets),np.array(train_target_dataset2.targets))), train_aug)
    train_target_loader = DataLoader(train_target_dataset,
                                     batch_size=batch_size,
                                     shuffle=True,
                                     num_workers=4,
                                     pin_memory=True
                                     )
    if mode_CoReD:
        train_target_loader_forcorrect = DataLoader(train_target_dataset,
                                                                batch_size=batch_size,
                                                                shuffle=False,
                                                                num_workers=4,
                                                                pin_memory=True
                                                                )

    val_target_loader = DataLoader(datasets.ImageFolder(val_target_dir, val_aug),
                                   batch_size=batch_size,
                                   shuffle=True,
                                   num_workers=4,
                                   pin_memory=True
                                   )
    val_target_loader2 = DataLoader(datasets.ImageFolder(val_target_dir2, val_aug),
                                   batch_size=batch_size,
                                    shuffle=True,
                                    num_workers=4,
                                    pin_memory=True
                                    )
    val_source_loader = DataLoader(datasets.ImageFolder(val_source_dir, val_aug),
                                    batch_size=batch_size,
                                   shuffle=True,
                                   num_workers=4,
                                   pin_memory=True
                                   )

    dic = {'train_target': train_target_loader, 'val_source': val_source_loader,
           'val_target': val_target_loader,'val_target2': val_target_loader2}
    dic_CoReD = {'train_target_dataset': train_target_dataset,
                  'train_target_forCorrect': train_target_loader_forcorrect}
    return dic, dic_CoReD


    
def Test(val_loader, model, criterion, log = None):
    global best_acc
    correct, total =0,0
    losses = AverageMeter()
    arc = AverageMeter()
    main_losses = AverageMeter()
    model.eval()
    model.cuda()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            loss_main = criterion(outputs, targets)
            loss = loss_main
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == targets).sum().item()
            total += len(targets)
            losses.update(loss.data.tolist(), inputs.size(0))
            main_losses.update(loss_main.tolist(), inputs.size(0))
    if log:
        log.write('Test | Loss:{loss:.4f} | MainLoss:{main:.4f} | top:{top:.4f}'.format(loss=losses.avg, main=main_losses.avg, top = correct/total*100)+ ' \n')
    else: 
        print('Test | Loss:{loss:.4f} | MainLoss:{main:.4f} | top:{top:.4f}'.format(loss=losses.avg, main=main_losses.avg, top = correct/total*100))
    return (losses.avg, arc.avg, correct/total*100)

def Eval(args,log):
    dicLoader,_, _ = initialization(args)
    model_list=[]
    weight_path = args.path_preweight
    if os.path.isdir(weight_path): 
        for a,b,c in os.walk(args.path_preweight):
            for _c in c:
                if '.pth.tar' in _c:
                    fullpath = os.path.join(a,_c)
                    model_list.append(fullpath)
    elif os.path.isfile(weight_path):
        model_list.append(weight_path)

    for model_item in model_list:
        _, student_model = load_models(model_item, log, args.test)
        criterion = nn.CrossEntropyLoss().cuda()
        _, _, _ = Test(dicLoader['test_dataset'], student_model, criterion, log)
    _, _, _ = Test(dicLoader['source_dataset'], student_model, criterion)

#LOSS-------------------
def loss_fn_kd(outputs, labels, teacher_outputs, KD_T=20, KD_alpha=0.5):
    KD_loss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(outputs/KD_T,dim=1),
                             F.softmax(teacher_outputs/KD_T,dim=1) * KD_alpha*KD_T*KD_T) +\
        F.cross_entropy(outputs, labels) * (1. - KD_alpha)
    return KD_loss

def get_augs():
    train_aug = transforms.Compose([
        transforms.Resize((128,128)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
    ])

    val_aug = transforms.Compose([
        transforms.Resize((128,128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
    ])

    return train_aug, val_aug


def load_models(path_preweight, name_sources, MakeTeacher=True):
    teacher_model, student_model = None,None
    path_preweight = os.path.join(path_preweight,name_sources)
    if MakeTeacher:
        teacher_model = xception_origin.xception(num_classes=2, pretrained='')
        checkpoint =torch.load(path_preweight+'/model_best_accuracy.pth.tar')
        teacher_model.load_state_dict(checkpoint['state_dict'])
        teacher_model.eval(); teacher_model.cuda()

    student_model = xception_origin.xception(num_classes=2,pretrained='')
    checkpoint =torch.load(path_preweight+'/model_best_accuracy.pth.tar')
    student_model.load_state_dict(checkpoint['state_dict'])
    student_model.train(); student_model.cuda()

    return teacher_model, student_model


def initialization(args):
    print('GPU num is' , args.num_gpu)
    os.environ['CUDA_VISIBLE_DEVICES'] =str(args.num_gpu)
    set_seeds()
    dict_source = {}
    name_sources = args.name_sources
    name_target = args.name_target
    path_data = args.path_data
    dict_source['source']= name_sources
    if '_' in name_sources:
        temp = name_sources.split('_')
        dict_source['source'] = temp[0]
        dict_source['source2'] = temp[1]
        try:
            dict_source['source3'] = temp[2]
        except:
            print('source3 is empty')

    print('--------------name--------------')
    print('source : ',dict_source['source'])
    if 'source2' in dict_source : print('source2 : ',dict_source['source2'])
    if 'source3' in dict_source: print('source3 :',dict_source['source3'])
    print('target : ',name_target)

    #train & valid
    train_aug, val_aug = get_augs()
    print(path_data,dict_source['source'],name_target)
    if '_' not in name_sources: #Task1 (pre-train before continual learning) or Test mode
        dicLoader,dicCoReD = Make_DataLoader(path_data, dict_source['source'], name_target, train_aug=train_aug, val_aug=val_aug,mode_CoReD=True, batch_size=args.batch_size, TRAIN_MODE=not args.test)
    else: #Task2-4 (continual learning)
        dicLoader,dicCoReD = Make_DataLoader_continual(path_data, name_source=dict_source['source'], name_target=name_target, train_aug=train_aug, val_aug=val_aug, mode_CoReD=True, batch_size=args.batch_size, TRAIN_MODE=not args.test)
    return dicLoader, dicCoReD, dict_source
    
def save_checkpoint(state, checkpoint='checkpoint', filename='checkpoint.pth.tar' , AUC_BEST = False, ACC_BEST = False):
    name_save = ''
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if AUC_BEST :
        name_save = 'model_best.pth.tar'
        shutil.copyfile(filepath, os.path.join(checkpoint, name_save))
    if ACC_BEST :
        name_save = 'model_best_accuracy.pth.tar'
        shutil.copyfile(filepath, os.path.join(checkpoint, name_save))