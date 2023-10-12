import os
import os.path
import numpy as np
import torch
import torch.nn as nn
import tqdm
import random
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.datasets as datasets
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from collections import OrderedDict
from PIL import Image
import xception_origin
from EfficientNet import *
import copy
import tqdm

def set_seeds(seed=2020):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

class CustumDataset(Dataset):
    def __init__(self, data, target, transform=None):
        #for debugging
        self.data = data[:400]
        self.target = target[:400]

        # self.data = data
        # self.target = target
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
        return img, int(self.target[idx])

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int32(W * cut_rat)
    cut_h = np.int32(H * cut_rat)

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


def initialization(args):
    print('GPU num is' , args.num_gpu)
    os.environ['CUDA_VISIBLE_DEVICES'] =str(args.num_gpu)
    set_seeds()
    # dict_source = {}
    dict_source = OrderedDict()
    name_sources = args.name_sources
    name_target = args.name_target
    path_data = args.data
    if not name_sources : dict_source['source'] = name_sources #TASK 0
    else: #TASK 1 ~
        temp = name_sources.split('_')
        cnt=1
        for _name in temp:
            dict_source[f'source{cnt}'] = _name
            print(f'Source Name : {_name}')
            cnt+=1
    if name_target : print('Target Name : ',name_target)

    #train & valid
    train_aug, val_aug = get_augs()
    if not name_sources: #Task1 (pre-train before continual learning) or Test mode
        if dict_source['source']:
            print("path_data, dict_source['source']")
            print(path_data, dict_source['source'])
        dicLoader,dicCoReD = Make_DataLoader(path_data, dict_source['source'],
                                            name_target,
                                            train_aug=train_aug,
                                            val_aug=val_aug,
                                            mode_CoReD=False,
                                            batch_size=args.batch_size,
                                            TRAIN_MODE=not args.test,
                                            MODE_BALANCED_DATA=False,
                                            fully_train=args.fully_train)
    else: #Task2-4 (continual learning)
        print('===> Making Loader for Continual Learning..')
        dicLoader,dicCoReD = Make_DataLoader_continual(path_data, name_source=dict_source,
                                                    name_target=name_target,
                                                    train_aug=train_aug,
                                                    val_aug=val_aug,
                                                    mode_CoReD=True,
                                                    batch_size=args.batch_size,
                                                    TRAIN_MODE=not args.test,
                                                    fully_train=args.fully_train)
    return dicLoader, dicCoReD, dict_source
    

def Make_DataLoader(dir,
                    name_source,
                    name_target,
                    name_mixed_folder='',
                    train_aug=None,
                    val_aug=None,
                    mode_CoReD = False,
                    batch_size=128,
                    TRAIN_MODE=True,
                    MODE_BALANCED_DATA = False,
                    fully_train = False

                    ):
    dic_CoReD = None
    if TRAIN_MODE:
        if name_mixed_folder :
            MIXED_MODE = True
        else :
            MIXED_MODE = False
        val_target_loader_mixed = None
        val_source_loader = None
        val_target_dir_mixed = ''
        val_source_dir = ''
        val_target_dir = ''
        print('param check')
        print(dir, name_source, name_target, name_mixed_folder)
        name_last_folder = '{}/train/'.format(name_target if not name_mixed_folder else name_mixed_folder)
        _name = 'TransferLearning' if not fully_train else ''
        train_dir = os.path.join(dir, _name if name_source else '', name_last_folder)
        print(train_dir)
        #For Validataion
        if name_source:
            source_dataset = os.path.join(dir,name_source)
            val_source_dir = os.path.join(source_dataset, 'val')
        target_dataset = os.path.join(dir,name_target)
        val_target_dir = os.path.join(target_dataset, 'val')

        #check the paths
        print("DATASET PATHS")
        print(train_dir)
        print(val_source_dir)
        print(val_target_dir)

        #check existing of folders
        assert(os.path.exists(train_dir) and os.path.exists(val_source_dir) and os.path.exists(val_target_dir), '===> CHECK THE PATHS')

        if MIXED_MODE:
            print('oteher mixed path is : {}'.format(val_target_dir_mixed))
            assert(os.path.exists(val_target_dir_mixed), '===> CHECK the paths of mix dataset')

        train_target_loader, train_target_loader_forcorrect = None,None
        train_target_dataset = datasets.ImageFolder(train_dir,transform=None)
        print(len(train_target_dataset))
        train_target_dataset = CustumDataset(np.array(train_target_dataset.samples)[:,0], np.array(train_target_dataset.targets), train_aug)
        
        train_target_loader = DataLoader(train_target_dataset,
                                        batch_size=batch_size,
                                        shuffle=True,
                                        num_workers=8,
                                        pin_memory=True
                                        )

        if mode_CoReD :
            train_target_loader_forcorrect = DataLoader(train_target_dataset,
                                                        batch_size=batch_size,
                                                        shuffle=False,
                                                        num_workers=8,
                                                        pin_memory=True
                                                        )
        val_target_loader = DataLoader(datasets.ImageFolder(val_target_dir, val_aug),
                                    batch_size=batch_size,
                                    shuffle=False,
                                    num_workers=4,
                                    pin_memory=True
                                    )
        if val_source_dir:
            print(val_source_dir)
            val_source_loader = DataLoader(datasets.ImageFolder(val_source_dir, val_aug),
                                        batch_size=batch_size,
                                        shuffle=False,
                                        num_workers=4,
                                        pin_memory=True
                                        )
        if name_mixed_folder:
            val_target_loader_mixed = DataLoader(datasets.ImageFolder(val_target_dir_mixed, val_aug),
                                                batch_size=batch_size,
                                                shuffle=False,
                                                num_workers=4,
                                                pin_memory=True
                                                )
        dic = {'train_target': train_target_loader, 'val_source': val_source_loader,'val_target': val_target_loader, 'val_target_mix':val_target_loader_mixed}
        dic_CoReD = {'train_target_dataset':train_target_dataset ,'train_target_forCorrect':train_target_loader_forcorrect}
        print("train_target_loader_forcorrect")
        print(train_target_loader_forcorrect)
    else: #Test mode
        test_dir = os.path.join(dir, name_target if name_target else '', 'test')
        #temp
        print("dir??")
        print(dir)
        train_target_dataset = datasets.ImageFolder(test_dir, transform=None)
        new_samples = train_target_dataset.samples

        if MODE_BALANCED_DATA:
            random.shuffle(train_target_dataset.samples)
            num_fake = train_target_dataset.targets[train_target_dataset.targets==1]
            for idx in range(len(train_target_dataset.targets)):
                _info = train_target_dataset.samples[idx]
                if len(new_samples) is not num_fake:
                    new_samples.append(_info)
                elif _info[1] == 1:
                    new_samples.append(_info)

        train_target_dataset = CustumDataset(np.array(new_samples)[:,0], np.array(new_samples)[:,1], val_aug)
        train_target_loader = DataLoader(train_target_dataset,
                                        batch_size=batch_size,
                                        shuffle=True,
                                        num_workers=8,
                                        pin_memory=True
                                        )
        dic = {'test_dataset':train_target_loader}

    return dic, dic_CoReD

def Make_DataLoader_continual(dir,
                              name_source,
                              name_target='',
                              name_mixed_folder='',
                              mode_CoReD = True,
                              train_aug=None,
                              val_aug=None,
                              batch_size=128,
                              TRAIN_MODE = True,
                              fully_train = False
                              ):
    name_last_folder = '{}/train/'.format(name_target if not name_mixed_folder else name_mixed_folder)
    _name = 'TransferLearning' if not fully_train else ''
    train_dir = os.path.join(dir, _name if name_source else '', name_last_folder)
        # train_dir = os.path.join(dir+'/TransferLearning' if name_source else '', '{}/train/'.format(name_target if not name_mixed_folder else name_mixed_folder))
    #For Validataion
    print(name_source)
    val_target_loader_mixed=None
    val_target_dir_MIXED = ''
    source_dataset=[]
    val_source_dir = []
    for _item in name_source:
        source_dataset.append(os.path.join(dir,name_source[_item]))
        val_source_dir.append(os.path.join(source_dataset[-1], 'val'))
        print(val_source_dir,'----------')
    target_dataset = os.path.join(dir,name_target)
    val_target_dir = os.path.join(target_dataset, 'val')
    #check the paths
    if name_mixed_folder :
        target_dataset_mix = os.path.join(dir.replace('CLRNet_jpg25', ''), name_target)
        val_target_dir_MIXED = os.path.join(target_dataset_mix,'val')


    val_source_loader = []
    cnt = 1
    NUM_WORKIER = 12
    for dir in val_source_dir:
            print('===> Making Loader :', dir)
            _loader = DataLoader(datasets.ImageFolder(dir, val_aug),
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=NUM_WORKIER,
                                pin_memory=True
                                )
            val_source_loader.append(copy.deepcopy(_loader))

    if TRAIN_MODE:
        print("DATASET PATHS")
        print('val_source_dir ' ,val_source_dir)
        print('val_target_dir ' ,val_target_dir)
        print('train_dir ' ,train_dir)
        train_target_loader, train_target_loader_forcorrect = None,None
        train_target_dataset = datasets.ImageFolder(train_dir, transform=None)
        train_target_dataset = CustumDataset(np.array(train_target_dataset.samples)[:,0],np.array(train_target_dataset.targets),train_aug)
        train_target_loader = DataLoader(train_target_dataset,
                                        batch_size=batch_size,
                                        shuffle=True,
                                        num_workers=NUM_WORKIER,
                                        pin_memory=True
                                        )
        assert (os.path.exists(train_dir) and os.path.exists(val_source_dir[0]) and os.path.exists(val_target_dir),'Check Path !!!')

        if mode_CoReD :
            train_target_loader_forcorrect = DataLoader(train_target_dataset,
                                                        batch_size=batch_size,
                                                        shuffle=False,
                                                        num_workers=NUM_WORKIER,
                                                        pin_memory=True
                                                        )

        val_target_loader = DataLoader(datasets.ImageFolder(val_target_dir, val_aug),
                                    batch_size=batch_size,
                                    shuffle=False,
                                    num_workers=NUM_WORKIER,
                                    pin_memory=True
                                    )
            
        if name_mixed_folder:
            val_target_loader_mixed = DataLoader(datasets.ImageFolder(val_target_dir_MIXED, val_aug),
                                                batch_size=batch_size,
                                                shuffle=False,
                                                num_workers=NUM_WORKIER,
                                                pin_memory=True
                                                )
        dic = OrderedDict()
        dic['train_target'] = train_target_loader;
        dic['val_target'] = val_target_loader
        if val_target_loader_mixed : dic['val_target_mix'] = val_target_loader_mixed

        for _loader in val_source_loader:
            dic[f'val_dataset{cnt}'] = _loader
            cnt += 1
        dic_CoReD = {'train_target_dataset':train_target_dataset ,'train_target_forCorrect':train_target_loader_forcorrect}
    else:
        dic = OrderedDict()
        for _loader in val_source_loader:
            print("sdfsdfsdf")
            dic[f'test_dataset{cnt}'] = _loader
            cnt += 1
        dic_CoReD = None

    return dic, dic_CoReD

def Test_PRF(test_loader, model, criterion, log=None): # precision/recall/f1-score
    losses = AverageMeter()
    acc_real = AverageMeter()
    acc_fake = AverageMeter()
    sum_of_AUROC=[]
    target=[]
    output = []

    y_true=np.zeros((0,2),dtype=np.int8)
    y_pred=np.zeros((0,2),dtype=np.int8)
    print(len(test_loader.dataset))
    with torch.no_grad():
        model.eval()
        model.cuda()
        for batch_idx, (inputs, targets) in tqdm.tqdm(enumerate(test_loader)):
            inputs, targets = inputs.cuda(),torch.from_array(np.array(targets)).cuda()
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == targets).squeeze()
            _y_pred = outputs.cpu().detach()
            _y_gt = targets.cpu().detach().numpy()
            acc = [0, 0]
            class_total = [0, 0]
            for i in range(len(targets)):
                label = targets[i]
                acc[label] += 1 if c[i].item() == True else 0
                class_total[label] += 1

            losses.update(loss.data.tolist(), inputs.size(0))
            if (class_total[0] != 0):
                acc_real.update(acc[0] / class_total[0])
            if (class_total[1] != 0):
                acc_fake.update(acc[1] / class_total[1])
 
            target.append(_y_gt)
            output.append(_y_pred.numpy()[:,1])
            auroc=None
            try:
                auroc = roc_auc_score(_y_gt, outputs[:,1].cpu().detach().numpy())
            except ValueError:
                pass
            sum_of_AUROC.append(auroc)
            _y_true = np.array(torch.zeros(targets.shape[0],2), dtype=np.int8)
            _y_gt = _y_gt.astype(int)
            for _ in range(len(targets)):
                _y_true[_][_y_gt[_]] = 1
            y_true = np.concatenate((y_true,_y_true))
            a = _y_pred.argmax(1)
            _y_pred = np.array(torch.zeros(_y_pred.shape).scatter(1, a.unsqueeze(1), 1),dtype=np.int8)
            y_pred = np.concatenate((y_pred,_y_pred))
        result = classification_report(y_true, y_pred, labels=None, target_names=None, sample_weight=None, digits=4, output_dict=False, zero_division='warn')
        print(result)
    
def Test(val_loader, model, criterion, log = None, source_name = ''): #Accuracy
    print(f'===> Starting the dataset {source_name}' if source_name else '===> Starting TEST')
    global best_acc
    correct, total =0,0
    losses = AverageMeter()
    arc = AverageMeter()
    main_losses = AverageMeter()
    model.eval()
    model.cuda()
    
    with torch.no_grad():
        model.eval()
        for (inputs, targets) in tqdm.tqdm(val_loader):
            inputs, targets = inputs.to('cuda'), targets.to('cuda')
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

def Eval(args, log = None,ok_PRF = False):
    if log :
        log.write(' ---------------- EVAL ----------------\n ')
    dicLoader,_, dicSourceName = initialization(args)
    model_list=[]
    weight_path = args.weight

    #if weight_path is not file type
    if os.path.isdir(weight_path): 
        for a,b,c in os.walk(weight_path):
            for _c in c:
                if 'epoch_.pth.tar' in _c: # you can change according to need
                    fullpath = os.path.join(a,_c)
                    model_list.append(fullpath)
    elif os.path.isfile(weight_path):
        model_list.append(weight_path)
    else:
        print(f"cannot load from {weight_path}")
        return None
    
    print(f'Loading BAKBONE MODEL {args.network} ...')
    for model_path in model_list:
        # model_item = os.path.join(model_item, args.name_folder1)
        model, _ = load_models(model_path, args.network, args.num_gpu, not args.test)
        if not model:
            print("FAIL Loadding MODEL !")
        criterion = nn.CrossEntropyLoss().cuda()

        for _key, _name in zip(dicLoader, dicSourceName):
            Test(dicLoader[_key], model, criterion, log, dicSourceName[_name])
        if ok_PRF: Test_PRF(dicLoader['test_dataset'], model, criterion, log)


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


def load_models(weigiht, nameNet='Xception', num_gpu='', TrainMode=True):
    teacher_model, student_model = None,None
    device = 'cuda' if num_gpu else 'cpu'
    checkpoint = None
    if weigiht:
        print(weigiht)
        if os.path.isdir(weigiht):
            checkpoint =torch.load(os.path.join(weigiht, 'model_best_accuracy.pth.tar'))
        elif os.path.isfile(weigiht):
            checkpoint = torch.load(weigiht)
        else:
            print("preweight is not exist !")
            if not TrainMode:
                return None, None

    #model load
    if nameNet=='Xception':
        teacher_model = xception_origin.xception(num_classes=2, pretrained='')
    elif nameNet=='Efficient':
        teacher_model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=2)
    if ',' in num_gpu :
        teacher_model = nn.DataParallel(teacher_model)

    #weight load
    if checkpoint:
        if ',' in num_gpu :
            teacher_model.module.load_state_dict(checkpoint['state_dict'])
        else:
            teacher_model.load_state_dict(checkpoint['state_dict'])
    if TrainMode:
        student_model = copy.deepcopy(teacher_model)
        student_model.train()
        student_model.to(device)

    teacher_model.eval()
    teacher_model.to(device)

    return teacher_model, student_model

def save_checkpoint(state, checkpoint, filename='checkpoint.pth.tar' , AUC_BEST = False, ACC_BEST = False):
    name_save= filename if filename else ''
    if AUC_BEST : name_save = 'model_best_auc'
    if ACC_BEST : name_save = 'model_best_accuracy'

    filepath = os.path.join(checkpoint, name_save+'.pth.tar')
    os.makedirs(os.path.dirname(filepath),exist_ok=True)
    torch.save(state, filepath)

#weill be refectored
def Make_DataLoader_together(rootpath_dataset,
                            name_source,
                            name_target,
                            name_target2,
                            train_aug=None,
                            val_aug=None,
                            mode_CoReD=False,
                            batch_size=128,
                            fully_train = False
                            ):
    _name = 'TransferLearning' if not fully_train else ''
    train_dir = os.path.join(rootpath_dataset ,_name, name_target , 'train/')
    train_dir2 = os.path.join(rootpath_dataset ,_name, name_target2 , 'train/')

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
    if val_source_dir:
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
