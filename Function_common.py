import os
import os.path
import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from sklearn.metrics import roc_auc_score
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
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
        
#         if not torch.is_tensor(img):
#             img = np.array(img)
#             img = torch.tensor(img)
        if self.transform:
            img = self.transform(img)
        return img, self.target[idx]

def rand_bbox_custum(size, lam):
    W = size[2]
    H = size[3]
    bbx1,bbx2,bby1,bby2=0,W,0,H
    cut_rat = np.sqrt(1. - lam)
    ran_num = np.random.randint(100)
    # uniform
    if ran_num >50:
        cx = np.random.randint(W//2)
        cx = W//2 + cx
        bbx1 = np.clip(cx, 0, W)
    else:
        cy = np.random.randint(H//2)
        cy = W//2 + cy
        bby1 = np.clip(cy, 0, H)
    return bbx1, bby1, bbx2, bby2

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
        
def save_checkpoint_for_unlearning(state, checkpoint, filename='checkpoint.pt',
                                   best_filename = 'student_model_best_acc.pt',
                                   cnt=0, isAcc=False):
    if isAcc:
        torch.save(state, os.path.join(checkpoint,best_filename))
        
def Test(val_loader, model, criterion):
    global best_acc
    correct, total =0,0
    losses = AverageMeter()
    arc = AverageMeter()
    main_losses = AverageMeter()
    top1 = AverageMeter()
    model.eval()
    model.cuda()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            loss_main = criterion(outputs, targets)
            loss_cls = 0
            loss_sp = 0
            loss = loss_main
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == targets).sum().item()
            total += len(targets)
            losses.update(loss.data.tolist(), inputs.size(0))
            main_losses.update(loss_main.tolist(), inputs.size(0))
#             top1.update(correct/total, inputs.size(0))
    print(
        'Test | Loss:{loss:.4f} | MainLoss:{main:.4f} | top:{top:.4f}'.format(loss=losses.avg, main=main_losses.avg, top = correct/total*100))
    return (losses.avg, arc.avg, correct/total*100)

def Eval(test_loader, model, criterion, epoch):
    model.eval()
    model = model.cuda()
    losses = AverageMeter()
    arc = AverageMeter()
    acc_real = AverageMeter()
    acc_fake = AverageMeter()
    acc_ = AverageMeter()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            if torch.cuda.is_available():
                inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            loss = criterion(outputs, target)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == targets).squeeze()
            acc = [0, 0]
            class_total = [0, 0]
            for i in range(len(targets)):
                label = targets[i]
                acc[label] += 1 if c[i].item() == True else 0
                class_total[label] += 1
            #top1 = accuracy(outputs.data, targets.data)
            losses.update(loss.data.tolist(), inputs.size(0))
            if (class_total[0] != 0):
                acc_real.update(acc[0] / class_total[0])
            if (class_total[1] != 0):
                acc_fake.update(acc[1] / class_total[1])
            auroc = roc_auc_score(targets.cpu().detach().numpy(), outputs.cpu().detach().numpy()[:, 1])
            arc.update(auroc, inputs.size(0))

        print("Real Accuracy : {:.2f}".format(acc_real.avg*100))
        print("Fake Accuracy : {:.2f}".format(acc_fake.avg*100))
        print("Accuracy : {:.2f}".format((acc_real.avg+acc_fake.avg)/2*100))
        print("arc.avg : {:.2f}".format(arc.avg*100))
        print("")
        
def Make_DataLoader(rootpath_dataset,name_source, name_target, name_mixed_folder='', train_aug=None,val_aug=None,mode_FReTAL = False):
    train_dir=''
    val_target_dir_MIXED=''
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
        val_target_dir_MIXED = os.path.join(target_dataset_mix,'val')
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
        print('oteher mixed path is : {}'.format(val_target_dir_MIXED))
        if not(os.path.exists(val_target_dir_MIXED)):
            print("check the paths of mix dataset")
            return -1

    train_target_loader, train_target_loader_forcorrect = None,None
    train_target_dataset = datasets.ImageFolder(train_dir,transform=None)
    train_target_dataset = CustumDataset(np.array(train_target_dataset.samples)[:,0],np.array(train_target_dataset.targets),train_aug)
    train_target_loader = DataLoader(train_target_dataset, batch_size=128, shuffle=True, num_workers=8, pin_memory=True)

    if mode_FReTAL : train_target_loader_forcorrect = DataLoader(train_target_dataset, batch_size=128, shuffle=False, num_workers=8, pin_memory=True)
    val_target_loader = DataLoader(datasets.ImageFolder(val_target_dir, val_aug),
                                   batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
    val_source_loader = DataLoader(datasets.ImageFolder(val_source_dir, val_aug),
                               batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
    if name_mixed_folder:
        val_target_loader_mixed = DataLoader(datasets.ImageFolder(val_target_dir_MIXED, val_aug),
                                       batch_size=128, shuffle=True, num_workers=4, pin_memory=True)


    dic = {'train_target':train_target_loader,'val_source':val_source_loader,'val_target':val_target_loader,'val_target_mix':val_target_loader_mixed}
    dic_FReTAL = {'train_target_dataset':train_target_dataset ,'train_target_forCorrect':train_target_loader_forcorrect}
    return dic, dic_FReTAL

def Make_DataLoader_continual(rootpath_dataset,name_source,name_target,name_source2='',name_source3='',name_mixed_folder='',train_aug=None,val_aug=None,mode_FReTAL = False):
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
    else:
        val_target_dir = os.path.join(target_dataset, 'val')
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
    train_target_loader = DataLoader(train_target_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
    if mode_FReTAL : train_target_loader_forcorrect = DataLoader(train_target_dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
    val_target_loader = DataLoader(datasets.ImageFolder(val_target_dir, val_aug),
                                   batch_size=300, shuffle=True, num_workers=4, pin_memory=True)
    val_source_loader = DataLoader(datasets.ImageFolder(val_source_dir, val_aug),
                               batch_size=300, shuffle=True, num_workers=4, pin_memory=True)
    if name_source2:
        val_source_loader2 = DataLoader(datasets.ImageFolder(val_source_dir2, val_aug),
                               batch_size=300, shuffle=True, num_workers=4, pin_memory=True)
    if name_source3:
        val_source_loader3 = DataLoader(datasets.ImageFolder(val_source_dir3, val_aug),
                               batch_size=300, shuffle=True, num_workers=4, pin_memory=True)
        print(rootpath_dataset,name_source, name_target, name_mixed_folder)
    if name_mixed_folder :
        train_dir = os.path.join(rootpath_dataset+'/TransferLearning',name_mixed_folder+'/train/')
        print(train_dir)
    else :
        train_dir = os.path.join(rootpath_dataset+'/TransferLearning',name_target+'/train/')
        
    if name_mixed_folder:
        val_target_loader_mixed = DataLoader(datasets.ImageFolder(val_target_dir_MIXED, val_aug),
                                       batch_size=128, shuffle=True, num_workers=4, pin_memory=True)

    dic = {'train_target':train_target_loader,'val_source':val_source_loader,'val_source2':val_source_loader2,'val_source3':val_source_loader3,'val_target':val_target_loader,'val_target_mix':val_target_loader_mixed}
    dic_FReTAL = {'train_target_dataset':train_target_dataset ,'train_target_forCorrect':train_target_loader_forcorrect}
    return dic, dic_FReTAL

def Make_DataLoader_togeter(rootpath_dataset, name_source, name_target, name_target2, train_aug=None, val_aug=None,
                              mode_FReTAL=False):
    train_dir = os.path.join(rootpath_dataset + '/TransferLearning', name_target + '/train/')
    train_dir2 = os.path.join(rootpath_dataset + '/TransferLearning', name_target2 + '/train/')

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
    train_target_loader = DataLoader(train_target_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)

    if mode_FReTAL: train_target_loader_forcorrect = DataLoader(train_target_dataset, batch_size=128, shuffle=False,
                                                                num_workers=4, pin_memory=True)
    val_target_loader = DataLoader(datasets.ImageFolder(val_target_dir, val_aug),
                                   batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
    val_target_loader2 = DataLoader(datasets.ImageFolder(val_target_dir2, val_aug),
                                   batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
    val_source_loader = DataLoader(datasets.ImageFolder(val_source_dir, val_aug),
                                    batch_size=128, shuffle=True, num_workers=4, pin_memory=True)

    dic = {'train_target': train_target_loader, 'val_source': val_source_loader,
           'val_target': val_target_loader,'val_target2': val_target_loader2}
    dic_FReTAL = {'train_target_dataset': train_target_dataset,
                  'train_target_forCorrect': train_target_loader_forcorrect}
    return dic, dic_FReTAL



#LOSS-------------------
def loss_fn_kd(outputs, labels, teacher_outputs, KD_T=20, KD_alpha=0.5):
    KD_loss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(outputs/KD_T,dim=1),
                             F.softmax(teacher_outputs/KD_T,dim=1) * KD_alpha*KD_T*KD_T) +\
        F.cross_entropy(outputs, labels) * (1. - KD_alpha)
    return KD_loss

# L2-reg & L2-norm
def reg_cls(model):
    l2_cls = torch.tensor(0.).cuda()
    for name, param in model.named_parameters():
        if name.startswith('last_linear'):
            l2_cls += 0.5 * torch.norm(param) ** 2
    return l2_cls

def reg_l2sp(model):
    sp_loss = torch.tensor(0.).cuda()
    for name, param in model.named_parameters():
        if not name.startswith('last_linear'):
            sp_loss += 0.5 * torch.norm(param - teacher_model_weights[name]) ** 2
    return sp_loss
#-------------------