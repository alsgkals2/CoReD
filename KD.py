from Function_common import *
from Function_net import *
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import torch.optim as optim

def Train(args, log = None, writer = None):
    lr = args.lr
    KD_alpha = args.KD_alpha
    num_class = args.num_class
    num_store_per = args.num_store

    print('GPU num is' , args.num_gpu)
    os.environ['CUDA_VISIBLE_DEVICES'] =str(args.num_gpu)
    savepath = f'./{args.name_sources}_{args.name_target}/{args.name_folder2}/'
    if '//' in savepath :
        savepath = savepath.replace('//','/')
    if not os.path.isdir(savepath):
        os.makedirs(savepath)
    print(f'save path : {savepath}')
    print('lr is ',lr)
    print('KD_alpha is ',KD_alpha)
    print('num_class is ',num_class)
    print('num_store_per is ',num_store_per)


    dicLoader, _, dicSourceName = initialization(args)
    teacher_model, student_model = load_models(args.weight, nameNet='Xception', num_gpu=args.num_gpu)
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(student_model.parameters(), lr=lr, momentum=0.1)
    scaler = GradScaler()
    best_epoch = 0
            
    # FIXED THE AVG OF FEATURES. IT IS FROM A TEACHER MODEL

    best_acc,epochs=0, args.epochs
    print('epochs={}'.format(epochs))
    is_best_acc = False
    writer.add_graph(teacher_model,dicLoader['train_target'].dataset[0][0].unsqueeze(0).cuda())
    
        # class_labels = [lab for lab in targets]
        # writer.add_embedding(features,
        #                     metadata=class_labels,
        #                     label_img=images.unsqueeze(1))
        # writer.close()
    n_tb = len(dicLoader['train_target'])//2
    for epoch in range(epochs):
        running_loss = []
        running_loss_kd = []
        correct,total = 0,0
        teacher_model.eval()
        student_model.train()

        for i, (inputs, targets) in tqdm(enumerate(dicLoader['train_target'])):
            inputs, targets = inputs.cuda(), targets.cuda()
            sne_loss = None
            r = np.random.rand(1)
            if r > 0.8:
                rand_index = torch.randperm(inputs.size()[0]).cuda()
                tt = targets[rand_index]
                boolean = targets != tt #THIS IS ALWAYS ATTACHING THE OPPOSITED THE 'SMALL PIECE OF A DATA'
                if True in boolean:
                    rand_index = rand_index[boolean]
                    lam = np.random.beta(0.5,0.5)
                    bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)
                    inputs[boolean, :, bbx1:bbx2, bby1:bby2] = inputs[rand_index, :, bbx1:bbx2, bby1:bby2]
            optimizer.zero_grad()
            with autocast(enabled=True):
                teacher_outputs = teacher_model(inputs)
                outputs = student_model(inputs)
                loss_main = criterion(outputs, targets)
                loss_kd = loss_fn_kd(outputs,targets,teacher_outputs)
                loss = loss_main + loss_kd
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == targets).sum().item()
            total += len(targets)
            running_loss.append(loss_main.cpu().detach().numpy())
            running_loss_kd.append(loss_kd.cpu().detach().numpy())

            if i % n_tb == 0:
                writer.add_scalar('Loss/Train', loss.item(), epoch * len(dicLoader['train_target']) + i)
                writer.add_scalar('ACC/Train', correct / total, epoch * len(dicLoader['train_target']) + i)

        
        print("Epoch: {}/{} - CE_Loss: {:.4f} | KD_Loss: {:.4f} | ACC: {:.4f}".format(epoch+1, epochs, np.mean(running_loss), np.mean(running_loss_kd), correct / total))
        
        #validataion
        if log:
            log.write(' Validation..... \n ')
        _, _, test_acc = Test(dicLoader['val_target'], student_model, criterion, log = log, source_name = args.name_target)
        total_acc = test_acc
        cnt = 1
        for name in dicLoader:
            if 'val_dataset' in name:
                _, _, source_acc = Test(dicLoader[name], student_model, criterion, log = log, source_name = dicSourceName[f'source{cnt}'])
                total_acc += source_acc
                cnt+=1
            
        is_best_acc = total_acc > best_acc  
        if (epoch+1) % 20 ==0 or is_best_acc:
            if is_best_acc : best_acc = total_acc
            is_best_acc = True
            best_acc = max(total_acc,best_acc)
            save_checkpoint({
                'epoch': epoch + 1,
                'best_epoch': best_epoch,
                'state_dict': student_model.state_dict(),
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict()},
            checkpoint = savepath,
            filename = '{}_epoch_{}.pth.tar'.format(args.weight,epoch+1 if (epoch+1)%10==0 else ''),
            ACC_BEST=is_best_acc
            )
            best_epoch = epoch+1
            print('===> save best model !!!' if is_best_acc else '===> save checkpoint model!')

