from Function_common import *
from Function_CoReD import *
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

#This is for Task1 (Single model, No teacher-student framework)
def Train(args):
    lr = args.lr
    savepath = f'./weights/{args.name_sources}/{args.name_folder2}/'
    if '//' in savepath :
        savepath = savepath.replace('//','/')
    if not os.path.isdir(savepath):
        os.makedirs(savepath)
    print(f'save path : {savepath}')
    print('lr is ',lr)

    dicLoader,dicCoReD, dicSourceName = initialization(args)
    _, model = load_models(args.weigiht, args.name_sources, False)
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.1)
    scaler = GradScaler()
            
    # FIXED THE AVG OF FEATURES. IT IS FROM A TEACHER MODEL

    best_acc,epochs=0, args.epochs
    print('epochs={}'.format(epochs))
    is_best_acc = False
    
    for epoch in range(epochs):
        running_loss = []
        correct,total = 0,0
        model.train()

        for(inputs, targets) in tqdm(dicLoader['train_target']):
            inputs, targets = inputs.cuda(), targets.cuda()
            r = np.random.rand(1)
            if 0.5 > 0 and r < 0.5:
                rand_index = torch.randperm(inputs.size()[0]).cuda()
                tt= targets[rand_index]
                boolean = targets==tt
                rand_index = rand_index[boolean]
                lam = np.random.beta(1.0, 1.0)
                bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)
                inputs[boolean, :, bbx1:bbx2, bby1:bby2] = inputs[rand_index, :, bbx1:bbx2, bby1:bby2]
            optimizer.zero_grad()
            
            with autocast(enabled=True):
                outputs = model(inputs)
                loss_main = criterion(outputs, targets)
                loss = loss_main

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == targets).sum().item()
            total += len(targets)
            running_loss.append(loss_main.cpu().detach().numpy())
        print("Epoch: {}/{} - CE_Loss: {:.4f} | ACC: {:.4f}".format(epoch+1, epochs, np.mean(running_loss), correct / total))
        
        #validataion
        _, _, test_acc = Test(dicLoader['val_target'], student_model, criterion)
        total_acc = test_acc
        #continual val check
        for name in dicSourceName:
            if 'val_source' in name:
                _, _, source_acc = Test(dicLoader[name], student_model, criterion)
                total_acc += source_acc
            
        is_best_acc = total_acc > best_acc  
        if (epoch+1)%20 ==0 or is_best_acc:
            if is_best_acc : best_acc = total_acc
            is_best_acc = True
            best_acc = max(total_acc,best_acc)
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': student_model.state_dict(),
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict()
            },  args,
                filename = '{}_epoch_{}.pth.tar'.format(args.weigiht,epoch+1 if (epoch+1)%10==0 else ''),
                ACC_BEST=True
                )

            print('! ! ! save ! ! !')

