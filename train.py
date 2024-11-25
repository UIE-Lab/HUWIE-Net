import sys
import os
import time
import argparse
from getpass import getuser
from socket import gethostname
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from core.models import HUWIE_Net, HUWIE_Net_I2IM, HUWIE_Net_PIM
from core.losses import Loss
from core.datasets import UIEBD

def main():
    
    parser = argparse.ArgumentParser(description='OD')
    parser.add_argument('--name', type=str, default='Train145')
    parser.add_argument('--work_dir', type=str, default='../../Data/checkpoints/')
    parser.add_argument('--UIEBD_konf', type=int, default=3)
    parser.add_argument('--model', type=object, default=HUWIE_Net)
    parser.add_argument('--loss', type=object, default=Loss)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--train_val_batch_size', type=int, default=8)
    parser.add_argument('--test_batch_size', type=int, default=1)
    parser.add_argument('--step_size', type=int, default=50)
    parser.add_argument('--gamma', type=float, default=0.5)
    args = parser.parse_args()
        
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime()) 
    work_dir = args.work_dir + args.name + '_' + args.model.__name__ + '_' + timestamp + '/'    
    p = os.path.abspath(work_dir)
    if not os.path.exists(p): os.makedirs(p) 
    work_dir_img_output = work_dir + 'img_output' '/'
    p = os.path.abspath(work_dir_img_output)
    if not os.path.exists(work_dir_img_output): os.makedirs(work_dir_img_output) 
    
    # create text log
    logger = logging.getLogger(args.name)
    log_file = os.path.join(work_dir, f'{timestamp}.log')
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    file_handler = logging.FileHandler(log_file, 'w', encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)
    logger.info(args)
    
    # tensorboard log
    writer = SummaryWriter(os.path.join(work_dir, 'tensorboard_logs'))
    
    # dataset
    konf = args.UIEBD_konf
    train_dataset = UIEBD(data_type='train', konf=konf)
    val_dataset = UIEBD(data_type='val', konf=konf)
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_val_batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.train_val_batch_size, shuffle=True)
    logger.info('Dataset: ' + train_dataset.__class__.__name__)

    # model
    model = args.model()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info('Device: ' + str(device))
    model.to(device)
    
    # print model's state_dict
    logger.info("Model's state_dict:")
    for param_tensor in model.state_dict():
        logger.info(param_tensor + " - " + str(model.state_dict()[param_tensor].size()))
    logger.info('Finish Build Model')
    
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    # print optimizer's state_dict
    logger.info("Optimizer's state_dict:")
    for var_name in optimizer.state_dict():
        logger.info(var_name + " - " + str(optimizer.state_dict()[var_name]))
    logger.info('Finish Build Optimizer')
    
    # loss
    criterion = args.loss()
    
    # start
    train_iter = args.epochs * len(train_dataloader)
    val_iter = args.epochs * len(val_dataloader)
    
    logger.info('Host: %s, Work Dir: %s', f'{getuser()}@{gethostname()}', work_dir)
    logger.info('Epoch: %d, Train Iteration: %d, Validation Iteration: %d', args.epochs, train_iter, val_iter)
    
    logger.info('Start')
    t = time.time()
    
    # Çalışan kodda yüklenen tüm kütüphaneleri al
    loaded_modules = list(sys.modules.keys())
    with open("requirements.txt", "w") as file:
        for module in loaded_modules:
            file.write(module + "\n")

    for epoch in range(1, args.epochs + 1):
        
        logger.info('Epoch %d', epoch)
                   
        ###---TRAIN---###
        
        train_epoch_loss_item = np.zeros(criterion.loss_fn_num, dtype='float32')
        
        model.train()
        for i, tdata in enumerate(train_dataloader):
                    
            data_time = time.time() - t
                        
            inputs = tdata['raw_data'].to(device)
            labels = tdata['gt_data'].to(device)
                                    
            outputs = model(inputs)
            
            losses, weight = criterion(outputs, labels)
            total_loss = losses[-1]
                        
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
                                
            logger.info('Train Epoch: [%d][%d/%d] Time: %.3f lr: %f ' + 
                        ' '.join([criterion.loss_name[k] + ': ' + str(np.round(losses[k].item(), 4)) + ' (x' + str(weight[k]) + ')' for k in range(len(losses))]), 
                        epoch, i+1, len(train_dataloader), data_time, optimizer.param_groups[0]['lr'])
            
            for k in range(len(losses)):
                train_epoch_loss_item[k] += losses[k].item()
            
            for loss in losses:
                if torch.isnan(loss):
                    torch.save(model.state_dict(), work_dir + 'nan_loss_' + 'epoch{}.pth'.format(epoch))
                    logger.info('NaN loss...')
                    sys.exit()
                
        train_epoch_loss_item = train_epoch_loss_item / len(train_dataloader)
        
        logger.info('Train Epoch (Average): [%d] ' + ' '.join([criterion.loss_name[k] + ': ' + str(np.round(train_epoch_loss_item[k], 4)) for k in range(len(losses))]), epoch)
        
        for k in range(len(train_epoch_loss_item)):
            writer.add_scalars('Train vs. Val Loss', {'Train_' + criterion.loss_name[k]: train_epoch_loss_item[k]}, epoch)
                
        for name, param in model.named_parameters():
            writer.add_histogram('model_param/' + name, param, epoch)
            writer.add_histogram('model_param_grad/' + name, param.grad, epoch)
            writer.add_scalar('model_param_grad_abs_sum/' + name, torch.sum(torch.abs(param.grad)), epoch)

        if epoch == args.epochs:
            torch.save(model.state_dict(), work_dir + args.model.__name__ + '_epoch{}.pth'.format(epoch))
            
        scheduler.step()
        
        ###---VALIDATION---###

        val_epoch_loss_item = np.zeros(criterion.loss_fn_num, dtype='float32')
        
        model.eval()
        with torch.no_grad():
            for j, vdata in enumerate(val_dataloader):
                
                data_time = time.time() - t      
                
                vinputs = tdata['raw_data'].to(device)
                vlabels = tdata['gt_data'].to(device)
                
                voutputs  = model(vinputs)
                
                losses, weight = criterion(voutputs, vlabels)
                total_loss = losses[-1]
                
                logger.info('Val Epoch: [%d][%d/%d] Time: %.3f ' + 
                            ' '.join([criterion.loss_name[k] + ': ' + str(np.round(losses[k].item(), 4)) + ' (x' + str(weight[k]) + ')' for k in range(len(losses))]), 
                            epoch, i+1, len(val_dataloader), data_time)
                
                for k in range(len(losses)):
                    val_epoch_loss_item[k] += losses[k].item()
        
        val_epoch_loss_item = val_epoch_loss_item / len(val_dataloader)
        
        logger.info('Val Epoch (Average): [%d] ' + ' '.join([criterion.loss_name[k] + ': ' + str(np.round(val_epoch_loss_item[k], 4)) for k in range(len(losses))]), epoch)
        
        for k in range(len(train_epoch_loss_item)):
            writer.add_scalars('Train vs. Val Loss', {'Val_' + criterion.loss_name[k]: val_epoch_loss_item[k]}, epoch)
                   
    writer.flush()
    writer.close()
        
    logger.info('Finish')
    
if __name__ == '__main__':
    main()

    


























