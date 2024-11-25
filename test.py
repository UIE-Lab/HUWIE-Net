import os
import time
import argparse
from getpass import getuser
from socket import gethostname
import logging
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
from core.models import HUWIE_Net, HUWIE_Net_I2IM, HUWIE_Net_PIM
from core.datasets import UIEBD
from core.metrics import SSIMMetric, PSNRMetric, MSEMetric

def main():
    
    parser = argparse.ArgumentParser(description='OD')
    parser.add_argument('--name', type=str, default='Test145')
    parser.add_argument('--work_dir', type=str, default='../../Data/checkpoints/')
    parser.add_argument('--UIEBD_konf', type=int, default=3)
    parser.add_argument('--model', type=object, default=HUWIE_Net)
    parser.add_argument('--train_val_batch_size', type=int, default=1)
    parser.add_argument('--test_batch_size', type=int, default=1)
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
    test_dataset = UIEBD(data_type='test', konf=konf)
    test_dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False)
    logger.info('Dataset: ' + test_dataset.__class__.__name__)

    # model
    model = args.model()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info('Device: ' + str(device))
    model.load_state_dict(torch.load('./pre_trained_models/HUWIE_Net_epoch50.pth', map_location=torch.device(device)))
    model.to(device)

    # metric
    metrics_MSE = MSEMetric()
    metrics_PSNRMetric = PSNRMetric()
    metrics_SSIMMetric = SSIMMetric()
    
    # start    
    logger.info('Host: %s, Work Dir: %s', f'{getuser()}@{gethostname()}', work_dir)
    
    logger.info('Start')
    t = time.time()
    
    avg_score_MSE_raw_gt = 0.0
    avg_score_MSE_output_gt = 0.0
    
    avg_score_PSNR_raw_gt = 0.0
    avg_score_PSNR_output_gt = 0.0
    
    avg_score_SSIM_raw_gt = 0.0
    avg_score_SSIM_output_gt = 0.0
    
    model.eval()
    with torch.no_grad():
        for k, tdata in enumerate(test_dataloader):
            
            data_time = time.time() - t
            
            tinputs = tdata['raw_data'].to(device)
            tlabels = tdata['gt_data'].to(device)
                                    
            toutputs = model(tinputs)
            
            score_MSE_raw_gt = metrics_MSE(tinputs, tlabels)
            score_MSE_output_gt = metrics_MSE(toutputs, tlabels)
            
            score_PSNR_raw_gt = metrics_PSNRMetric(tinputs, tlabels)
            score_PSNR_output_gt = metrics_PSNRMetric(toutputs, tlabels)
            
            score_SSIM_raw_gt = metrics_SSIMMetric(tinputs, tlabels)
            score_SSIM_output_gt = metrics_SSIMMetric(toutputs, tlabels)
            
            avg_score_MSE_raw_gt += score_MSE_raw_gt
            avg_score_MSE_output_gt += score_MSE_output_gt
            
            avg_score_PSNR_raw_gt += score_PSNR_raw_gt
            avg_score_PSNR_output_gt += score_PSNR_output_gt
            
            avg_score_SSIM_raw_gt += score_SSIM_raw_gt
            avg_score_SSIM_output_gt += score_SSIM_output_gt
            
            file = tdata['raw_data_path'][0].split('/')[-1]
            fp = os.path.join(work_dir_img_output, args.model.__name__ + '_' + file)
            torchvision.utils.save_image(toutputs, fp)

            logger.info('-------------------------------------------------')
            logger.info('Test => [%d/%d] Time: %.3f Image: %s', k+1, len(test_dataloader), data_time, file)
            logger.info('MSE_Raw-Gt: %.4f MSE_Out-Gt: %.4f ', score_MSE_raw_gt, score_MSE_output_gt)
            logger.info('PSNR_Raw-Gt: %.4f PSNR_Out-Gt: %.4f ', score_PSNR_raw_gt, score_PSNR_output_gt)
            logger.info('SSIM_Raw-Gt: %.4f SSIM_Out-Gt: %.4f ', score_SSIM_raw_gt, score_SSIM_output_gt)
            
    avg_score_MSE_raw_gt /= len(test_dataloader)
    avg_score_PSNR_raw_gt /= len(test_dataloader)
    avg_score_SSIM_raw_gt /= len(test_dataloader)
    avg_score_MSE_output_gt /= len(test_dataloader)
    avg_score_PSNR_output_gt /= len(test_dataloader)
    avg_score_SSIM_output_gt /= len(test_dataloader)
    
    logger.info('-------------------------------------------------')
    logger.info('Epoch Test Results (Average) =>')
    logger.info('MSE_Raw-Gt: %.4f MSE_Out-Gt: %.4f', avg_score_MSE_raw_gt, avg_score_MSE_output_gt)
    logger.info('PSNR_Raw-Gt: %.4f PSNR_Out-Gt: %.4f', avg_score_PSNR_raw_gt, avg_score_PSNR_output_gt)
    logger.info('SSIM_Raw-Gt: %.4f SSIM_Out-Gt: %.4f', avg_score_SSIM_raw_gt, avg_score_SSIM_output_gt)
                
    writer.add_scalars('Test Results/MSE', {'MSE Raw-GT':avg_score_MSE_raw_gt, 'MSE Out-GT':avg_score_MSE_output_gt})
    writer.add_scalars('Test Results/PSNR', {'PSNR Raw-GT':avg_score_PSNR_raw_gt, 'PSNR Out-GT':avg_score_PSNR_output_gt})
    writer.add_scalars('Test Results/SSIM', {'SSIM Raw-GT':avg_score_SSIM_raw_gt, 'SSIM Out-GT':avg_score_SSIM_output_gt})
    
    writer.flush()
    writer.close()
        
    logger.info('Finish')
    
if __name__ == '__main__':
    main()

    


























