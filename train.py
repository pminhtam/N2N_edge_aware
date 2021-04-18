import os
import torch
import torch.optim
import numpy as np
import argparse

from model.models import DenoiseNet
from lossfunction import *
from datasets.DenoisingDatasets import BenchmarkTrain, SIDD_VAL
import time
from skimage.metrics import peak_signal_noise_ratio

np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def train(config):
    # Train in CPU
    if config.gpu_id == -1:
        device = torch.device('cpu')
    # Train in GPU
    else:
        device = torch.device('cuda')
        os.environ['CUDA_VISIBLE_DEVICES'] = str(config.gpu_id)

    batch_size      = config.batch_size
    epochs          = config.epochs
    data_dir        = config.data_dir
    num_workers     = config.num_workers
    N               = config.N
    
    # Load dataset
    train_dataset = BenchmarkTrain(data_dir, N*batch_size, pch_size=256)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True)

    val_dataset = SIDD_VAL('/vinai/tampm2/SIDD')
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True)
        

    lossfunc = BasicLoss().cuda()
    # Load model
    net = DenoiseNet().to(device)

    
    # Optimization
    optimizer = torch.optim.Adam(
        net.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    # milestones = [10-5, 20-5, 25-5, 30-5, 35-5, 40-5, 45-5, 50-5]
    milestones = [20, 40, 60, 80, 100]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, 0.5)


    cur_epoch = 0
    global_step = 0
    if config.pretrain_model:
        print('Loading pretrained model.')
        checkpoint = torch.load(config.pretrain_model)
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        #scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        cur_epoch = checkpoint['cur_epoch']
        global_step = checkpoint['global_step']
        print('Loaded pretrained model sucessfully.')

    with torch.no_grad():
        psnr = []
        for ii, data in enumerate(val_loader):
            im_noisy = data[0].to(device)
            im_gt = data[1].to(device)

            restored = torch.clamp(net(im_noisy), 0, 1)
            restored = np.transpose(restored.cpu().numpy(), [0, 2, 3, 1])
            im_gt = np.transpose(im_gt.cpu().numpy(), [0, 2, 3, 1])

            psnr.extend([peak_signal_noise_ratio(restored[i], im_gt[i], data_range=1) for i in range(batch_size)])
        print('psnr={:.4e}'.format(np.mean(psnr)))
        
    for epoch in range(cur_epoch, cur_epoch + epochs):
        # ----------------------------------------------------
        # Training
        # ----------------------------------------------------
        loss_per_epoch = {x:[] for x in ['loss']}
        lr = scheduler.get_last_lr()[0]
        tic = time.time()
        for ii, data in enumerate(train_loader):
            net.train()
            global_step += 1

            # --------------------------------------------------------------------
            im_noisy = data[0].to(device) 
            im_gt = data[1].to(device)

            input_red = data[2].to(device) 
            input_blue = data[3].to(device)

            mask_red = data[4].to(device)
            mask_blue = data[5].to(device)
            mask_edge = data[6].to(device)

            # Inference
            output_red = net(input_red)
            with torch.no_grad():
                im_restore = net(im_noisy)

            output_red = torch.clamp(output_red, 0, 1)
            im_restore = torch.clamp(im_restore, 0, 1)

            loss = lossfunc(output_red, input_blue, im_restore, mask_red, mask_blue, mask_edge)

            # --------------------------------------------------------------------
            # Optimization
            optimizer.zero_grad()
            loss.backward()
  
                        
            optimizer.step()

            loss_per_epoch['loss'].append(loss.item())

                           
            # --------------------------------------------------------------
            # Log 
            # --------------------------------------------------------------
            if (ii+1) % config.print_freq == 0:
                mean_loss = np.mean(loss_per_epoch['loss'])
                with torch.no_grad():
                    restored = torch.clamp(net(im_noisy), 0, 1)
                    restored = np.transpose(restored.cpu().numpy(), [0, 2, 3, 1])
                    im_gt = np.transpose(im_gt.cpu().numpy(), [0, 2, 3, 1])

                    psnr = np.mean([peak_signal_noise_ratio(restored[i], im_gt[i], data_range=1) for i in range(batch_size)])
                log_str = '[Epoch:{:>2d}/{:<2d}] : {:0>4d}/{:0>4d}, Loss={:.1e}, pnsr={:.2e}, lr={:.1e}'
                print(log_str.format(epoch+1, epochs, ii+1, N, mean_loss, psnr, lr))
            
        # --------------------------------------------------------------
        ###############################################################
        # --------------------------------------------------------------
        # Save after some loops
        # --------------------------------------------------------------
        scheduler.step()
        if (epoch+1) % config.save_model_freq == 0 or epoch+1==epochs:
            model_prefix = 'model_'
            save_path_model = os.path.join(config.model_dir, model_prefix+str(epoch+1)+'.pth')
            torch.save({
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'global_step': global_step,
                'cur_epoch': epoch+1}, 
                save_path_model)

            model_state_prefix = 'model_state_'
            save_path_model_state = os.path.join(config.model_dir, model_state_prefix+str(epoch+1)+'.pth')
            torch.save(net.state_dict(), save_path_model_state)
        toc = time.time()
        
        if (epoch + 1)  % 1 == 0:
            with torch.no_grad():
                psnr = []
                for ii, data in enumerate(val_loader):
                    im_noisy = data[0].to(device)
                    im_gt = data[1].to(device)

                    restored = torch.clamp(net(im_noisy), 0, 1)
                    restored = np.transpose(restored.cpu().numpy(), [0, 2, 3, 1])
                    im_gt = np.transpose(im_gt.cpu().numpy(), [0, 2, 3, 1])

                    psnr.extend([peak_signal_noise_ratio(restored[i], im_gt[i], data_range=1) for i in range(batch_size)])
                print('psnr={:.4e}'.format(np.mean(psnr)))
        print('This epoch take time {:.2f}'.format(toc-tic))
        



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", type=int, default=1)
    parser.add_argument('--data_dir', type=str,
                        default="../SIDD_Medium/RAW/crop_medium")


    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--pretrain_model', type=str, default=None)

    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-6)

    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--save_model_freq', type=int, default=1)
    parser.add_argument('--print_freq', type=int, default=100)

    parser.add_argument('--N', type=int, default=2500)

    parser.add_argument('--model_dir', type=str, default="model")

    config = parser.parse_args()
    if not os.path.exists(config.model_dir):
        os.mkdir(config.model_dir)
    train(config)
