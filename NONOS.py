import os
from datetime import datetime
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.optim as optim
import torch.distributed as dist
from tqdm import tqdm
from scipy.optimize import curve_fit
import sklearn.model_selection as ms
import numpy as np
from models import *
from utils import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_path', default='/home/', type=str,
                        help='default path to save model')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus to use')
    parser.add_argument('--epochs', default=50, type=int,
                        help='the number of epochs')
    parser.add_argument('--mode', default=0, type=int,
                        help='0: train, 1: test')

    args = parser.parse_args()
    #########################################################
    args.world_size = args.gpus              #
    os.environ['MASTER_ADDR'] = 'localhost'                 #
    os.environ['MASTER_PORT'] = '12355'                     #
    mp.spawn(train, nprocs=args.gpus, args=(args,))    
  
    #########################################################

def train(gpu, args):
    ############################################################
    rank = gpu	                                               # global rank of the process within all of the processes   
    dist.init_process_group(                                   # Initialize the process and join up with the other processes
    	backend='nccl',                                        # This is 'blocking', meaning that no process will continue
   		init_method='env://',                                  # untill all processes have joined.  
    	world_size=args.world_size,                              
    	rank=rank                                               
    )                                                          
    ############################################################
    
    ############################################################
    # Data loading
    fs =  200
    t_len = 10
    data_len = fs*t_len
    freq_range = [1, fs//2]

    fpath = args.load_path + '/'
    save_path = fpath + '/'
    fname_ap_params = 'aperiodic_params'+str(fs)+'Hz'+str(t_len)+'sec_' + str(freq_range[0]) + '_' + str(freq_range[1]) +'Hz' + '.npy'
    fname_p_params = 'peak_params'+str(fs)+'Hz'+str(t_len)+'sec_' + str(freq_range[0]) + '_' + str(freq_range[1]) +'Hz' + '.npy'
    fname_ap_guides = 'ap_guides'+str(fs)+'Hz'+str(t_len)+'sec_' + str(freq_range[0]) + '_' + str(freq_range[1]) +'Hz' + '.npy'

    X = np.load(fpath + 'data.npy') # B x L
    X = torch.tensor(X).unsqueeze(1) # B x 1 x L

    ap_guides = np.load(fpath + fname_ap_guides) # B x num_F
    ap_guides = torch.tensor(ap_guides).unsqueeze(1) # B x 1 x num_F
    ap_params = np.load(fpath + fname_ap_params) # B x 2
    ap_params = torch.tensor(ap_params).unsqueeze(1) # B x 1 x 2
    p_params = np.load(fpath + fname_p_params) # B x 4
    p_params = torch.tensor(p_params).unsqueeze(1) # B x 1 x 4
    X = torch.concatenate((X, ap_params, p_params, ap_guides), dim=-1) # B x 1 x (L+6+num_F) 

    ap_guides = np.load(fpath + fname_ap_guides) # B x 1 x L

    data_ap = np.load(fpath + 'data_ap.npy') # B x L
    data_ap = torch.tensor(data_ap).unsqueeze(1) # B x 1 x L
    data_p = np.load(fpath + 'data_p.npy') # B x L
    data_p = torch.tensor(data_p).unsqueeze(1) # B x 1 x L
    data_ap_org = np.load(fpath + 'data_ap_org.npy') # B x L
    data_ap_org = torch.tensor(data_ap_org).unsqueeze(1) # B x 1 x L
    exp_series = np.load(fpath + 'exp_series.npy') # B x 1 x L
    exp_series = torch.tensor(exp_series).unsqueeze(1) # B x 1 x L

    Y = np.concatenate((data_ap, data_p, data_ap_org, exp_series), axis=1) # B x 4 x L

    # split
    X_train, X_test, Y_train, Y_test, = ms.train_test_split(X, Y,
                                                            test_size=0.2, random_state=100)

    train_dataset = list(zip(X_train, Y_train))
    test_dataset = list(zip(X_test, Y_test))
    ############################################################

    ############################################################
    ## Hyperparameters for model structure
    input_dim = 1 #data_len//2+1
    inner_dim = 256
    kernel_size = 11
    depth = 5
    num_layers = 4 # >=4
    target_length = 2048
    beta = 0.05
    rho = 100
    print('beta: {}, rho: {}'.format(beta, rho))

    ## Hyperparameters for training
    batch_size = 32
    lr = 5*1e-5 # 0.5*1e-5
    num_epochs = args.epochs
    ############################################################

    ############################################################
    # Model saved location
    fname = 'results'
    model_checkpoint_path = fpath + "/"
    model_checkpoint_model = model_checkpoint_path + fname + '_model.checkpoint'
    ############################################################

    ############################################################
    # Data sampling
    if args.mode == 0:
        dataset = list(zip(X_train, Y_train))
    elif args.mode == 1:
        dataset = list(zip(X_test, Y_test))
    
    sampler = torch.utils.data.distributed.DistributedSampler( # this makes sure that each process gets a different slice of the training data
    	dataset,
    	num_replicas=args.world_size,
    	rank=rank
    )
    
    loader = torch.utils.data.DataLoader(dataset=dataset,
                                               batch_size=batch_size,
                                               shuffle=False, ## should change to false; because we use distributedsampler instead of shuffling
                                               num_workers=0,
                                               pin_memory=True,
                                               sampler=sampler ## should specify sampler
                                               )
    ############################################################

    ############################################################
    # Define a model
    torch.manual_seed(0)
    NONOS = NONOS_UNET(input_dim, inner_dim, kernel_size, depth, num_layers)
    torch.cuda.set_device(gpu)
    NONOS.cuda(gpu)
    Optimizer = torch.optim.AdamW(NONOS.parameters(), lr)
    Scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(Optimizer, T_0=10, T_mult=1)
    cycle_loss_l1 = torch.nn.L1Loss()
    cycle_loss_l1.cuda(gpu)
    cycle_loss_l1_sum = torch.nn.L1Loss(reduction='sum')
    cycle_loss_l1_sum.cuda(gpu)
    cycle_loss_l2 = torch.nn.MSELoss()
    cycle_loss_l2.cuda(gpu)
    ############################################################

    ###############################################################
    # Wrap the model; this reproduces the model onto the GPU for the proesses
    NONOS = nn.parallel.DistributedDataParallel(NONOS,
                                                device_ids=[gpu])
    ###############################################################

    ###############################################################
    # Load the trained model
    if (args.mode == 1):
        NONOS.load_state_dict(torch.load(model_checkpoint_model))
    ###############################################################

    start = datetime.now()
    ###############################################################
    cycle_loss_l1 = torch.nn.L1Loss()

    ###############################################################
    # Training part
    if args.mode == 0:
        for epoch in range(args.epochs):
            for batch in loader:
                x = batch[0].type(torch.float32) # B x 1 x L+2
                batch_params = x[:, :, data_len:].squeeze(1) # B x 6
                batch_ap_params = batch_params[:, 0:2] # B x 2
                batch_p_params = batch_params[:, 2:6] # B x 4
                batch_ap_guides = batch_params[:, 6::] # B x num_F
                batch_ap_guides = batch_ap_guides.unsqueeze(1) # B x 1 x num_F
                batch_ap_guides = batch_ap_guides.cuda(non_blocking=True)

                x = x[:, :, 0:data_len] # B x 1 x L

                R = torch.abs(torch.fft.rfft(x, dim=-1)) # amplitude (radius) in frequency domain
                R = torch.log10(R+1e-6)
                R = R.type(torch.float32)
                R = R[:, :, 1::]

                y = batch[1].type(torch.float32) # Labels
                            
                # Get initial curve fitting
                freqs = get_freqs(data_len, fs)
                freqs = torch.tensor(freqs, dtype=torch.float32)
                freq_indices = torch.where((freqs >= freq_range[0]) & (freqs <= freq_range[1]))[0]
                freqs = freqs[freq_indices]
                freqs = freqs.repeat(R.shape[0], 1).unsqueeze(1) # B x 1 x F

                fits = torch.tensor([])
                for i in range(R.shape[0]):
                    temp_fit = ap_fit(freqs[i,0,:], batch_ap_params[i, 0], batch_ap_params[i, 1])
                    fits = torch.cat((fits, temp_fit.unsqueeze(0)), dim=0)
                fits = fits.unsqueeze(1)
                fits = fits.cuda(non_blocking=True)

                # Get cfs_indices
                params_cfs_expanded = batch_p_params.unsqueeze(2)      # Shape: [N, M, 1]
                diff = torch.abs(freqs - params_cfs_expanded)      # Shape: [N, M, K]
                params_cfs_indices_tensor = torch.argmin(diff, dim=2)     # Shape: [N, M]
                params_cfs_indices_tensor[batch_p_params == 0] = 0
                mask = params_cfs_indices_tensor != 0
                params_cfs_indices_tensor = params_cfs_indices_tensor.cuda(non_blocking=True)
                mask = mask.cuda(non_blocking=True)

                R = R.cuda(non_blocking=True) 
                x = x.cuda(non_blocking=True)
                x = symmetric_zero_pad(x, target_length)  

                ######### Forward #########
                Optimizer.zero_grad()

                fake_x_ap = unpad(NONOS(x), data_len)
                R_ap = abs(torch.fft.rfft(fake_x_ap, dim=-1))
                R_ap = R_ap[:, :, 1::]
                R_ap = torch.log10(R_ap+1e-6)
                R_ap = R_ap[:, :, freq_indices]

                R_res = F.relu(R_ap - fits) #torch.abs(R_ap - fits)
                R_res.cuda(non_blocking=True)     

                R_res = torch.gather(R_res.squeeze(1), 1, params_cfs_indices_tensor)

                ###### NONOS ######
                ## Calculate gradients and update parameters
                loss_t = cycle_loss_l1_sum(unpad(x, data_len), fake_x_ap)
                loss_f = cycle_loss_l1_sum(batch_ap_guides, R_ap)
                loss_pen = torch.sum(torch.sum(R_res * mask, dim=1)) #torch.mean(torch.max(R_res, dim=-1)[0])

                loss_G = loss_f + beta*loss_t + rho*loss_pen

                loss_G.backward()
                Optimizer.step()

            if gpu == 0: # we assume that all GPUs are synchronized, hence, only print the result of GPU0
                print('Epoch: [{}/{}], loss_f: {:6f}, loss_t: {:6f}, loss_pen: {:6f}'.format(epoch+1, num_epochs, loss_f, loss_t, loss_pen))
            
            Scheduler.step()

        if gpu == 0:
            print("Training complete in: " + str(datetime.now() - start))
            torch.save(NONOS.state_dict(), model_checkpoint_model)
    ###############################################################

    ###############################################################
    # Test part
    elif args.mode == 1:
        start = datetime.now()
        total_step = len(loader)
        with torch.no_grad():
            input_x = torch.tensor([])
            output_x_ap = torch.tensor([])
            labels = torch.tensor([])
            input_R = torch.tensor([])
            input_fits = torch.tensor([])
            ap_guides = torch.tensor([])
            for batch in loader:
                x = batch[0].type(torch.float32) # B x 1 x L+2
                batch_params = x[:, :, data_len:].squeeze(1) # B x 2
                batch_ap_params = batch_params[:, 0:2] # B x 2
                batch_p_params = batch_params[:, 2:6] # B x 4
                batch_ap_guides = batch_params[:, 6::] # B x num_F
                batch_ap_guides = batch_ap_guides.unsqueeze(1) # B x 1 x num_F
                batch_ap_guides = batch_ap_guides.cuda(non_blocking=True)

                x = x[:, :, 0:data_len] # B x 1 x L

                R = torch.abs(torch.fft.rfft(x, dim=-1)) # amplitude (radius) in frequency domain
                R = torch.log10(R+1e-6)
                R = R.type(torch.float32)
                R = R[:, :, 1::]

                y_label = batch[1].type(torch.float32) # Labels
                            
                # Get initial curve fitting
                freqs = get_freqs(data_len, fs)
                freqs = torch.tensor(freqs, dtype=torch.float32)
                freq_indices = torch.where((freqs >= freq_range[0]) & (freqs <= freq_range[1]))[0]
                freqs = freqs[freq_indices]
                freqs = freqs.repeat(R.shape[0], 1).unsqueeze(1) # B x 1 x F

                fits = torch.tensor([])
                for i in range(R.shape[0]):
                    temp_fit = ap_fit(freqs[i,0,:], batch_ap_params[i, 0], batch_ap_params[i, 1])
                    fits = torch.cat((fits, temp_fit.unsqueeze(0)), dim=0)
                fits = fits.unsqueeze(1)
                fits = fits.cuda(non_blocking=True)

                R = R.cuda(non_blocking=True) 

                x = x.cuda(non_blocking=True)
                x = symmetric_zero_pad(x, target_length)  

                ######### Forward #########
                fake_x_ap = unpad(NONOS(x), data_len)
                R_ap = abs(torch.fft.rfft(fake_x_ap, dim=-1))
                R_ap = R_ap[:, :, 1::]
                R_ap = torch.log10(R_ap+1e-6)
                R_ap = R_ap[:, :, freq_indices]

                input_x = torch.cat((input_x, x.cpu()), dim=0)
                output_x_ap = torch.cat((output_x_ap, fake_x_ap.cpu()), dim=0)
                labels = torch.cat((labels, y_label), dim=0)
                input_R = torch.cat((input_R, R.cpu()), dim=0)
                input_fits = torch.cat((input_fits, fits.cpu()), dim=0)
                ap_guides = torch.cat((ap_guides, batch_ap_guides.cpu()), dim=0)

        print("Test complete in: " + str(datetime.now() - start))
        np.save(save_path + fname + '_input_x' + '_gpu' + str(gpu) + '.npy', input_x.cpu().numpy())
        np.save(save_path + fname + '_output_x_ap' + '_gpu' + str(gpu) + '.npy', output_x_ap.cpu().numpy())
        np.save(save_path + fname + '_labels' + '_gpu' + str(gpu) + '.npy', labels.cpu().numpy())
        np.save(save_path + fname + '_input_R' + '_gpu' + str(gpu) + '.npy', input_R.cpu().numpy())
        np.save(save_path + fname + '_input_fits' + '_gpu' + str(gpu) + '.npy', input_fits.cpu().numpy())
        np.save(save_path + fname + '_ap_guides' + '_gpu' + str(gpu) + '.npy', ap_guides.cpu().numpy())
 
if __name__ == '__main__':
    main()
