# Import sim functions
from neurodsp.sim import sim_powerlaw, sim_oscillation
from neurodsp.utils import set_random_seed

# Import sim functions for modulation
from neurodsp.sim import sim_modulated_signal
from neurodsp.sim.utils import modulate_signal
from neurodsp.sim.utils import rotate_timeseries
from neurodsp.sim import sim_combined

# Import utilities for plotting data
from neurodsp.utils import create_times
from neurodsp.plts import plot_time_series

import os
import argparse
import numpy as np
from specparam import *
from specparam import SpectralGroupModel
from specparam.utils import trim_spectrum
from neurodsp.spectral import compute_spectrum
import time
import torch
from random import randrange

def ap_fit(freqs, offset, exponent):
    fits = offset - np.log10(freqs**(exponent))
    return fits

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', default='/home/', type=str,
                        help='path to save the data')
    parser.add_argument('--num_data', default=10000, type=int,
                        help='number of data to generate')
    args = parser.parse_args()

    ###### Hyperparameters ########
    # Set some general settings, to be used across all simulations
    fs = 200 # sampling rate [Hz]
    t_len = 10 # time legnth [sec]
    num_pts = t_len * fs
    
    num_data = args.num_data # number of data to generate

    # data name
    exp_num = 'Figure1'
    save_path = args.save_path + '/'

    fname_nosc = 'nosc_'+str(fs)+'Hz'+str(t_len)+'sec_' + 'Experiment' + exp_num + '.npy'
    fname_osc = 'osc_'+str(fs)+'Hz'+str(t_len)+'sec_' + 'Experiment' + exp_num + '.npy'
    
    data_ap = np.zeros((num_data, num_pts), dtype=np.float32)
    data_p = np.zeros((num_data, num_pts), dtype=np.float32)
    data_ap_org = np.zeros((num_data, num_pts), dtype=np.float32)

    # periodic simulation settings
    num_oscillations = 1
    # dur_range = [1, 3]
    amp_range = [0.1, 0.2]
    fc_range = np.arange(20, 30) #np.linspace(5, 150, 30) #[5, 40] # range of center frequencies

    exp_init_range = [2, 3]
    exp_change_range = [1, 1.5]
    exp_change_time_range = [2, 8]
    exp_change_dur_range = [1, 3]
    exp_change_sign = [-1, 1]
    exp_series = np.zeros((num_data, num_pts), dtype=np.float32)

    opt = 0 # 0: sinusoidal ; 1: non-sinusoidal
    rdsym = 0.2

    for data_idx in range(num_data):
        set_random_seed(data_idx)
        np.random.seed(data_idx)

        if data_idx % 100 == 0:
            print('data_idx: ', data_idx)

        # generate oscillations
        p = np.zeros((1, num_pts))
        selected_fc = np.random.randint(0, fc_range.shape[0], (num_oscillations,))  
        amp = np.random.uniform(amp_range[0], amp_range[1], (num_oscillations+1,))
        # rand_duration = np.random.randint(dur_range[0], dur_range[1], (2,))   #dur_range[1]
        
        # generate non-oscillation
        rand_exp_init = np.random.uniform(exp_init_range[0], exp_init_range[1], (1,))
        rand_exp_change = np.random.uniform(exp_change_range[0], exp_change_range[1], (1,))
        rand_exp_change_time = np.random.randint(exp_change_time_range[0], exp_change_time_range[1], (1,))
        exp_sign = -1*np.random.choice(exp_change_sign, (1,))
        
        if rand_exp_change == 2:
            exp_sign = 1

        ap = sim_combined(n_seconds=t_len, fs=fs,
                        components={'sim_powerlaw': {'exponent': -rand_exp_init}})
        data_ap_org[data_idx, :] = ap
        ap_mod = rotate_timeseries(ap[int(rand_exp_change_time[0]*fs):int(t_len*fs)], fs=fs, delta_exp=exp_sign*rand_exp_change)
        ap[int(rand_exp_change_time[0]*fs):int(t_len*fs)] = ap_mod

        # generate first two oscillations
        p_duration = rand_exp_change_time[0]
        t = np.linspace(0, t_len, num_pts, endpoint=False)

        for osc_idx in range(num_oscillations):
            fc_idx = selected_fc[osc_idx]
            fc = fc_range[fc_idx] # np.random.uniform(fc_range[0], fc_range[1], (1,))  
            temp_p = np.sin(2 * np.pi * fc * t[:int(rand_exp_change_time[0] * fs)])
            p[:, 0:int(rand_exp_change_time[0]*fs)] = amp[osc_idx]*temp_p

        data_p[data_idx, :] = p
        data_ap[data_idx, :] = ap

        if data_idx == 0:
            print(rand_exp_change)

        exp_series[data_idx, :] = np.ones((1, num_pts)) * rand_exp_init
        exp_series[data_idx, int(rand_exp_change_time[0]*fs):int(t_len*fs)] += np.ones(int(t_len*fs) - int(rand_exp_change_time[0]*fs)) * exp_sign*rand_exp_change

    data = data_p + data_ap
    data = torch.tensor(data)
    data = data.unsqueeze(1)
    data = (data - torch.mean(data, dim=-1, keepdim=True)) / torch.std(data, dim=-1, keepdim=True)
    data = data.squeeze(1)
    data = data.numpy()

    ## Move data
    t = time.time()
    fg = SpectralGroupModel(peak_width_limits=[0.2, 2], min_peak_height=0.1, max_n_peaks=4) #, aperiodic_mode='knee')
    input_data = data # (num_data, num_time)
    R = torch.abs(torch.fft.rfft(torch.tensor(input_data), dim=-1))
    R = R + 1e-6 # (num_data, num_freqs)
    freqs = torch.fft.rfftfreq(input_data.shape[-1], 1/fs)

    freq_range = [1, fs//2]
    freq_lb_idx = np.where(freqs.numpy() >= freq_range[0])[0][0]
    freq_ub_idx = np.where(freqs.numpy() <= freq_range[1])[0][-1]

    fg.fit(freqs[freq_lb_idx:freq_ub_idx+1].numpy(), R[:, freq_lb_idx:freq_ub_idx+1].numpy())

    # Aperiodic parameters
    params = fg.get_params('aperiodic_params')

    # Periodic parameters
    peaks = fg.get_params('peak_params')
    cfs = fg.get_params('peak_params', 'CF') # extra clumn specifies which model fit it came from
    cfs_info = cfs[:, 1].astype(int)
    max_n_peaks = 4
    params_cfs = np.zeros((params.shape[0], max_n_peaks))
    for data_idx in range(params.shape[0]):
        
        aligned_idx = np.where(cfs_info == data_idx)[0]
        # print(aligned_idx)

        aligned_cfs = cfs[aligned_idx, 0]
        params_cfs[data_idx, :aligned_cfs.shape[0]] = aligned_cfs
        # params_cfs[data_idx, :peaks.shape[0]] = peaks[:, 0]
    
    # Aperiodic guidelines
    selected_freqs = freqs[freq_lb_idx:freq_ub_idx+1].numpy()
    ap_guides = np.zeros((num_data, selected_freqs.shape[0]), dtype=np.float32)

    for data_idx in range(num_data):
        ap_params = params[data_idx, :]
        aperiodic_model = ap_fit(freqs[freq_lb_idx:freq_ub_idx+1], ap_params[0], ap_params[1]).numpy()
        full_model = fg.get_model(ind=data_idx).modeled_spectrum_
        periodic_model = full_model - aperiodic_model
        ap_guides[data_idx, :] = np.log10(R[data_idx, freq_lb_idx:freq_ub_idx+1].numpy()) - periodic_model
        
    fname_ap_params = 'aperiodic_params'+str(fs)+'Hz'+str(t_len)+'sec_' + str(freq_range[0]) + '_' + str(freq_range[1]) + 'Hz' + '.npy'
    fname_p_params = 'peak_params'+str(fs)+'Hz'+str(t_len)+'sec_' + str(freq_range[0]) + '_' + str(freq_range[1]) +'Hz' + '.npy'
    fname_ap_guides = 'ap_guides'+str(fs)+'Hz'+str(t_len)+'sec_' + str(freq_range[0]) + '_' + str(freq_range[1]) +'Hz' + '.npy'

    np.save(save_path + fname_ap_params, params)
    np.save(save_path + fname_p_params, params_cfs)
    np.save(save_path + fname_ap_guides, ap_guides)
    np.save(save_path + "data.npy", data)
    np.save(save_path + "data_ap.npy", data_ap)
    np.save(save_path + "data_p.npy", data_p)
    np.save(save_path + "data_ap_org.npy", data_ap_org)
    np.save(save_path + "exp_series.npy", exp_series)

if __name__ == '__main__':
    main()
