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
    parser.add_argument('--load_fname', default='/home/', type=str,
                        help='file name to load')
    parser.add_argument('--fs', default=200, type=int,
                        help='sampling rate [Hz]')
    parser.add_argument('--t_len', default=10, type=int,
                        help='time length of data [sec]')
    parser.add_argument('--save_path', default='/home/', type=str,
                        help='path to save the guides')
    args = parser.parse_args()

    ###### Hyperparameters ########
    # Set some general settings, to be used across all simulations
    fs = args.fs # sampling rate [Hz]
    t_len = args.t_len # time legnth [sec]
    num_pts = t_len * fs
    
    # data name
    save_path = args.save_path + '/'

    # Load data
    data = np.load(args.load_fname) # (num_data, num_pts)
    num_data = data.shape[0]

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

if __name__ == '__main__':
    main()