#!/usr/bin/env python
# coding: utf-8

import librosa
import os
import numpy as np
import scipy.signal
import random
import math
from scipy.io import loadmat
import pescador
import os
import random
import librosa

from angular import rotate_bformat, rotate_coord, cartesian_to_spherical, beamformer, steer_vector, get_sc_list


def generate_example(src_audio, noise_data, ch_ir_list, d_yaw, d_pitch, d_roll,
                     snr_ratio, sr=16000):
    """
    Generates a ambisonic mixing example

    Parameters
    ----------
    speech_path
    noise_dir
    ch_ir_list
    d_yaw
    d_pitch
    d_roll
    snr_ratio
    sr

    Returns
    -------

    """
    # Convolve speech with srir
    ch_out_list = []
    sh_names = ["W", "X", "Y", "Z"]
    for sh_str, ch_ir in zip(sh_names, ch_ir_list):
        ch_ir_len = ch_ir.shape[0]
        src_len = src_audio.shape[0]

        if ch_ir_len > src_len:
            pad_len = ch_ir_len - src_len
            src_audio = np.pad(src_audio, (0, pad_len), mode='constant')
        elif ch_ir_len < src_len:
            pad_len = src_len - ch_ir_len
            ch_ir = np.pad(ch_ir, (0, pad_len), mode='constant')

        ch_out = scipy.signal.fftconvolve(src_audio, ch_ir, mode='full')[:src_len]
        ch_out_list.append(ch_out)

    src = np.array(ch_out_list)

    # Apply rotation
    src = rotate_bformat(src, d_yaw, d_pitch, d_roll, order='xyz')

    # Align B_format noise and B-format speech
    clip_len = src.shape[1]
    start_idx = np.random.randint(0, noise_data.shape[1] - clip_len)
    noise_data = noise_data[:,start_idx:start_idx + clip_len]

    # Designate snr and scale
    snr = 10 * np.log10(np.mean(src[0,:] ** 2) / np.mean(noise_data[0,:] ** 2))
    snr_target = snr_ratio * 40.0 - 20.0
    alpha = 10.0**((snr_target - snr) / 20.0)#scaling factor
    src *= alpha

    # Combine the noise + speech
    mix_bformat = src_bformat + noise_data

    return src_bformat, noise_data, mix_bformat


def lstm_speech_mask_sampler(speech_path, noise_dir, srir_dir,
        sc_to_pos_dict, azi_list, elv_list, steer_mat, num_steps, num_frames, num_frames_hop, fft_size, hop_size, sr):
    """
    Generates ambisonic source separation examples from a given speech file

    Parameters
    ----------
    speech_path
    noise_dir
    srir_dir
    sc_to_pos_dict
    fft_size
    hop_size

    Returns
    -------

    """

    # Load in mono speech
    src_audio, sr = librosa.load(speech_path, sr=sr, mono=True)
    src_audio /= np.abs(src_audio).max()

    # Randomly sample a noise example (in B-format) that is at least as long as the example
    noise_audio = None
    while noise_audio is None or noise_audio.shape[1] < src_audio.shape[0]:
        fname = random.choice(os.listdir(noise_dir))
        noise_path = os.path.join(noise_dir, fname)
        noise_audio, sr = librosa.load(noise_path, sr=sr, mono=False)

    steer_idx = np.random.randint(len(sc_list))
    azi, elv = azi_list[steer_idx], elv_list[steer_idx]
    pos_x, pos_y, d_yaw, d_pitch, d_roll = random.choice(sc_to_pos_dict[(azi, elv)])

    # Load SRIR
    sh_names = ["W", "X", "Y", "Z"]
    grid_x = pos_x
    grid_y = pos_y
    ch_ir_list = []
    room_type = random.choice(os.listdir(srir_dir))
    for sh_str in sh_names:
        ch_ir_path = os.path.join(srir_dir, room_type, sh_str,
                              "{}x{:02d}y{:02d}.wav".format(sh_str, grid_x, grid_y))
        ch_ir, sr = librosa.load(ch_ir_path, sr=sr)
        ch_ir_list.append(ch_ir)

    # Convolve speech with srir
    ch_out_list = []
    for sh_str, ch_ir in zip(sh_names, ch_ir_list):
        ch_ir_len = ch_ir.shape[0]
        src_len = src_audio.shape[0]

        if ch_ir_len > src_len:
            pad_len = ch_ir_len - src_len
            src_audio = np.pad(src_audio, (0, pad_len), mode='constant')
        elif ch_ir_len < src_len:
            pad_len = src_len - ch_ir_len
            ch_ir = np.pad(ch_ir, (0, pad_len), mode='constant')

        ch_out = scipy.signal.fftconvolve(src_audio, ch_ir, mode='full')[:src_len]
        ch_out_list.append(ch_out)

    src = np.array(ch_out_list)

    if src.ndim == 1:
        import pdb
        pdb.set_trace()

    # Apply rotation
    src = rotate_bformat(src, d_yaw, d_pitch, d_roll, order='xyz')

    # Align B_format noise and B-format speech
    clip_len = src.shape[1]
    start_idx = np.random.randint(0, noise_audio.shape[1] - clip_len)
    noise = noise_audio[:,start_idx:start_idx + clip_len]
    snr = 10 * np.log10(np.mean(src[0,:] ** 2) / np.mean(noise[0,:] ** 2))

    sw = librosa.stft(src[0], n_fft=fft_size, window='hann', hop_length=hop_size)
    nw = librosa.stft(noise[0], n_fft=fft_size, window='hann', hop_length=hop_size)

    while True:
        snr_ratio = np.random.random()
        # Designate snr and scale
        snr_target = snr_ratio * 40.0 - 20.0
        alpha = 10.0**((snr_target - snr) / 20.0)#scaling factor
        scaled_src = alpha * src
        scaled_sw = alpha * sw

        # Combine the noise + speech
        mix = scaled_src + noise

        inp = compute_feature_matrix(steer_idx, mix, steer_mat,
                                     num_steps=num_steps, num_frames=num_frames,
                                     num_frames_hop=num_frames_hop, fft_size=fft_size,
                                     hop_size=hop_size, sr=sr)

        mask, _ = compute_masks(scaled_sw, nw, fft_size, hop_size, sr)

        yield {
            'input': inp,
            'mask': mask
        }


def lstm_data_generator(speech_list, noise_dir, srir_dir, sc_to_pos_dict,
                        num_steps, num_frames, num_frames_hop, fft_size,
                        hop_size, sr, batch_size,
                        active_streamers, rate, random_state=12345678):

    sc_list = get_sc_list(sc_to_pos_dict)
    azi_list, elv_list = zip(*sc_list)
    azi_list = np.array(list(azi_list))
    elv_list = np.array(list(elv_list))
    steer_mat = steer_vector(azi_list, elv_list)

    seeds = []
    for speech_path in speech_list:
        if not speech_path.endswith('.wav'):
            continue

        streamer = pescador.Streamer(lstm_speech_mask_sampler,
                                     speech_path, noise_dir, srir_dir, sc_to_pos_dict,
                                     azi_list, elv_list, steer_mat,
                                     num_steps, num_frames, num_frames_hop,
                                     fft_size, hop_size, sr)
        seeds.append(streamer)

    # Randomly shuffle the seeds
    random.shuffle(seeds)

    mux = pescador.StochasticMux(seeds, active_streamers, rate=rate, random_state=random_state)

    if batch_size == 1:
        return mux
    else:
        return pescador.maps.buffer_stream(mux, batch_size)


def compute_feature_matrix(tgt_idx, clip, D, num_steps=50, num_frames=25, num_frames_hop=13, fft_size=1024, hop_size=512, sr=16000):
    """
    Computes the input feature matrix for the LSTM model, consisting of the beamformed spectrogram and the
    spectrogram of the omnidirectional channel

    Parameters
    ----------
    tgt_idx : index of azimuth/elevation pair in steering matrix
    clip : 4-channel audio clip
    D : steeering matrix, which is constant for all calculations
    fft_size : size of FFT
    hop_size : hop size
    sr : sample rate

    Returns
    -------
    feature : feature matrix

    """
    bf = beamformer(tgt_idx, D)
    # 1 by 4 vector
    bf_tgt = bf[tgt_idx,:].flatten()

    # Compute stft of 4 channels of audioclip
    x_sp_w = librosa.stft(clip[0,:], n_fft=fft_size, window='hann', hop_length=hop_size)
    x_sp_x = librosa.stft(clip[1,:], n_fft=fft_size, window='hann', hop_length=hop_size)
    x_sp_y = librosa.stft(clip[2,:], n_fft=fft_size, window='hann', hop_length=hop_size)
    x_sp_z = librosa.stft(clip[3,:], n_fft=fft_size, window='hann', hop_length=hop_size)
    # Dimension should be (#freq_bin, #time frame, 4)
    x_sp = np.stack((x_sp_w, x_sp_x, x_sp_y, x_sp_z), axis=-1)

    # s_hat should be (#freq_bins, #time frames)
    s_hat = np.abs(np.dot(x_sp, bf_tgt))

    F, T = s_hat.shape

    # The final concatenated feature (#frequency bins, #time frames, 2)
    feature = np.stack((np.abs(x_sp_w), s_hat), axis=-1)

    # Reshape
    feature = feature.reshape((T, F, 2))

    # Split into frames
    total_num_frames = num_frames + num_frames_hop * (num_steps - 1)
    num_pad = num_frames + num_frames_hop * (num_steps - 1) - T
    if num_pad > 0:
        feature = np.pad(feature, ((0, num_pad), (0,0), (0,0)), mode='constant')

    frame_idxs = librosa.util.frame(np.arange(total_num_frames),
                                    frame_length=num_frames,
                                    hop_length=num_frames_hop).T

    feature = feature[frame_idxs.T]

    return feature


def compute_masks(sw, nw, fft_size, hop_size, sr):
    #assuming each audio clip is sampled at 16kHz,compute the STFT
    #with a sinusoidal window of 1024 samples and 50% overlap.
    #window=signal.get_window('hann',1024)
    Ms = (np.abs(sw)**2) / (np.abs(sw)**2 + np.abs(nw)**2)
    Mn = 1 - Ms
    return Ms, Mn



