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


def generate_example(speech_path, noise_dir, srir_dir, pos_x, pos_y, d_yaw, d_pitch, d_roll, snr_ratio, sr=44100):
    """
    Generates a ambisonic mixing example

    Parameters
    ----------
    speech_path
    noise_dir
    srir_dir
    pos_x
    pos_y
    d_yaw
    d_pitch
    d_roll
    snr_ratio
    sr

    Returns
    -------

    """
    # Load in mono speech
    src_audio, sr = librosa.load(speech_path, sr=sr, mono=True)
    src_audio /= np.abs(src_audio).max()

    # Convolve speech with srir
    grid_x = pos_x
    grid_y = pos_y

    room_type = random.choice(os.listdir(srir_dir))

    ch_out_list = []
    sh_names = ["W", "X", "Y", "Z"]
    for sh_str in sh_names:
        ch_ir_path = os.path.join(srir_dir, room_type, sh_str,
                              "{}x{:02d}y{:02d}.wav".format(sh_str, grid_x, grid_y))
        ch_ir, sr = librosa.load(ch_ir_path, sr=44100)

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

    src_bformat = np.array(ch_out_list)

    # Apply rotation
    src_bformat = rotate_bformat(src_bformat, d_yaw, d_pitch, d_roll, order='xyz')

    # Randomly sample a noise example (in B-format) that is at least as long as the example
    noise_data = None
    while noise_data is None or noise_data.shape[1] < src_bformat.shape[1]:
        noise_data, sr = librosa.load(noise_dir, sr=44100, mono=False)

    # Align B_format noise and B-format speech
    clip_len = src_bformat.shape[1]
    start_idx = np.random.randint(0, noise_data.shape[1] - clip_len)
    noise_data = noise_data[:,start_idx:start_idx + clip_len]

    # Designate snr and scale
    snr = 10 * np.log10(np.mean(src_bformat[0,:] ** 2) / np.mean(noise_data[0,:] ** 2))
    snr_target = snr_ratio * 40.0 - 20.0
    alpha = 10.0**((snr_target - snr) / 20.0)#scaling factor
    src_bformat *= alpha

    # Combine the noise + speech
    mix_bformat = src_bformat + noise_data

    return src_bformat, noise_data, mix_bformat


def lstm_speech_mask_sampler(speech_path, noise_dir, srir_dir, sc_to_pos_dict, num_steps, num_frames, num_frames_hop, fft_size, hop_size, sr):
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
    sc_list = get_sc_list(sc_to_pos_dict)
    azi_list, elv_list = zip(*sc_list)
    azi_list = np.array(list(azi_list))
    elv_list = np.array(list(elv_list))
    steer_mat = steer_vector(azi_list, elv_list)

    while True:
        steer_idx = np.random.randint(len(sc_list))
        azi, elv = azi_list[steer_idx], elv_list[steer_idx]
        pos_x, pos_y, d_yaw, d_pitch, d_roll = random.choice(sc_to_pos_dict[(azi, elv)])

        snr_ratio = np.random.random()

        src, noise, mix = generate_examples(speech_path,noise_dir,srir_dir,pos_x,pos_y,d_yaw,d_pitch,d_roll,snr_ratio)
        inp = compute_feature_matrix(steer_idx, mix, steer_mat, num_steps=num_steps, num_frames=num_frames, num_frames_hop=num_frames_hop, fft_size=fft_size, hop_size=hop_size, sr=sr)

        mask, _ = compute_masks(src[0], noise[0], sr=sr)

        yield {
            'input': inp,
            'mask': mask
        }


def lstm_data_generator(speech_list, noise_dir, srir_dir, sc_to_pos_dict,
                        num_steps, num_frames, num_frames_hop, fft_size,
                        hop_size, sr, batch_size,
                        active_streamers, rate, random_state=12345678):
    seeds = []
    for speech_path in speech_list:
        if not speech_path.endswith('.wav'):
            continue

        streamer = pescador.Streamer(lstm_speech_mask_sampler,
                                     speech_path, noise_dir, srir_dir, sc_to_pos_dict,
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
    x_sp_w = signal.stft(clip[0,:], fs=sr, window=('hann',fft_size), nperseg=fft_size, noverlap=fft_size-hop_size)
    x_sp_x = signal.stft(clip[1,:], fs=sr, window=('hann',fft_size), nperseg=fft_size, noverlap=fft_size-hop_size)
    x_sp_y = signal.stft(clip[2,:], fs=sr, window=('hann',fft_size), nperseg=fft_size, noverlap=fft_size-hop_size)
    x_sp_z = signal.stft(clip[3,:], fs=sr, window=('hann',fft_size), nperseg=fft_size, noverlap=fft_size-hop_size)
    # Dimension should be (#freq_bin, #time frame, 4)
    x_sp = np.stack((x_sp_w, x_sp_x, x_sp_y, x_sp_z), axis=-1)

    # s_hat should be (#freq_bins, #time frames)
    s_hat = np.abs(np.dot(x_sp, bf_tgt))

    F, T = s_hat.shape

    # The final concatenated feature (#frequency bins, #time frames, 2)
    feature = np.stack((x_sp_w, s_hat), axis=-1)

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


def compute_masks(src, noise, fft_size, hop_size, sr):
    #assuming each audio clip is sampled at 16kHz,compute the STFT
    #with a sinusoidal window of 1024 samples and 50% overlap.
    #window=signal.get_window('hann',1024)
    sw = signal.stft(src, fs=sr, window=('hann',fft_size), nperseg=fft_size, noverlap=fft_size-hop_size)
    nw = signal.stft(noise, fs=sr, window=('hann',fft_size), nperseg=fft_size, noverlap=fft_size-hop_size)
    Ms = (np.abs(sw)**2) / (np.abs(sw)**2 + np.abs(nw)**2)
    Mn = 1 - Ms
    return Ms, Mn



