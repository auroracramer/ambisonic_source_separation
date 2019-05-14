import argparse
import keras
import os
import sys
import random
import scipy
import scipy.signal
import librosa
import numpy as np
import pickle as pk
from angular import *
from sample import compute_feature_matrix, compute_masks
from reconstruct import *
import tqdm


def generate_speech_data(speech_path, srir_dir, sc_to_pos_dict, azi_list, elv_list,
                         steer_mat, num_frames, fft_size, hop_size, sr):
    # Load in mono speech
    src_audio, sr = librosa.load(speech_path, sr=sr, mono=True)
    src_audio /= np.abs(src_audio).max()


    steer_idx = np.random.randint(len(sc_to_pos_dict))
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

    src_len = src_audio.shape[0]
    # Convolve speech with srir
    ch_out_list = []
    for sh_str, ch_ir in zip(sh_names, ch_ir_list):
        ch_ir_len = ch_ir.shape[0]

        if ch_ir_len > src_len:
            pad_len = ch_ir_len - src_len
            src_audio = np.pad(src_audio, (0, pad_len), mode='constant')
        elif ch_ir_len < src_len:
            pad_len = src_len - ch_ir_len
            ch_ir = np.pad(ch_ir, (0, pad_len), mode='constant')

        ch_out = scipy.signal.fftconvolve(src_audio, ch_ir, mode='full')[:src_len]
        ch_out_list.append(ch_out)

    src = np.array(ch_out_list)
    src = rotate_bformat(src, d_yaw, d_pitch, d_roll, order='xyz')

    return src, steer_idx


def generate_mix_data(src, sw, noise_audio, snr_target, steer_idx, steer_mat, num_frames, fft_size, hop_size, sr):
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

    # Align B_format noise and B-format speech
    clip_len = src.shape[1]
    start_idx = np.random.randint(0, noise_audio.shape[1] - clip_len)
    noise = noise_audio[:,start_idx:start_idx + clip_len]
    snr = 10 * np.log10(np.mean(src[0,:] ** 2) / np.mean(noise[0,:] ** 2))
    alpha = 10.0**((snr_target - snr) / 20.0)#scaling factor

    sw = alpha * sw
    nw = librosa.stft(noise[0], n_fft=fft_size, window='hann', hop_length=hop_size)

    F, T = sw.shape

    # Combine the noise + speech
    mix = alpha * src + noise

    inp = compute_feature_matrix(steer_idx, mix, steer_mat,
                                 num_frames=num_frames,
                                 num_frames_hop=num_frames, fft_size=fft_size,
                                 hop_size=hop_size, sr=sr)

    mask, _ = compute_masks(sw, nw, fft_size, hop_size, sr)
    mask = mask.T

    # Split into frames
    frame_idxs = librosa.util.frame(np.arange(T),
                                    frame_length=num_frames,
                                    hop_length=num_frames).T

    inp = inp[frame_idxs]
    mask = mask[frame_idxs]

    return inp, mask, mix


def run_evaluation(model, speech_list, noise_dir, srir_dir, output_dir, sc_to_pos_dict,
                   num_frames, fft_size, hop_size, sr, random_state=12345678):

    np.random.seed(random_state)
    random.seed(random_state)

    sc_list = get_sc_list(sc_to_pos_dict)
    azi_list, elv_list = zip(*sc_list)
    azi_list = np.array(list(azi_list))
    elv_list = np.array(list(elv_list))
    steer_mat = steer_vector(azi_list, elv_list)

    noise_names = []
    noise_list = []
    for fname in os.listdir(noise_dir):
        noise_path = os.path.join(noise_dir, fname)
        noise_audio, _ = librosa.load(noise_path, sr=sr, mono=False)

        noise_names.append(fname)
        noise_list.append(noise_audio)

    clean_dir = os.path.join(output_dir, "clean")
    mix_dir = os.path.join(output_dir, "mix")
    rec_dir = os.path.join(output_dir, "recon")
    res_dir = os.path.join(output_dir, "results")

    os.makedirs(clean_dir, exist_ok=True)
    os.makedirs(mix_dir, exist_ok=True)
    os.makedirs(rec_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)

    random.shuffle(speech_list)
    speech_list = speech_list[:100]


    for speech_path in tqdm.tqdm(speech_list):
        if not speech_path.endswith('.wav'):
            continue

        snr_ratio = np.random.random()
        # Designate snr and scale
        snr_target = snr_ratio * 40.0 - 20.0

        speech_audio, steer_idx = generate_speech_data(speech_path, srir_dir, sc_to_pos_dict, azi_list, elv_list,
                                            steer_mat, num_frames, fft_size, hop_size, sr)
        sw = librosa.stft(speech_audio[0], n_fft=fft_size, window='hann', hop_length=hop_size)

        for noise_name, noise_audio in zip(noise_names, noise_list):

            inp, mask, mix = generate_mix_data(speech_audio, sw, noise_audio, snr_target, steer_idx,
                                               steer_mat, num_frames, fft_size, hop_size, sr)

            trunc_length = inp.shape[0] * inp.shape[1]
            trunc_sw = sw[:, :trunc_length]

            pred_mask_frames = model.predict(inp)
            pred_mask = np.vstack([x for x in pred_mask_frames])

            recon = get_GEVD(pred_mask, mix, fft_size, hop_size).flatten()
            speech = speech_audio[0,:recon.shape[0]]

            snr = (speech ** 2).sum() / ((speech - recon) ** 2 + np.finfo('float64').eps).sum()


            out_fname = os.path.splitext(os.path.basename(speech_path))[0] + "_" + os.path.splitext(noise_name)[0]

            clean_path = os.path.join(clean_dir, out_fname + "_clean.wav")
            mix_path = os.path.join(mix_dir, out_fname + "_mix.wav")
            rec_path = os.path.join(rec_dir, out_fname + "_recon.wav")
            res_path = os.path.join(res_dir, out_fname + "_results.pkl")

            librosa.output.write_wav(clean_path, speech, sr)
            librosa.output.write_wav(mix_path, mix.T, sr)
            librosa.output.write_wav(rec_path, recon, sr)

            res = {
                'path': speech_path,
                'input': inp,
                'mask': mask,
                'snr': snr,
                'clean_path': clean_path,
                'recon_path': rec_path,
                'mix_path': mix_path

            }

            with open(res_path, 'wb') as f:
                pk.dump(res, f)



def evaluate(model_path, test_list_path, noise_dir, srir_dir, output_dir,
             num_frames, fft_size, hop_size, sample_rate, random_state):

    sc_to_pos_dict = create_sc_to_pos_dict()

    with open(test_list_path, 'r') as f:
        test_list = f.read().split("\n")

    print("Loading model.")
    sys.stdout.flush()
    model = keras.models.load_model(model_path)

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    results = run_evaluation(model, test_list, noise_dir, srir_dir, output_dir, sc_to_pos_dict,
                             num_frames, fft_size, hop_size, sample_rate, random_state)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path')
    parser.add_argument('test_list_path')
    parser.add_argument('noise_dir')
    parser.add_argument('srir_dir')
    parser.add_argument('output_dir')

    parser.add_argument('--sample-rate', type=int, default=16000)
    parser.add_argument('--num-frames', type=int, default=25)
    parser.add_argument('--fft-size', type=int, default=1024)
    parser.add_argument('--hop-size', type=int, default=512)
    parser.add_argument('--random-state', type=int, default=12345678)

    args = vars(parser.parse_args())
    evaluate(**args)
