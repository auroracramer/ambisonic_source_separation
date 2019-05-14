import argparse
import soundfile as sf
import os
import random
import math


def get_split(audio_dir, output_dir, valid_ratio=0.15, test_ratio=0.15,
              sample_rate=16000, max_duration=None):
    # Not optimal but sufficient for now

    speaker_dirs = os.listdir(audio_dir)
    random.shuffle(speaker_dirs)

    num_speakers = len(speaker_dirs)

    num_valid = int(num_speakers * valid_ratio)
    num_test = int(num_speakers * test_ratio)
    num_train = num_speakers - num_valid - num_test

    train_speakers = speaker_dirs[:num_train]
    valid_speakers = speaker_dirs[num_train:num_train+num_valid]
    test_speakers = speaker_dirs[num_train+num_valid:]

    train_path = os.path.join(output_dir, 'train.txt')
    valid_path = os.path.join(output_dir, 'valid.txt')
    test_path = os.path.join(output_dir, 'test.txt')

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    with open(train_path, 'w') as f:
        for speaker_dir in train_speakers:
            speaker_dir = os.path.join(audio_dir, speaker_dir)
            for fname in os.listdir(speaker_dir):
                path = os.path.join(speaker_dir, fname)
                try:
                    with sf.SoundFile(path, 'rb') as sound_file:
                        num_samples = math.ceil(len(sound_file) * sample_rate \
                                                / sound_file.samplerate)
                except TypeError:
                    print("Skipping {}.".format(fname))
                    continue

                if max_duration is not None and max_duration > (num_samples / sample_rate):
                    continue

                f.write(path + "\n")

    with open(valid_path, 'w') as f:
        for speaker_dir in valid_speakers:
            speaker_dir = os.path.join(audio_dir, speaker_dir)
            for fname in os.listdir(speaker_dir):
                path = os.path.join(speaker_dir, fname)
                try:
                    with sf.SoundFile(path, 'rb') as sound_file:
                        num_samples = math.ceil(len(sound_file) * sample_rate \
                                                / sound_file.samplerate)
                except TypeError:
                    print("Skipping {}.".format(fname))
                    continue

                f.write(path + "\n")

    with open(test_path, 'w') as f:
        for speaker_dir in test_speakers:
            speaker_dir = os.path.join(audio_dir, speaker_dir)
            for fname in os.listdir(speaker_dir):
                path = os.path.join(speaker_dir, fname)
                try:
                    with sf.SoundFile(path, 'rb') as sound_file:
                        num_samples = math.ceil(len(sound_file) * sample_rate \
                                                / sound_file.samplerate)
                except TypeError:
                    print("Skipping {}.".format(fname))
                    continue

                f.write(path + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('audio_dir', type=str)
    parser.add_argument('output_dir', type=str)
    parser.add_argument('--valid_ratio', type=float, default=0.15)
    parser.add_argument('--test_ratio', type=float, default=0.15)
    parser.add_argument('--sample-rate', type=int, default=16000)
    parser.add_argument('--max-duration', type=float)
    args = vars(parser.parse_args())

    get_split(**args)
