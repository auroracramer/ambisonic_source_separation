#!/usr/bin/env python
# coding: utf-8

import argparse
import os
from keras.models import Model
from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint
from keras.layers import Input, Dense, Reshape, Masking, LSTM, TimeDistributed
from keras.optimizers import Nadam
from keras.regularizers import l2
import sys
import pescador
from angular import *
from sample import lstm_data_generator


def create_mask_model(num_steps, num_frames, fft_size, hidden_units, num_channels=2, weight_decay=1e-4, dropout=0.5):
    feature_size = fft_size // 2 + 1
    input_shape = (num_steps, num_frames, feature_size, num_channels)
    inp = Input(input_shape)
    net = Reshape((num_steps, num_frames * feature_size * num_channels))(inp)
    net = Masking(mask_value=0.0)(net)
    net = LSTM(hidden_units, activation='tanh',
               return_sequences=True,
               dropout=dropout,
               recurrent_dropout=dropout,
               kernel_regularizer=l2(weight_decay),
               recurrent_regularizer=l2(weight_decay),
               bias_regularizer=l2(weight_decay))(net)

    out = TimeDistributed(Dense(feature_size, activation='sigmoid', kernel_regularizer=l2(weight_decay),
                                bias_regularizer=l2(weight_decay)),
                          input_shape=input_shape)(net)

    model = Model(inputs=inp, outputs=out)

    return model


def train(train_list_path, valid_list_path, noise_dir, srir_dir, output_dir, num_steps, num_frames, num_frames_hop,
          hidden_units, fft_size, hop_size, sample_rate, num_epochs, batch_size, steps_per_epoch, valid_steps, weight_decay,
          learning_rate, dropout, patience, active_streamers, streamer_rate, random_state):


    sc_to_pos_dict = create_sc_to_pos_dict()

    with open(train_list_path, 'r') as f:
        train_list = f.read().split("\n")

    with open(valid_list_path, 'r') as f:
        valid_list = f.read().split("\n")

    print("Setting up generators.")
    sys.stdout.flush()

    train_gen = lstm_data_generator(train_list, noise_dir, srir_dir, sc_to_pos_dict,
                                    num_steps, num_frames, num_frames_hop,
                                    fft_size, hop_size, sample_rate, batch_size,
                                    active_streamers, streamer_rate,
                                    random_state=random_state)
    valid_gen = lstm_data_generator(valid_list, noise_dir, srir_dir, sc_to_pos_dict,
                                    num_steps, num_frames, num_frames_hop,
                                    fft_size, hop_size, sample_rate, batch_size,
                                    active_streamers, streamer_rate,
                                    random_state=random_state)

    print("Creating model.")
    sys.stdout.flush()

    model = create_mask_model(num_steps, num_frames, fft_size, hidden_units, weight_decay=weight_decay, dropout=dropout)

    model.compile(loss='mse', optimizer=Nadam(lr=learning_rate), metrics=['accuracy'])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model_filepath = os.path.join(output_dir, 'model.h5')
    log_filepath = os.path.join(output_dir, 'train_log.csv')

    callbacks = []
    callbacks.append(EarlyStopping(patience=patience))
    callbacks.append(ModelCheckpoint(model_filepath, save_best_only=True))
    callbacks.append(CSVLogger(log_filepath))

    print("Fitting model.")
    sys.stdout.flush()

    model.fit_generator(train_gen, steps_per_epoch=steps_per_epoch,
                        epochs=num_epochs, callbacks=callbacks,
                        validation_data=valid_gen,
                        validation_steps=valid_steps, verbose=1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('train_list_path')
    parser.add_argument('valid_list_path')
    parser.add_argument('noise_dir')
    parser.add_argument('srir_dir')
    parser.add_argument('output_dir')

    parser.add_argument('--sample-rate', type=int, default=16000)
    parser.add_argument('--num-steps', type=int, default=50)
    parser.add_argument('--num-frames', type=int, default=25)
    parser.add_argument('--num-frames-hop', type=int, default=13)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--hidden_units', type=int, default=512)
    parser.add_argument('--fft-size', type=int, default=1024)
    parser.add_argument('--hop-size', type=int, default=512)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--steps-per-epoch', type=int, default=64)
    parser.add_argument('--num-epochs', type=int, default=512)
    parser.add_argument('--valid-steps', type=int, default=1024)
    parser.add_argument('--active-streamers', type=int, default=20)
    parser.add_argument('--streamer-rate', type=int, default=16)
    parser.add_argument('--random-state', type=int, default=12345678)

    args = vars(parser.parse_args())
    train(**args)
