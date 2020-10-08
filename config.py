"""
This script saves the parameters needed in the implementation
Author: Andong Li
Time: 2020/10/08
"""
import os
# signal processing-based feature extraction parameters
fs = 16000
# the maximum length of each utterance for training stability
chunk_length = 4 * 16000
win_size = 320
fft_num = 320
win_shift = 160

# network paramter
causal_flag = True
stage_number = 2
batch_size = 2
epoch = 50
lr = 1e-3

# directory related parameters
dataset_path = './Dataset'
json_path = './Json'
loss_path = './Loss/darcn_loss_record.mat'
save_path = './Model'
check_point = 1
continue_from = ''
best_path = './Best_model/darcn_causal_final.pth'
os.makedirs(save_path, exist_ok=True)
os.makedirs('./Best_model', exist_ok=True)
os.makedirs('./Loss', exist_ok=True)