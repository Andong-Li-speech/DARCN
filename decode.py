import torch
import argparse
import librosa
import os
import numpy as np
import soundfile as sf
from config import win_size, win_shift, fft_num
from network import darcn

def enhance(args):
    model = darcn(causal_flag=True, stage_number=2)
    # load model file
    model.load_state_dict(torch.load(args.Model_path))
    print(model)
    model.eval()
    model.cuda()
    with torch.no_grad():
        cnt = 0
        mix_file_path = args.mix_file_path
        esti_file_path = args.esti_file_path
        os.makedirs(esti_file_path, exist_ok= True)
        file_list = os.listdir(mix_file_path)
        for file_id in file_list:
            feat_wav, _ = sf.read(os.path.join(mix_file_path, file_id))
            c = np.sqrt(np.sum((feat_wav ** 2)) / len(feat_wav))
            feat_wav = feat_wav / c
            feat_x = librosa.stft(feat_wav, n_fft=fft_num, hop_length=win_shift, window='hanning').T
            phase_x = np.angle(feat_x)
            feat_x = np.abs(feat_x)
            feat_x = torch.FloatTensor(feat_x).cuda()
            esti_x = model(feat_x.unsqueeze(dim=0))
            esti_x = esti_x[-1].cpu().numpy()
            de_esti = np.multiply(esti_x, np.exp(1j * phase_x))
            esti_utt = librosa.istft((de_esti).T, hop_length=win_shift,
                                     win_length=win_size, window='hanning', length= len(feat_wav)).astype(np.float32)
            esti_utt = esti_utt * c
            sf.write(os.path.join(esti_file_path, file_id), esti_utt, args.fs)
            print(' The %d utterance has been decoded!' % (cnt+1))
            cnt = cnt + 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Recovering audio')
    parser.add_argument('--mix_file_path', type=str, default='./Test/mix')
    parser.add_argument('--esti_file_path', type=str, default='./Test/esti')
    parser.add_argument('--fs', type=int, default=16000,
                        help='The sampling rate of speech')
    parser.add_argument('--Model_path', type=str, default='./Best_model/darcn_causal_final.pth',
                        help='The place to save best model')
    args = parser.parse_args()
    print(args)
    enhance(args=args)