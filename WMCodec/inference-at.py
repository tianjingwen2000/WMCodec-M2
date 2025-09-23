from __future__ import absolute_import, division, print_function, unicode_literals
import glob
import os
import argparse
import json
import torch
from scipy.io.wavfile import write
from env import AttrDict
from meldataset import mel_spectrogram, MAX_WAV_VALUE, load_wav
from models import Generator, Encoder, Quantizer
from watermark import Random_watermark, Watermark_Encoder, Watermark_Decoder, sign_loss, attack, clip

# === M2 ADD: logging & metrics (for reproducibility & technical depth) ===
import csv
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
from pesq import pesq
from pystoi import stoi
import random
# =======================================================================

h = None
# === M2 MOD: sample_num from 10 → 100 (initial customization) ===
sample_num = 100
# =================================================================
bit_num = 4 

def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict

def get_mel(x):
    return mel_spectrogram(x, h.n_fft, h.num_mels, h.sampling_rate,
                           h.hop_size, h.win_size, h.fmin, h.fmax)

def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '*')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return ''
    return sorted(cp_list)[-1]

def count_common_elements(tensorA, tensorB):
    cnt = 0
    for i in range(tensorA.size(1)):
        if tensorA[0][i] == tensorB[0][i]:
            cnt += 1
    return cnt

# === M2 ADD: gaussian noise helper (for attack simulation) ===
def add_gaussian_noise(y_tensor: torch.Tensor, snr_db: float):
    y = y_tensor.detach().cpu().numpy().reshape(-1)
    rms_signal = np.sqrt(np.mean(y**2) + 1e-12)
    rms_noise = rms_signal / (10 ** (snr_db / 20.0))
    noise = np.random.randn(y.shape[0]) * rms_noise
    y_noisy = y + noise
    y_noisy = np.clip(y_noisy, -1.0, 1.0)
    y_noisy = torch.from_numpy(y_noisy).to(y_tensor.device).float().view(1,1,-1)
    return y_noisy
# =================================================================

def inference(a):
    generator = Generator(h).to(device)
    encoder = Encoder(h).to(device)
    quantizer_Audio = Quantizer(h, 'Audio').to(device)
    watermark_encoder = Watermark_Encoder(h).to(device)
    watermark_decoder = Watermark_Decoder(h).to(device)
    state_dict_g = load_checkpoint(a.checkpoint_file, device)
    generator.load_state_dict(state_dict_g['generator'])
    encoder.load_state_dict(state_dict_g['encoder'])
    quantizer_Audio.load_state_dict(state_dict_g['quantizer_Audio'])
    watermark_encoder.load_state_dict(state_dict_g['watermark_encoder'])
    watermark_decoder.load_state_dict(state_dict_g['watermark_decoder'])

    filelist = os.listdir(a.input_wavs_dir)
    print("filelist: ", len(filelist))

    os.makedirs(a.output_dir, exist_ok=True)
    # === M2 ADD: prepare results.csv for reproducibility ===
    out_dir = Path(a.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "results.csv"
    if not csv_path.exists():
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "run_id","file","iter_j","clip","opera",
                "cross_entropy","equal_bits","bit_num","acc",
                "pesq","stoi","attack_type","snr_db","seed"
            ])
    # =================================================================

    generator.eval(); generator.remove_weight_norm()
    encoder.eval(); encoder.remove_weight_norm()
    watermark_encoder.eval(); watermark_decoder.eval()

    N_result_dic = {k:[0,0,0] for k in ["CLP","RSP-90","Noise-W35","SS-01","AS-90","EA-0301","LP5000"]}
    Y_result_dic = {k:[0,0,0] for k in ["CLP","RSP-90","Noise-W35","SS-01","AS-90","EA-0301","LP5000"]}

    short_time_raw_discard = []
    short_time_clip_discard = []

    print("device : ", device)
    with torch.no_grad():
        for i, filename in enumerate(filelist):
            wav, sr = load_wav(os.path.join(a.input_wavs_dir, filename))
            wav = wav / MAX_WAV_VALUE
            wav = torch.FloatTensor(wav).to(device)
            y = wav.unsqueeze(0).unsqueeze(1)

            # === M2 ADD: optional gaussian noise as input attack ===
            y_in = y
            if a.attack_type == 'gaussian':
                y_in = add_gaussian_noise(y_in, a.snr_db)
            
            # ================ WHAT WE DID ================
            # === MP3 Compression Attack ===
            elif a.attack_type == 'mp3':
              import tempfile, subprocess
              import soundfile as sf
              tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
              tmp_mp3 = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
              tmp_out = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)

              # save as wav
              wav_np = y_in.squeeze().cpu().numpy()
              sf.write(tmp_wav.name, wav_np, sr)
              # ffmpeg, compress and then decompress
              subprocess.run(["ffmpeg","-y","-i",tmp_wav.name,"-b:a",a.bitrate,tmp_mp3.name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
              subprocess.run(["ffmpeg","-y","-i",tmp_mp3.name,tmp_out.name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
              wav_re, sr_re = librosa.load(tmp_out.name, sr=sr)
              y_in = torch.from_numpy(wav_re).float().to(y_in.device).view(1,1,-1)

            # === Resample Attack ===
            elif a.attack_type == 'resample':
              wav_np = y_in.squeeze().cpu().numpy()
              wav_re = librosa.resample(wav_np, orig_sr=sr, target_sr=a.target_sr)
              wav_re = librosa.resample(wav_re, orig_sr=a.target_sr, target_sr=sr) 
              y_in = torch.from_numpy(wav_re).float().to(y_in.device).view(1,1,-1)

            # === Reverb Attack ===
            elif a.attack_type == 'reverb':
              # Simple reverb: Original signal + delayed attenuation version
              wav_np = y_in.squeeze().cpu().numpy()
              delay = int(0.03 * sr)   # 30ms delay
              decay = 0.6
              echo = np.zeros_like(wav_np)
              if len(wav_np) > delay:
                  echo[delay:] = wav_np[:-delay] * decay
              wav_rev = wav_np + echo
              y_in = torch.from_numpy(wav_rev).float().to(y_in.device).view(1,1,-1)
            # ======================================================

            if y.shape[2] <= 1.125 * sr: 
                short_time_raw_discard.append((i, filename, y.shape[2]))
                continue

            for j in range(sample_num):
                sign = Random_watermark(1).to(device)
                sign_en = watermark_encoder(sign)

                en_y = encoder(y_in, sign_en)  # === M2 MOD: use noisy y_in ===
                q, loss_q, c = quantizer_Audio(en_y) 
                y_g_hat = generator(q)

                if j == 0: 
                    audio = y_g_hat.squeeze()
                    audio = audio * MAX_WAV_VALUE
                    audio = audio.cpu().numpy().astype('int16')
                    output_file = os.path.join(a.output_dir, os.path.splitext(filename)[0] + '.wav')
                    write(output_file, h.sampling_rate, audio)
                
                y_g_hat, clip_flag = clip(y_g_hat)
                y_g_hat, Opera = attack(y_g_hat, [("CLP",0.13),("RSP-90",0.15),("Noise-W35",0.14),
                                                  ("SS-01",0.15),("AS-90",0.15),("EA-0301",0.14),
                                                  ("LP5000",0.14)]) 
                if y_g_hat.shape[2] <= 1.125 * sr: 
                    short_time_clip_discard.append((i,j,filename,clip_flag,y_g_hat.shape[2],Opera))
                    continue 
                
                y_g_hat_mel = mel_spectrogram(y_g_hat.squeeze(1), h.n_fft, h.num_mels,
                                              h.sampling_rate, h.hop_size, h.win_size,
                                              h.fmin, h.fmax_for_loss)
                sign_score, sign_g_hat = watermark_decoder(y_g_hat_mel)
                audiomark_loss = sign_loss(sign_score, sign)

                # === M2 ADD: per-sample metrics (acc, PESQ, STOI) ===
                equal_bits = count_common_elements(sign, sign_g_hat)
                acc = float(equal_bits) / float(bit_num)

                ref_wav = y.squeeze().detach().cpu().numpy()
                deg_wav = y_g_hat.squeeze().detach().cpu().numpy()
                sr_here = h.sampling_rate  
                target_sr = 16000
                try:
                    if sr_here != target_sr:
                        ref_m = librosa.resample(ref_wav, orig_sr=sr_here, target_sr=target_sr)
                        deg_m = librosa.resample(deg_wav, orig_sr=sr_here, target_sr=target_sr)
                    else:
                        ref_m, deg_m = ref_wav, deg_wav
                    pesq_val = float(pesq(target_sr, ref_m, deg_m, 'wb'))
                except Exception:
                    pesq_val = None
                try:
                    if sr_here != target_sr:
                        ref_s = librosa.resample(ref_wav, orig_sr=sr_here, target_sr=target_sr)
                        deg_s = librosa.resample(deg_wav, orig_sr=sr_here, target_sr=target_sr)
                        stoi_val = float(stoi(ref_s, deg_s, target_sr, extended=False))
                    else:
                        stoi_val = float(stoi(ref_wav, deg_wav, sr_here, extended=False))
                except Exception:
                    stoi_val = None

                with open(csv_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        a.run_id, filename, j, clip_flag, Opera,
                        float(audiomark_loss.detach().cpu().numpy()),
                        int(equal_bits), int(bit_num), float(acc),
                        pesq_val, stoi_val, a.attack_type, float(a.snr_db), a.seed
                    ])
                # =====================================================

                if clip_flag == "N":
                    N_result_dic[Opera][0]+=1
                    N_result_dic[Opera][1]+=equal_bits
                    N_result_dic[Opera][2]+=audiomark_loss
                if clip_flag == "Y":
                    Y_result_dic[Opera][0]+=1
                    Y_result_dic[Opera][1]+=equal_bits
                    Y_result_dic[Opera][2]+=audiomark_loss

        # === M2 ADD: safe statistics printing (avoid div/0) ===
        for tag,res_dic in [("No CLIP",N_result_dic),("Yes CLIP",Y_result_dic)]:
            print("===============================")
            print(tag,"statistics")
            print("audiomark_loss:")
            for Opera,v in res_dic.items():
                n = v[0]
                if n>0:
                    print("Opera",Opera,"iter",n,"value", v[2]/n)
            print("ACC:")
            for Opera,v in res_dic.items():
                n = v[0]
                if n>0:
                    print("Opera",Opera,"iter",n,"value", v[1]/(n*bit_num))
        # =======================================================

        # === M2 ADD: generate per-clip figures ===
        try:
            import pandas as pd
            df = pd.read_csv(csv_path)
            df = df.replace([np.inf,-np.inf], np.nan)
            df = df.dropna(subset=["cross_entropy","acc"], how="any")
            for flag in ["N","Y"]:
                sub = df[df["clip"]==flag]
                if sub.empty: continue
                fig, axes = plt.subplots(1,2,figsize=(10,4))
                grp = sub.groupby("opera", dropna=False)
                grp["cross_entropy"].mean().plot(kind="bar", ax=axes[0],
                                                 title=f"Mean CE by Opera (clip={flag})")
                grp["acc"].mean().plot(kind="bar", ax=axes[1],
                                       title=f"Mean ACC by Opera (clip={flag})")
                for ax in axes:
                    ax.set_xlabel("Opera")
                    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
                plt.tight_layout()
                plt.savefig(out_dir / f"results_summary_clip_{flag}.png", dpi=200)
                plt.close(fig)
                print(f"Saved summary figure: results_summary_clip_{flag}.png")
        except Exception as e:
            print("Plot summary failed:", e)
        # =======================================================

def main():
    print('Initializing Inference Process..')
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_wavs_dir', default='')
    parser.add_argument('--output_dir', default='')
    parser.add_argument('--checkpoint_file', default='') 
    # === M2 ADD: CLI args for reproducibility & attacks ===
    parser.add_argument('--run_id', default='m2')
    # =================== WHAT WE DID ===================
    parser.add_argument('--attack_type', default='none', choices=['none','gaussian','mp3','resample','lowpass','reverb'])
    parser.add_argument('--snr_db', type=float, default=30.0)
    parser.add_argument('--bitrate', type=str, default="64k")     # mp3 attack parameter
    parser.add_argument('--target_sr', type=int, default=16000)    # resample attack parameter
    parser.add_argument('--seed', type=int, default=2025)
    # =======================================================
    a = parser.parse_args()

    config_file = os.path.join(os.path.split(a.checkpoint_file)[0], 'config.json')
    with open(config_file) as f: data = f.read()
    global h
    h = AttrDict(json.loads(data))

    # === M2 ADD: set seeds for reproducibility ===
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed)
    # =================================================

    global device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    inference(a)

if __name__ == '__main__':
    main()
