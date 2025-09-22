import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import itertools
import os
import time
import argparse
import json
import pdb
import torch
import torch.nn.functional as F
from torchaudio.transforms import MelSpectrogram
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DistributedSampler, DataLoader
import torch.multiprocessing as mp
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel
from env import AttrDict, build_env
from meldataset import MelDataset, mel_spectrogram, get_dataset_filelist
from msstftd import MultiScaleSTFTDiscriminator
from watermark import Random_watermark, Watermark_Encoder, Watermark_Decoder, sign_loss, attack
from models import Generator, MultiPeriodDiscriminator, MultiScaleDiscriminator, feature_loss, generator_loss,\
    discriminator_loss, Encoder, Quantizer
try:
    from utils import plot_spectrogram, scan_checkpoint, load_checkpoint, save_checkpoint
except:
    from .utils import plot_spectrogram, scan_checkpoint, load_checkpoint, save_checkpoint

torch.backends.cudnn.benchmark = True

def reconstruction_loss(x, G_x, device, eps=1e-7):
    L = 100*F.mse_loss(x, G_x) # wav L1 loss
    for i in range(6,11):
        s = 2**i
        melspec = MelSpectrogram(sample_rate=24000, n_fft=s, hop_length=s//4, n_mels=64, wkwargs={"device": device}).to(device)
        # 64, 16, 64
        # 128, 32, 128
        # 256, 64, 256
        # 512, 128, 512
        # 1024, 256, 1024
        S_x = melspec(x)
        S_G_x = melspec(G_x)
        loss = ((S_x-S_G_x).abs().mean() + (((torch.log(S_x.abs()+eps)-torch.log(S_G_x.abs()+eps))**2).mean(dim=-2)**0.5).mean())/(i)
        L += loss
        #print('i ,loss ', i, loss)
    #assert 1==2
    return L

def train(rank, a, h):
    if h.num_gpus > 1:
        init_process_group(backend=h.dist_config['dist_backend'], init_method=h.dist_config['dist_url'],
                           world_size=h.dist_config['world_size'] * h.num_gpus, rank=rank)

    torch.cuda.manual_seed(h.seed)
    device = torch.device('cuda:{:d}'.format(rank))

    encoder = Encoder(h).to(device)
    generator = Generator(h).to(device)
    quantizer_Audio = Quantizer(h,'Audio').to(device)
    watermark_encoder = Watermark_Encoder(h).to(device)
    watermark_decoder = Watermark_Decoder(h).to(device)
    mpd = MultiPeriodDiscriminator().to(device)
    msd = MultiScaleDiscriminator().to(device)
    mstftd = MultiScaleSTFTDiscriminator(32).to(device)
    if rank == 0:
        print(encoder)
        print(quantizer_Audio)
        print(generator)
        print(watermark_encoder)
        print(watermark_decoder)
        os.makedirs(a.checkpoint_path, exist_ok=True)
        print("checkpoints directory : ", a.checkpoint_path)

    if os.path.isdir(a.checkpoint_path):
        cp_g = scan_checkpoint(a.checkpoint_path, 'g_')
        cp_do = scan_checkpoint(a.checkpoint_path, 'do_')

    steps = 0
    if cp_g is None or cp_do is None:
        state_dict_do = None
        last_epoch = -1
    else:
        state_dict_g = load_checkpoint(cp_g, device)
        state_dict_do = load_checkpoint(cp_do, device)
        generator.load_state_dict(state_dict_g['generator'])
        encoder.load_state_dict(state_dict_g['encoder'])
        quantizer_Audio.load_state_dict(state_dict_g['quantizer_Audio'])
        watermark_encoder.load_state_dict(state_dict_g['watermark_encoder'])
        watermark_decoder.load_state_dict(state_dict_g['watermark_decoder'])
        mpd.load_state_dict(state_dict_do['mpd'])
        msd.load_state_dict(state_dict_do['msd'])
        mstftd.load_state_dict(state_dict_do['mstftd'])
        steps = state_dict_do['steps'] + 1
        last_epoch = state_dict_do['epoch']

    if h.num_gpus > 1:
        generator = DistributedDataParallel(generator, device_ids=[rank]).to(device)
        encoder = DistributedDataParallel(encoder, device_ids=[rank]).to(device)
        quantizer_Audio = DistributedDataParallel(quantizer_Audio, device_ids=[rank]).to(device)
        watermark_encoder = DistributedDataParallel(watermark_encoder, device_ids=[rank]).to(device)
        watermark_decoder = DistributedDataParallel(watermark_decoder, device_ids=[rank]).to(device)
        mpd = DistributedDataParallel(mpd, device_ids=[rank]).to(device)
        msd = DistributedDataParallel(msd, device_ids=[rank]).to(device)
        mstftd = DistributedDataParallel(mstftd, device_ids=[rank]).to(device)

    optim_g = torch.optim.Adam(itertools.chain(generator.parameters(), encoder.parameters(), quantizer_Audio.parameters(), watermark_encoder.parameters(), watermark_decoder.parameters()),
                                h.learning_rate, betas=[h.adam_b1, h.adam_b2])
    optim_d = torch.optim.Adam(itertools.chain(msd.parameters(), mpd.parameters(), mstftd.parameters()),
                                h.learning_rate, betas=[h.adam_b1, h.adam_b2])
    if state_dict_do is not None:
        optim_g.load_state_dict(state_dict_do['optim_g'])
        optim_d.load_state_dict(state_dict_do['optim_d'])

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=h.lr_decay, last_epoch=last_epoch)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=h.lr_decay, last_epoch=last_epoch)

    training_filelist, validation_filelist = get_dataset_filelist(a)

    trainset = MelDataset(training_filelist, h.segment_size, h.n_fft, h.num_mels,
                          h.hop_size, h.win_size, h.sampling_rate, h.fmin, h.fmax, n_cache_reuse=0,
                          shuffle=False if h.num_gpus > 1 else True, fmax_loss=h.fmax_for_loss, device=device,
                          fine_tuning=a.fine_tuning, base_mels_path=a.input_mels_dir)

    train_sampler = DistributedSampler(trainset) if h.num_gpus > 1 else None

    train_loader = DataLoader(trainset, num_workers=h.num_workers, shuffle=False,
                              sampler=train_sampler,
                              batch_size=h.batch_size,
                              pin_memory=True,
                              drop_last=True)

    if rank == 0:
        validset = MelDataset(validation_filelist, h.segment_size, h.n_fft, h.num_mels,
                              h.hop_size, h.win_size, h.sampling_rate, h.fmin, h.fmax, False, False, n_cache_reuse=0,
                              fmax_loss=h.fmax_for_loss, device=device, fine_tuning=a.fine_tuning,
                              base_mels_path=a.input_mels_dir)
        validation_loader = DataLoader(validset, num_workers=1, shuffle=False,
                                       sampler=None,
                                       batch_size=1,
                                       pin_memory=True,
                                       drop_last=True)
        sw = SummaryWriter(os.path.join(a.checkpoint_path, 'logs'))
    plot_gt_once = False
    generator.train()
    encoder.train()
    quantizer_Audio.train()
    watermark_encoder.train()
    watermark_decoder.train()
    mpd.train()
    msd.train()
    for epoch in range(max(0, last_epoch), a.training_epochs):
        if rank == 0:
            start = time.time()
            print("Epoch: {}".format(epoch+1))
        if h.num_gpus > 1:
            train_sampler.set_epoch(epoch)
        for i, batch in enumerate(train_loader):
            if rank == 0:
                start_b = time.time()
            x, y, _, y_mel = batch
            sign = Random_watermark(h.batch_size)
            # x:[32, 80, 50] y:[32, 12000] y_mel:[32, 80, 50] hop_size=240 batch_size=24
            x = torch.autograd.Variable(x.to(device, non_blocking=True))
            y = torch.autograd.Variable(y.to(device, non_blocking=True))
            y_mel = torch.autograd.Variable(y_mel.to(device, non_blocking=True))
            sign = torch.autograd.Variable(sign.to(device, non_blocking=True))
            sign_en = watermark_encoder(sign)
            y = y.unsqueeze(1)

            c = encoder(y, sign_en) # [32, 512, 50]
            # print("c.shape: ", c.shape)
            q, loss_q, c = quantizer_Audio(c)
            # print("q.shape: ", q.shape)
    
            y_g_hat = generator(q) # [32, 1, 12000]

            y_g_hat, Opera = attack(y_g_hat, [("CLP", 0.4), ("RSP-90", 0.10), ("Noise-W35", 0.15), ("SS-01", 0.05), ("AS-90", 0.05), ("EA-0301", 0.05), ("LP5000", 0.20)])

            y_g_hat_mel = mel_spectrogram(y_g_hat.squeeze(1), h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size,
                                          h.fmin, h.fmax_for_loss) # 1024, 80, 24000, 240,1024 # [32, 80, 50]
            sign_score, sign_g_hat = watermark_decoder(y_g_hat_mel)

            y_r_mel_1 = mel_spectrogram(y.squeeze(1), 512, h.num_mels, h.sampling_rate, 120, 512,
                                          h.fmin, h.fmax_for_loss)
            y_g_mel_1 = mel_spectrogram(y_g_hat.squeeze(1), 512, h.num_mels, h.sampling_rate, 120, 512,
                                          h.fmin, h.fmax_for_loss)
            y_r_mel_2 = mel_spectrogram(y.squeeze(1), 256, h.num_mels, h.sampling_rate, 60, 256,
                                          h.fmin, h.fmax_for_loss)
            y_g_mel_2 = mel_spectrogram(y_g_hat.squeeze(1), 256, h.num_mels, h.sampling_rate, 60, 256,
                                          h.fmin, h.fmax_for_loss)
            y_r_mel_3 = mel_spectrogram(y.squeeze(1), 128, h.num_mels, h.sampling_rate, 30, 128,
                                          h.fmin, h.fmax_for_loss)
            y_g_mel_3 = mel_spectrogram(y_g_hat.squeeze(1), 128, h.num_mels, h.sampling_rate, 30, 128,
                                          h.fmin, h.fmax_for_loss)
            # print("x.shape: ", x.shape)
            # print("y.shape: ", y.shape)
            # print("y_g_hat.shape: ", y_g_hat.shape)
            optim_d.zero_grad()

            # MPD
            y_df_hat_r, y_df_hat_g, _, _ = mpd(y, y_g_hat.detach())
            loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(y_df_hat_r, y_df_hat_g)

            # MSD
            y_ds_hat_r, y_ds_hat_g, _, _ = msd(y, y_g_hat.detach())
            loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(y_ds_hat_r, y_ds_hat_g)

            y_disc_r, fmap_r = mstftd(y)
            y_disc_gen, fmap_gen = mstftd(y_g_hat.detach())
            loss_disc_stft, losses_disc_stft_r, losses_disc_stft_g = discriminator_loss(y_disc_r, y_disc_gen)
            loss_disc_all = loss_disc_s + loss_disc_f + loss_disc_stft

            loss_disc_all.backward()
            optim_d.step()

            # Generator
            optim_g.zero_grad()

            # L1 Mel-Spectrogram Loss
            loss_mel1 = F.l1_loss(y_r_mel_1, y_g_mel_1)
            loss_mel2 = F.l1_loss(y_r_mel_2, y_g_mel_2)
            loss_mel3 = F.l1_loss(y_r_mel_3, y_g_mel_3)
            #print('loss_mel1, loss_mel2 ', loss_mel1, loss_mel2)
            loss_mel = F.l1_loss(y_mel, y_g_hat_mel) * 45 + loss_mel1 + loss_mel2
            # print('loss_mel ', loss_mel)
            # assert 1==2
            y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(y, y_g_hat)
            y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(y, y_g_hat)
            y_stftd_hat_r, fmap_stftd_r = mstftd(y)
            y_stftd_hat_g, fmap_stftd_g = mstftd(y_g_hat)
            loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
            loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
            loss_fm_stft = feature_loss(fmap_stftd_r, fmap_stftd_g)
            loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)
            loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g)
            loss_gen_stft, losses_gen_stft = generator_loss(y_stftd_hat_g)
            # Audiomark loss
            audiomark_loss = sign_loss(sign_score, sign)
            # epoch 1：loss_gen_s 2.3417 loss_gen_f 3.8850 loss_gen_stft 4.7083 loss_fm_s 0.1931 loss_fm_f 0.2926 loss_fm_stft 0.1049 loss_mel 189.7696 loss_q 0.0108 audiomark_loss 2.3130
            loss_gen_all = loss_gen_s + loss_gen_f + loss_gen_stft + loss_fm_s + loss_fm_f + loss_fm_stft + loss_mel + loss_q * 10 + audiomark_loss * 5
            loss_gen_all.backward()
            optim_g.step()
            if rank == 0:
                # STDOUT logging
                if steps % a.stdout_interval == 0:
                    with torch.no_grad():
                        mel_error = F.l1_loss(y_mel, y_g_hat_mel).item()
                    print('Steps : {:d}, Gen Loss Total : {:4.3f}, Loss Q : {:4.3f}, Mel-Spec. Error : {:4.3f}, audiomark_loss : {:4.3f}, s/b : {:4.3f}'.
                          format(steps, loss_gen_all, loss_q, mel_error, audiomark_loss, time.time() - start_b))
                # checkpointing
                if steps % a.checkpoint_interval == 0 and steps != 0:
                    checkpoint_path = "{}/g_{:08d}".format(a.checkpoint_path, steps)
                    save_checkpoint(checkpoint_path,
                                    {'generator': (generator.module if h.num_gpus > 1 else generator).state_dict(),
                                     'encoder': (encoder.module if h.num_gpus > 1 else encoder).state_dict(),
                                     'quantizer_Audio': (quantizer_Audio.module if h.num_gpus > 1 else quantizer_Audio).state_dict(),
                                     'watermark_encoder': (watermark_encoder.module if h.num_gpus > 1 else watermark_encoder).state_dict(),
                                     'watermark_decoder': (watermark_decoder.module if h.num_gpus > 1 else watermark_decoder).state_dict()
                                     }, num_ckpt_keep=a.num_ckpt_keep)
                    checkpoint_path = "{}/do_{:08d}".format(a.checkpoint_path, steps)
                    save_checkpoint(checkpoint_path,
                                    {'mpd': (mpd.module if h.num_gpus > 1
                                                         else mpd).state_dict(),
                                     'msd': (msd.module if h.num_gpus > 1
                                                         else msd).state_dict(),
                                     'mstftd': (mstftd.module if h.num_gpus > 1
                                                               else mstftd).state_dict(),
                                     'optim_g': optim_g.state_dict(), 'optim_d': optim_d.state_dict(), 'steps': steps,
                                     'epoch': epoch}, num_ckpt_keep=a.num_ckpt_keep)
                # Tensorboard summary logging
                if steps % a.summary_interval == 0:
                    sw.add_scalar("training/gen_loss_total", loss_gen_all, steps)
                    sw.add_scalar("training/mel_spec_error", mel_error, steps)

                # Validation
                if steps % a.validation_interval == 0 and steps != 0:
                    generator.eval()
                    encoder.eval()
                    quantizer_Audio.eval()
                    watermark_encoder.eval()
                    watermark_decoder.eval()
                    torch.cuda.empty_cache()
                    val_err_tot = 0
                    audiomark_loss_tot = 0
                    with torch.no_grad():
                        for j, batch in enumerate(validation_loader):
                            x, y, _, y_mel = batch
                            sign = Random_watermark(x.shape[0])
                            sign = sign.to(device)
                            y_mel = y_mel.to(device)
                            sign_en = watermark_encoder(sign)
                            c = encoder(y.to(device).unsqueeze(1), sign_en)
                            q, loss_q, c = quantizer_Audio(c)
                            y_g_hat = generator(q)
                            y_g_hat, Opera = attack(y_g_hat, [("CLP", 0.4), ("RSP-90", 0.10), ("Noise-W35", 0.15), ("SS-01", 0.05), ("AS-90", 0.05), ("EA-0301", 0.05), ("LP5000", 0.20)])
                            y_g_hat_mel = mel_spectrogram(y_g_hat.squeeze(1), h.n_fft, h.num_mels, h.sampling_rate,
                                                          h.hop_size, h.win_size,
                                                          h.fmin, h.fmax_for_loss)
                            sign_score, sign_g_hat = watermark_decoder(y_g_hat_mel)
                            audiomark_loss_tot += sign_loss(sign_score, sign)
                            i_size = min(y_mel.size(2), y_g_hat_mel.size(2))
                            val_err_tot += F.l1_loss(y_mel[:, :, :i_size], y_g_hat_mel[:, :, :i_size]).item()                          
                          
                            if j <= 8:
                                # if steps == 0:
                                if not plot_gt_once:
                                    sw.add_audio('gt/y_{}'.format(j), y[0], steps, h.sampling_rate)
                                    sw.add_figure('gt/y_spec_{}'.format(j), plot_spectrogram(x[0]), steps)

                                sw.add_audio('generated/y_hat_{}'.format(j), y_g_hat[0], steps, h.sampling_rate)
                                y_hat_spec = mel_spectrogram(y_g_hat.squeeze(1), h.n_fft, h.num_mels,
                                                             h.sampling_rate, h.hop_size, h.win_size,
                                                             h.fmin, h.fmax)
                                sw.add_figure('generated/y_hat_spec_{}'.format(j),
                                              plot_spectrogram(y_hat_spec.squeeze(0).cpu().numpy()), steps)
                                sign = ", ".join(str(num) for num in sign.view(-1).tolist())
                                sw.add_text("sign", sign, steps)
                                sign_g_hat = ", ".join(str(num) for num in sign_g_hat.view(-1).tolist())
                                sw.add_text("sign_g_hat", sign_g_hat, steps)
                        
                        audiomark_loss = audiomark_loss_tot / (j+1)
                        val_err = val_err_tot / (j+1)
                        sw.add_scalar("validation/mel_spec_error", val_err, steps)
                        sw.add_scalar("validation/audiomark_error", audiomark_loss, steps)
                        if not plot_gt_once:
                            plot_gt_once = True

                    generator.train()
                    watermark_decoder.train() 
                
                #if (steps + 1) % 100 == 0:
                #    pdb.set_trace() 
             
            steps += 1
            if steps == 150100:
                print("满足条件，程序终止。")
                os._exit()  

        scheduler_g.step()
        scheduler_d.step()

        if rank == 0:
            print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, int(time.time() - start)))


def main():
    print('Initializing Training Process..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_training_file', default='') 
    parser.add_argument('--input_validation_file', default='')
    parser.add_argument('--checkpoint_path', default='')
    parser.add_argument('--config', default='./config.json')
    parser.add_argument('--training_epochs', default=2000, type=int)
    parser.add_argument('--stdout_interval', default=5, type=int) # 5
    parser.add_argument('--checkpoint_interval', default=5000, type=int) # 20 5000
    parser.add_argument('--summary_interval', default=100, type=int) # 100 
    parser.add_argument('--validation_interval', default=500, type=int) # 20 1000 500
    parser.add_argument('--num_ckpt_keep', default=5, type=int) # 5
    parser.add_argument('--fine_tuning', default=False, type=bool)

    a = parser.parse_args()

    with open(a.config) as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)
    build_env(a.config, 'config.json', a.checkpoint_path)

    torch.manual_seed(h.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        h.num_gpus = torch.cuda.device_count()
        h.batch_size = int(h.batch_size / h.num_gpus)
        print('Batch size per GPU :', h.batch_size)
    else:
        pass

    if h.num_gpus > 1:
        mp.spawn(train, nprocs=h.num_gpus, args=(a, h,))
    else:
        train(0, a, h)


if __name__ == '__main__':
    main()
