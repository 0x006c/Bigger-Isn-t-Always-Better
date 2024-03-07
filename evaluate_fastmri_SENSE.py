from pathlib import Path
from models import utils as mutils
from sde_lib import VESDE
from sampling import (ReverseDiffusionPredictor,
                      LangevinCorrector,
                      get_pc_fouriercs_RI,
                      get_pc_fouriercs_fast)
from models import ncsnpp
import time
from utils import fft2, ifft2, get_mask, get_data_scaler, get_data_inverse_scaler, restore_checkpoint, normalize_complex, normalize, ifft2_m, fft2_m, SSIM, nmse, psnr, get_radial_mask, get_outer_mask
import torch
import torch.nn as nn
import numpy as np
from models.ema import ExponentialMovingAverage
import matplotlib.pyplot as plt
import importlib
import argparse
import h5py
from torchvision.transforms.functional import center_crop
import mydata
import os
from sigpy.mri.app import L1WaveletRecon, TotalVariationRecon


def main():
    ###############################################
    # 1. Configurations
    ###############################################

    # args
    args = create_argparser().parse_args()
    if args.complex.lower() in ('true', 't', 'yes', 'y', '1'):
        use_complex = True
    else:
        use_complex = False
    if args.fat_suppression.lower() in ('true', 't', 'yes', 'y', '1'):
        fat_suppression = True
    else:
        fat_suppression = False
    if args.brain.lower() in ('true', 't', 'yes', 'y', '1'):
        brain = True
        assert fat_suppression == False
    else:
        brain = False

    print('initaializing...')
    configs = importlib.import_module(f'configs.ve.fastmri_knee_4_attention')
    config = configs.get_config()
    img_size = config.data.image_size
    batch_size = 1

    np.random.seed(config.seed)
    if args.mask_type == 'radial':
        mask = get_radial_mask((320, 320), 29, np.pi / 29)
    elif args.mask_type == 'outer':
        mask = get_outer_mask(args.cutout)
    else:
        mask = get_mask(torch.zeros((1, 1, 320, 320)), img_size, batch_size,
                        type=args.mask_type,
                        acc_factor=args.acc_factor,
                        center_fraction=args.center_fraction).to(config.device)
    


    # Specify save directory for saving generated samples
    print(args.center_fraction)
    save_root = Path(f'./results/{args.prior}' + ('' if args.mask_type == 'gaussian1d' else f'_{args.mask_type}') + (f'_{args.cutout}' if args.mask_type == 'outer' else '') + ('' if args.center_fraction == 0.08 else f'_CF_0{str(args.center_fraction)[2:]}') + ('' if args.acc_factor == 4 else f'_AF_{args.acc_factor}') + ('_FS' if fat_suppression else '') + ('_brain' if brain else '') + 'no_norm')
    save_root.mkdir(parents=True, exist_ok=True)

    mask_sv = mask.squeeze().cpu().detach().numpy()

    np.save(str(save_root) + '/mask.npy', mask_sv)
    plt.imsave(str(save_root) + '/mask.png', mask_sv, cmap='gray')

    lambdas = [0.001, 0.0025, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.075]


    ssim = SSIM()
    if brain:
        test_dl = mydata.create_brain_dataloader()
    else:
        _, test_dl = mydata.create_dataloader(config, fat_suppression=fat_suppression)
    psnr_values = np.zeros((len(test_dl), len(lambdas)))
    ssim_values = np.zeros((len(test_dl), len(lambdas)))
    nmse_values = np.zeros((len(test_dl), len(lambdas)))


    ###############################################
    # 2. Inference
    ###############################################
    
    for i, img in enumerate(test_dl):
        print(f'reconstructing slice {i + 1} of {len(test_dl)}')
        # fft
        # plt.imsave(str(save_root) + f'/input_{i}.png', img.squeeze().cpu().detach().numpy(), cmap='gray')
        img = img.view(1, 1, 320, 320)
        img = img.to(config.device)
        kspace = fft2(img)

        # undersampling
        under_kspace = (kspace * mask).view(1, 320, 320).numpy()
        mps = np.ones((1, img_size, img_size))

        for j, lam in enumerate(lambdas):
            if args.prior == 'l1':
                x = L1WaveletRecon(under_kspace, mps, lam).run()
            else:
                x = TotalVariationRecon(under_kspace, mps, lam).run()
            #x = normalize(torch.from_numpy(np.abs(x)).view(1, 1, 320, 320))
            x = torch.from_numpy(np.abs(x)).view(1, 1, 320, 320)
            print(psnr(x, img))
            psnr_values[i, j] = psnr(x, img)
            nmse_values[i, j] = nmse(x, img)
            ssim_values[i, j] = ssim(x, img)
            recon = x.squeeze().cpu().detach().numpy()
            plt.imsave(str(save_root) + f'/recon_l_{lam}_{i}.png', recon, cmap='gray')
        np.savetxt(os.path.join(save_root, 'psnr_values.csv'), psnr_values, delimiter=',')
        np.savetxt(os.path.join(save_root, 'nmse_values.csv'), nmse_values, delimiter=',')
        np.savetxt(os.path.join(save_root, 'ssim_values.csv'), ssim_values, delimiter=',')




def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--complex', type=str, help='Whether reconstructing complex- or real-valued data.', default='False')
    parser.add_argument('--fat_suppression', type=str, help='Whether reconstructing CORPDFS or CORPD data.', default='False')
    parser.add_argument('--brain', type=str, help='Whether reconstructing brain data.', default='False')
    parser.add_argument('--prior', type=str, help='which prior to use', required=True, choices=['l1', 'TV'])
    parser.add_argument('--mask_type', type=str, help='which mask to use for retrospective undersampling.'
                                                      '(NOTE) only used for retrospective model!', default='gaussian1d',
                        choices=['gaussian1d', 'uniform1d', 'gaussian2d', 'poisson', 'radial', 'outer'])
    parser.add_argument('--cutout', type=int, help='Size of the cutout if outer mask is used.', default=0)
    parser.add_argument('--acc_factor', type=int, help='Acceleration factor for Fourier undersampling.'
                                                       '(NOTE) only used for retrospective model!', default=4)
    parser.add_argument('--center_fraction', type=float, help='Fraction of ACS region to keep.'
                                                       '(NOTE) only used for retrospective model!', default=0.08)
    parser.add_argument('--save_dir', default='./results')
    return parser


if __name__ == "__main__":
    main()