"""Validate one aligned NIfTI pair supplied on the command line."""

import argparse

from synthetic_imaging_validation import load_pair, mae, psnr, ssim

parser = argparse.ArgumentParser()
parser.add_argument("real")
parser.add_argument("synthetic")
args = parser.parse_args()

real, synthetic = load_pair(args.real, args.synthetic)
print("Spacing:", real.spacing)
print("MAE:", mae(real.array, synthetic.array))
print("PSNR:", psnr(real.array, synthetic.array))
print("SSIM:", ssim(real.array, synthetic.array))

