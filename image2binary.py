# -*- coding: utf-8 -*-
import argparse
import os
import shutil

import cv2


def imgtobinary(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 5)
    # res2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 5)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def main(indir, outdir):
    shutil.rmtree(outdir)
    os.makedirs(outdir, exist_ok=False)
    for claz in sorted(os.listdir(indir)):
        indir_class = os.path.join(indir, claz)
        outdir_class = os.path.join(outdir, claz)
        os.makedirs(outdir_class, exist_ok=False)
        for fname in sorted(os.listdir(indir_class)):
            fp_in = os.path.join(indir_class, fname)
            print(f'processing {fp_in}')
            img = cv2.imread(fp_in)
            img = img[:, :, :3]  # remove A of BGRA
            img = imgtobinary(img)
            fname_out = os.path.splitext(fname)[0] + '.jpg'
            cv2.imwrite(os.path.join(outdir_class, fname_out), img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input')
    parser.add_argument('--output')
    args = parser.parse_args()
    main(args.input, args.output)
