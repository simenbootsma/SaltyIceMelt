import cv2 as cv
import numpy as np
import glob
import pickle
import time


PIV_FOLDER = "/path/to/piv/exp_{:s}"        # folder containing DAT and TIF files for PIV


def main():
    min_frames = 500  # minimum number of frames to make a mask
    dframes = 10  # save every ... frame
    for k in '1234':
        k = 'piv' + k
        folder = PIV_FOLDER.format(k)
        fnames = glob.glob(folder + r'\*.tif')

        st = time.time()
        mask = read_tif(fnames[0])
        for i in range(1, len(fnames)):
            mask += read_tif(fnames[i])
            if i >= min_frames and i % dframes == 0:
                save_mask(process_mask(mask), k, i)

            # show progress
            tm = (time.time() - st) / (i + 1) * (len(fnames) - i - 1)
            tm_h, tm_m, tm_s = int(tm / 3600), int((tm % 3600) / 60), int(tm % 60)
            print("\r[{:s}] masking: {:d}/{:d} (done in {:02d}:{:02d}:{:02d})".format(k, i + 1, len(fnames), tm_h, tm_m, tm_s), end='')


def read_tif(filepath):
    thresh = 0.13 if 'piv1' in filepath or 'piv4' in filepath else 0.07
    img = cv.imread(filepath, cv.IMREAD_UNCHANGED)

    img = img.astype(np.float64)
    img = img / np.max(img)  # normalize
    img[img >= thresh] = 1  # threshold up
    img[img < thresh] = 0  # threshold down

    if np.sum(img) > 1e5:
        img[:] = 0  # flash of laser light, skip this frame
    return img


def process_mask(msk):
    msk[msk > 0] = 1                        # turn to binary
    msk = msk.astype(np.uint8)              # convert to uint8
    msk = cv.blur((1 - msk), (100, 10))     # blur to fill gaps
    msk[msk > .5] = 1                       # threshold up
    msk[msk <= .5] = 0                      # threshold down
    return msk.astype(bool)


def save_mask(mask, k, i):
    save_path = PIV_FOLDER + r'\masks\masks_{:s}\masks_{:s}_{:05d}.pkl'.format(k, k, i)
    with open(save_path, 'wb') as f:
        pickle.dump(mask, f)


if __name__ == "__main__":
    main()

