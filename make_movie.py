import cv2 as cv

import analysis
from load_settings import load_settings
import glob
import numpy as np
from tif_to_mask import read_tif


def folder_to_raw_movie(k):
    folder = "/path/to/data/exp_{:s}/JPG"  # folder containing JPG images for boundary tracking
    settings = load_settings(k)
    save_path = "save_path.avi"

    files = glob.glob(folder + r"\*.jpg")

    HOR_SLICE = slice(settings["hcrop"][0], settings["hcrop"][1])
    VERT_SLICE = slice(settings["vcrop"][0], settings["vcrop"][1])

    dim_before = (HOR_SLICE.stop - HOR_SLICE.start, VERT_SLICE.stop - VERT_SLICE.start)
    out = cv.VideoWriter(save_path, cv.VideoWriter_fourcc(*'DIVX'), 10, dim_before)
    for i, fn in enumerate(files):
        img = cv.imread(fn)

        # # rotate
        # img = rotate_image(img, param["angle"])

        # crop
        img = img[VERT_SLICE, HOR_SLICE]
        img = img.astype(np.uint8)

        # write
        out.write(img)
        print("\r[{:s}] processed {:d}/{:d}".format(k, i + 1, len(files)), end='')
    out.release()


def folder_to_raw_movie_piv(k):
    folder = "/path/to/piv/exp_{:s}"        # folder containing DAT and TIF files for PIV
    save_path = "save_path.avi"

    files = glob.glob(folder + r"\*.TIF")
    streak_sz = 25
    y0 = 100  # skip top ... pixels
    yN = 20  # skip bottom ... pixels
    dim = (1024, 1024-y0-yN)
    out = cv.VideoWriter(save_path, cv.VideoWriter_fourcc(*'DIVX'), 25, dim)
    edges = analysis.compute_average_mask(k)
    edges[:, 0] = analysis.smoothen_array(edges[:, 0])
    edges[:, 1] = 1024 - analysis.smoothen_array(edges[:, 1])
    melt = analysis.piv_melt_rate(k)
    adv = analysis.piv_advection(k)

    img = np.zeros((1024, 1024))
    for i, fn in enumerate(files):
        img += read_tif(files[i])
        if i >= streak_sz:
            img -= read_tif(files[i-streak_sz])
        if i % 10 == 0:
            x = edges[:, 0] + melt * (i - 500)
            y = edges[:, 1] - adv * (i - 500)

            # crop
            # img = img[VERT_SLICE, HOR_SLICE]
            img2 = 255 * img.astype(np.uint8)
            img2[img2 > 255] = 255
            img2[img2 < 0] = 0
            if k == 'piv1':
                img2 = np.fliplr(img2)
            for j in range(len(x)):
                if 0 <= x[j] < 1024 and 0 <= y[j] < 1024:
                    xw = int(x[j])
                    img2[int(y[j]), :xw] = 125
                    img2[int(y[j]), xw-2:xw+3] = 50
            img_rgb = np.zeros((1024, 1024, 3), dtype=np.uint8)
            for j in range(3):
                img_rgb[:, :, j] = img2

            # crop
            img_rgb = img_rgb[y0:-yN, :, :]

            # insert text
            cv.putText(img_rgb, '5x speed', (20, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            # cv.imshow('name', img_rgb)
            # cv.waitKey(0)

            # write
            out.write(img_rgb)
            print("\r[{:s}] processed {:d}/{:d}".format(k, i + 1, len(files)), end='')
    out.release()


if __name__ == '__main__':
    pass

