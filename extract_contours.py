import time
import glob
import cv2 as cv
import numpy as np
import pickle
from load_settings import load_settings, ALL_KEYS


DATA_FOLDER = "/path/to/data/exp_{:s}/JPG"  # folder containing JPG images for boundary tracking


def main():
    settings = load_settings('all')
    for k in ALL_KEYS:
        folder = DATA_FOLDER.format(k)
        contours = process_experiment(folder, settings[k])

        with open('contours/contours{:s}.pkl'.format(k), 'wb') as f:
            pickle.dump(contours, f)
    print("DONE")


def process_experiment(folder, settings):
    filepaths = glob.glob(folder + "/*.jpg")
    N = len(filepaths)
    contours = []
    st = time.time()
    for n in range(N):
        # Process image
        img = process(filepaths[n], settings)

        # Find edges
        try:
            edges = find_edges(img)
            contours.append(edges)
        except IndexError:
            return

        perc = (n+1)/N * 100
        ip = int(perc)
        etc_sec = (time.time() - st) * (N/(n+1) - 1)
        etc = "{:02d}:{:02d}:{:02d}".format(int(etc_sec/3600), int((etc_sec % 3600)/60), int(etc_sec % 60))
        print("\r{:s}:  [".format(settings["id"]) + "#"*(ip//4)+"-"*(25-ip//4)+"] {:.1f}%   done in {:s}".format(perc, etc), end='')
    print("")
    return contours


def process(filepath, settings):
    # Load image
    img = cv.imread(filepath)

    if img is None:
        return None

    img = rotate_image(img, angle=settings["rot"])  # rotate such that sides are perfectly vertical
    gray = cv.cvtColor(img, cv.COLOR_RGBA2GRAY)     # convert to grayscale

    # Crop to size
    HOR_SLICE = slice(settings["hcrop"][0], settings["hcrop"][1])
    VERT_SLICE = slice(settings["vcrop"][0], settings["vcrop"][1])
    crop = gray[VERT_SLICE, HOR_SLICE]

    # Global binarization
    ret, glob_bin = cv.threshold(crop, settings["binthresh"], 255, cv.THRESH_BINARY)
    inverted = 255 - glob_bin

    # Close
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, ksize=(5, 5))
    closed = cv.morphologyEx(inverted, cv.MORPH_CLOSE, kernel, iterations=settings["CL_ITER"])
    processed = closed

    # Manual tweaking
    if 'a2' in filepath or 'a4' in filepath:
        processed = inverted
        processed[3500:4000, :] = closed[3500:4000, :]
    elif 'a6' in filepath:
        processed = inverted
        processed[2800:3800, :] = closed[2800:3800, :]
    return processed.astype(np.uint8)


def find_edges(img):
    skip_largest = np.sum(img[:, 0] > 0) > 0.9 * img.shape[0] or np.sum(img[:, -1] > 0) > img.shape[0]
    if skip_largest:
        # connect both sides if either the left or right side is black
        img[-1, :] = 255

    cont, hierarchy = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    order = 1 if skip_largest else 0  # largest contour is the background for experiments 4 and 5
    idx = np.argsort([np.max(c[:,0,1]) - np.min(c[:,0,1]) for c in cont])[::-1][order]
    edges = np.reshape(cont[idx], (cont[idx].shape[0], 2))
    end_idx = [i for i in range(edges.shape[0]//10, edges.shape[0]) if edges[i, 1] == 0]
    if len(end_idx) > 0:
        edges = edges[:end_idx[0] + 1]  # skip the base of the cylinder
    return edges


def rotate_image(img, angle):
    # angle >> negative: clockwise, positive: counter-clockwise
    rows, cols = img.shape[0], img.shape[1]
    M = cv.getRotationMatrix2D((cols/2, rows/2), angle, 1)
    img = cv.warpAffine(img, M, (cols, rows))
    return img


if __name__ == "__main__":
    main()

