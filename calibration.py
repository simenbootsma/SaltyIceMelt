import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import sys


def calibrate_image(filepath):
    DOT_DIST = 0.5  # distance between dots in the same row or column [cm]
    MIN_RADIUS = 15  # pixels
    MAX_RADIUS = 30  # pixels
    BIN_THRESH = 60  # binary threshold [0-255]

    # Load and compress image
    img = cv.imread(filepath) * 3
    img = rotate_image(img, angle=-1.4)
    gray = cv.cvtColor(img, cv.COLOR_RGBA2GRAY)
    fac = 3
    coarse = cv.resize(img, (img.shape[1]//fac, img.shape[0]//fac))
    # Choose bounding box to analyze
    fig, ax = plt.subplots()
    ax.imshow(coarse)
    ax.set_title('use RIGHT button to select top left and bottom right corners of the bounding box')
    # points = plt.ginput(n=4, timeout=-1, mouse_add=plt.MouseButton.RIGHT, mouse_pop=None)
    # x, y = sorted([points[i][0] for i in range(4)]), sorted([points[i][1] for i in range(4)])
    points = plt.ginput(n=2, timeout=-1, mouse_add=plt.MouseButton.RIGHT, mouse_pop=None)
    x, y = sorted([points[i][0] for i in range(2)]), sorted([points[i][1] for i in range(2)])
    x_slice = slice(int(fac * x[0]), int(fac * x[1]))
    y_slice = slice(int(fac * y[0]), int(fac * y[1]))
    ax.add_artist(plt.Rectangle((x[0], y[0]), width=x[1]-x[0], height=y[1]-y[0], facecolor='none', edgecolor='b'))
    ax.set_title('Close window to continue')
    plt.show()

    # Cut out bounding box
    crop = gray[y_slice, x_slice]

    # Binarize
    ret, glob_bin = cv.threshold(crop, BIN_THRESH, 255, cv.THRESH_BINARY)

    # Close gaps in binary image
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, ksize=(7, 7))
    closed = cv.morphologyEx(glob_bin, cv.MORPH_CLOSE, kernel, iterations=1)

    plt.figure()
    plt.imshow(closed)
    plt.show()

    # Find the circles
    circles = find_circleCenters(closed)
    circles = list(filter(lambda c: MIN_RADIUS < c["r"] < MAX_RADIUS, circles))

    # plt.figure()
    # plt.hist([c["r"] for c in circles])
    # plt.show()

    # Find the grid
    avg_r = np.mean([c["r"] for c in circles])
    cx = [c["x"] for c in circles]
    cxi = np.argsort(cx)
    cx = np.sort(cx)
    cy = [c["y"] for c in circles]
    cyi = np.argsort(cy)
    cy = np.sort(cy)
    row, col = 0, 0
    vert_div, horz_div = [], []
    for i in range(len(circles)):
        if i > 0 and cx[i] > cx[i-1] + avg_r:
            row += 1
            vert_div.append((cx[i]+cx[i-1])/2)
        if i > 0 and cy[i] > cy[i-1] + avg_r:
            col += 1
            horz_div.append((cy[i] + cy[i - 1]) / 2)
        circles[cxi[i]]["row"] = row
        circles[cyi[i]]["col"] = col

    # Compute calibration result
    values = []  # number of pixels per cm for each pair of dots
    weights = []  # distance in pixels between pairs of dots
    for i in range(len(circles)):
        for j in range(i+1, len(circles)):
            pixel_dist = np.sqrt((circles[i]["x"] - circles[j]["x"])**2 + (circles[i]["y"] - circles[j]["y"])**2)
            cm_dist = DOT_DIST * np.sqrt((circles[i]["row"] - circles[j]["row"])**2 + (circles[i]["col"] - circles[j]["col"])**2)
            values.append(pixel_dist/cm_dist)
            weights.append(pixel_dist)
    pixels_per_cm = np.average(values, weights=weights)
    std_dev = np.sqrt(np.average((np.array(values) - pixels_per_cm)**2, weights=weights))

    # Show resulting grid with circles
    fig, ax = plt.subplots()
    ax.imshow(crop)
    for c in circles:
        ax.add_artist(plt.Circle((c["x"], c["y"]), radius=c["r"], facecolor='none', edgecolor='r'))
        ax.add_artist(plt.Rectangle((c["x"] - 4, c["y"] - 4), width=8, height=8, angle=45, rotation_point='center',
                                    facecolor=(0.6, 0, 0)))
    for hd in horz_div:
        ax.add_artist(plt.Line2D([0, closed.shape[1]], [hd, hd]))
    for vd in vert_div:
        ax.add_artist(plt.Line2D([vd, vd], [0, closed.shape[0]]))
    ax.set_title("{:.1f} +/- {:.1f} pixels per cm".format(pixels_per_cm, std_dev))

    plt.figure()
    plt.hist(values)
    plt.show()


def find_circleCenters(bin_img, min_area=100):
    blobs = []
    mask = np.zeros(bin_img.shape)
    for i in range(bin_img.shape[0]):
        if i % (bin_img.shape[0] // 10) == 0:
            print("\rFinding circles at {:.0f}%".format(100*i/bin_img.shape[0]), end='')
        for j in range(bin_img.shape[1]):
            if bin_img[i, j] and not mask[i, j]:
                m = mask_image(bin_img, start_pos=(i, j))
                n_pixels_on = np.sum(m)
                if n_pixels_on >= min_area:
                    blob_x = np.sum(np.sum(m, axis=0) * np.arange(m.shape[1])) / n_pixels_on
                    blob_y = np.sum(np.sum(m, axis=1) * np.arange(m.shape[0])) / n_pixels_on
                    blob_radius = np.sqrt(n_pixels_on / np.pi)
                    blobs.append({"x": blob_x, "y": blob_y, "r": blob_radius})
                mask += m
    print('\nDone!')
    return blobs


def mask_image(img, box_size=1, start_pos=None):
    if start_pos is None:
        start_pos = (0, img.shape[1]//2)

    # Declare functions
    def get_neighbours(tup, shape):
        x, y = tup
        adjacent = [(x - 1, y), (x + 1, y), (x - 1, y - 1), (x, y - 1), (x + 1, y - 1), (x - 1, y + 1),
                    (x, y + 1), (x + 1, y + 1)]
        return [(ix, iy) for ix, iy in adjacent if (0 <= ix < shape[0]) and (0 <= iy < shape[1])]

    # Find edges of initial mask, if available
    mask0 = np.zeros(img.shape, dtype=np.uint8)
    q = [start_pos]

    # Mask image
    cnt = 0
    max_val = 255 * box_size ** 2
    while len(q) > 0:
        p = q.pop(0)
        mask0[p[0], p[1]] = 1
        neighbours = get_neighbours(p, shape=mask0.shape)
        q += [nb for nb in neighbours if
              (img[nb[0], nb[1]] == max_val) and (not mask0[nb[0], nb[1]]) and (nb not in q)]
        cnt += 1
    # print("done in {:d} iterations".format(cnt))
    return mask0


def rotate_image(img, angle):
    # angle >> negative: clockwise, positive: counter-clockwise
    rows, cols = img.shape[0], img.shape[1]
    M = cv.getRotationMatrix2D((cols/2, rows/2), angle, 1)
    img = cv.warpAffine(img, M, (cols, rows))
    return img


if __name__ == "__main__":
    calibrate_image(sys.argv[1])

