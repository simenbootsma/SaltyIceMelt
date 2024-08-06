from extract_contours import load_nef
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
from multiprocessing import Process, Queue
import time


SCREEN_RES = (1920, 1080)


def main():
    # real_image_test()
    calibrate()


def run():
    # STEP 1: calibrate, find the part of the screen that the camera sees, iteratively
    # STEP 2: find a target for the ice (e.g. by dilation)
    # STEP 3: display target on the screen

    pass


def calibrate():
    q = Queue()
    p0 = Process(target=show_mask, args=(q,))
    p1 = Process(target=calib_worker, args=(q,))

    p0.start()
    p1.start()
    p0.join()
    p1.join()

    print("Calibration DONE!")


def calib_worker(queue):
    """ Use binary search to find camera view boundaries """
    screen = np.ones(SCREEN_RES)
    queue.put(screen)
    time.sleep(1)

    img = take_synthetic_image(screen)
    white = np.mean(img)

    # HORIZONTAL, VERTICAL
    flipped = [False, False]
    center = [0, 0]
    size = [0, 0]
    for direc in [1, 0]:
        screen = np.ones(SCREEN_RES)
        x = SCREEN_RES[direc] // 2
        prev_x = 0
        make_black = True
        prev_white_frac = 1
        for i in range(8):
            if direc == 0:
                screen[min(prev_x, x):max(prev_x, x), :] = 0 if make_black else 1
            else:
                screen[:, min(prev_x, x):max(prev_x, x)] = 0 if make_black else 1
            queue.put(screen)

            time.sleep(.1)
            img = take_synthetic_image(screen)
            bn = np.where(img < .5 * white, 0, 1)
            white_frac = np.sum(bn) / np.prod(img.shape)

            if 0.1 < white_frac < 0.9 and 0.1 < prev_white_frac < 0.9 and size[direc] == 0:
                size[direc] = int(abs(x - prev_x) / abs(white_frac - prev_white_frac))
            prev_x = x
            prev_white_frac = white_frac

            make_black = white_frac > 0.5
            x += int((-1)**int(not make_black) * 2**(-i-2) * SCREEN_RES[direc])
        flipped[direc] = img[0, 0] == 1  # if left top is white on camera, screen is flipped
        center[direc] = x

    # Show calibration on screen
    image = 255 * np.ones((SCREEN_RES[0], SCREEN_RES[1], 3))
    p0 = (center[1] - size[0]//2, center[0] - size[1]//2)
    p1 = (center[1] + size[0] // 2, center[0] + size[1] // 2)
    image = cv.rectangle(image, p0, p1, (255, 0, 0), 10)
    image = cv.putText(image, 'top', (center[1] - size[0]//4, center[0] - size[1]//2), cv.FONT_HERSHEY_PLAIN,
                        5, (0, 0, 0), 5, cv.LINE_AA)
    image = cv.putText(image, 'Press ESC if happy', (100, 100), cv.FONT_HERSHEY_PLAIN,
                       4, (0, 0, 0), 5, cv.LINE_AA)
    queue.put(image)


def take_synthetic_image(screen):
    crop = screen[1000:1700, 200:700]
    img = cv.resize(crop, dsize=(crop.shape[0]*5, crop.shape[1]*5))
    img = cv.blur(img, (51, 51)).T
    return img


def real_image_test():
    img = load_nef("/Users/simenbootsma/Pictures/HC test 5 jul/masked_300mm/DSC_3950.nef")
    plt.imshow(img)

    mask, ice = find_mask_and_ice(img)

    plt.figure()
    plt.imshow(img)

    plt.figure()
    plt.imshow(mask)
    # c, r = find_mask_circle(mask)
    # plt.gca().add_artist(plt.Circle(c, r, fc=None, ec='r'))

    plt.figure()
    plt.imshow(ice)

    x, y = np.meshgrid(np.arange(ice.shape[1]), np.arange(ice.shape[0]))
    ice_center = (np.mean(x[ice > 0]), np.mean(y[ice > 0]))
    plt.gca().add_artist(plt.Circle(ice_center, 50, fc='r', ec='r'))
    plt.show()

    it, sz = 5, 71
    target = cv.dilate(ice, kernel=np.ones((sz, sz)), iterations=it)

    plt.figure()
    plt.imshow(target + ice)


def find_mask_circle(mask):
    """ Finds center and radius of the circle displayed on the screen, as observed by the camera in pixels"""
    if np.all(np.concatenate([mask[0, :], mask[:, 0].T, mask[-1, :], mask[:, -1].T]) > 0):
        # full circle is within image boundaries
        r = np.sqrt(np.sum(1-mask)/np.pi)
        x, y = np.meshgrid(np.arange(mask.shape[0]), np.arange(mask.shape[1]))
        c = (np.average(x[mask==0]), np.average(y[mask==0]))
    elif np.all(np.array([mask[0, :], mask[-1, :]]) > 0):
        # circle is within top and bottom image boundaries
        r = np.max(np.sum(1-mask, axis=0))/2
        cx = np.argmax(np.sum(1-mask, axis=0))/2
        cy = np.mean(np.argwhere(mask[:, cx] == 0))
        c = (cx, cy)
    elif np.all(np.array([mask[:, 0], mask[:, -1]]) > 0):
        # circle is within left and right image boundaries
        r = np.max(np.sum(1-mask, axis=1))/2
        cy = np.argmax(np.sum(1-mask, axis=1))/2
        cx = np.mean(np.argwhere(mask[cy, :] == 0))
        c = (cx, cy)
    else:
        raise ValueError("circle not visible within any image bounds")
    return c, r


def find_mask_and_ice(img):
    gray = np.mean(img, axis=2)
    blur = cv.GaussianBlur(gray, (5, 5), sigmaX=0)
    ret, otsu = cv.threshold(blur.astype(np.uint8), 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    print(ret)

    s0, s1 = otsu.shape
    mask = np.zeros(otsu.shape)
    edges = [(i, j) for i in range(s0) for j in range(s1) if (i%(s0-1))*(j%(s1-1)) == 0]
    for i, j in edges:
        if mask[i, j] == 0 and otsu[i, j] == 0:
            empty_mat = np.zeros((s0 + 2, s1 + 2), dtype=np.uint8)
            _, _, m, _ = cv.floodFill(otsu.copy(), empty_mat, (j, i), 0)
            mask[m[1:-1, 1:-1] == 1] = 1
    mask = cv.dilate(mask, kernel=np.ones((31, 31)))
    ice = (1 - otsu.copy()/255)
    ice[mask==1] = 0
    return mask, ice


def synthetic_ellipse():
    resolution = (3600, 5408)
    img = np.zeros(resolution)

    mid = (resolution[0]/2, resolution[1]/2)
    for i in range(resolution[0]):
        for j in range(resolution[1]):
            if (i-mid[0])**2 + 0.4*(j - mid[1])**2 < (resolution[0]*0.25)**2:
                img[i, j] = 1
    return img


def show_mask(queue):
    cv.namedWindow("window", cv.WINDOW_NORMAL)
    # cv.moveWindow("window", 900, 900)
    # cv.setWindowProperty("window", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
    img = np.ones((700, 1000))
    while True:
        if not queue.empty():
            img = queue.get()
        cv.imshow("window", img)
        key = cv.waitKey(100)
        if key == 27:
            break
    cv.destroyWindow("window")


if __name__ == '__main__':
    main()


