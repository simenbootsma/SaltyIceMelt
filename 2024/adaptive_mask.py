from extract_contours import load_nef
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
from multiprocessing import Process, Queue
import time
from scipy.fft import fft, fftfreq

SCREEN_RES = (1920, 1200)
CAM_VIEW_BOX = []


def main():
    run()


def run():
    auto_calibration()
    # TODO: check if view box is correct, otherwise do manual calibration

    # STEP 1: calibrate, find the part of the screen that the camera sees, iteratively
    # STEP 2: find a target for the ice (e.g. by dilation)
    # STEP 3: display target on the screen

    pass


def auto_calibration():
    q = Queue()
    p0 = Process(target=show_mask, args=(q,))
    p1 = Process(target=auto_calib_worker, args=(q,))

    p0.start()
    p1.start()
    p0.join()
    p1.join()

    print("Calibration DONE!")


man_img = 255 * np.ones((SCREEN_RES[0], SCREEN_RES[1], 3))
man_points = []


def auto_calib_worker(queue):
    global CAM_VIEW_BOX

    # Grid for size calibration
    screen = np.ones(SCREEN_RES[::-1])
    sq_sz = 50
    for i in range(0, screen.shape[0], sq_sz):
        screen[i - 3:i + 3, :] = 0
    for j in range(0, screen.shape[1], sq_sz):
        screen[:, j-3:j+3] = 0
    queue.put(screen)

    time.sleep(1)

    img = take_synthetic_image(screen)

    ret, otsu = cv.threshold(img.astype(np.uint8), 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    otsu = cv.blur(otsu, (13, 13))

    ffts = [np.abs(fft(np.sum(otsu, axis=i)))[1:otsu.shape[1-i]//2] for i in [0, 1]]
    fftfreqs = [fftfreq(otsu.shape[1-i], 1)[1:otsu.shape[1-i]//2] for i in [0, 1]]
    zoom = [1/f[np.argmax(y)]/sq_sz for y, f in zip(ffts, fftfreqs)]
    print(zoom)

    # Image for camera view localization
    screen = cv.imread('polar_bear_mosaic.jpg', cv.IMREAD_UNCHANGED)
    # screen = cv.cvtColor(screen, cv.COLOR_BGR2GRAY)
    screen = cv.rotate(screen[50:-50, 50:612], cv.ROTATE_90_COUNTERCLOCKWISE)

    queue.put(screen)

    time.sleep(1)

    img = take_synthetic_image(screen)
    img = cv.resize(img, (int(img.shape[1]/zoom[0]), int(img.shape[0]/zoom[1])))  # use zoom to rescale
    # img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # screen = cv.cvtColor(screen, cv.COLOR_BGR2GRAY)

    locs = []
    rotation_codes = [None, cv.ROTATE_90_CLOCKWISE, cv.ROTATE_180, cv.ROTATE_90_COUNTERCLOCKWISE]
    for rot in rotation_codes:
        rot_img = img if rot is None else cv.rotate(img, rot)
        conv = cv.matchTemplate(screen, rot_img, cv.TM_SQDIFF)
        locs.append(cv.minMaxLoc(conv))  # min_val, max_val, min_loc, max_loc
    locs_sorted = sorted(locs, key=lambda tup: tup[0])
    opt_rot = rotation_codes[locs.index(locs_sorted[0])]
    top_left = locs_sorted[0][2]
    rot_img = img if opt_rot is None else cv.rotate(img, opt_rot)
    w, h = rot_img.shape[1], rot_img.shape[0]
    CAM_VIEW_BOX = [top_left[0], top_left[1], top_left[0] + w, top_left[1] + h]


def show_mask(queue):
    cv.namedWindow("window", cv.WINDOW_NORMAL)
    # cv.moveWindow("window", 900, 900)
    # cv.setWindowProperty("window", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
    img = np.ones(SCREEN_RES)
    while True:
        if not queue.empty():
            img = queue.get()
        cv.imshow("window", img)
        key = cv.waitKey(100)
        if key == 27:
            break
    cv.destroyWindow("window")


def manual_calibration():
    global man_img, man_points
    cv.namedWindow("window", cv.WINDOW_NORMAL)
    # cv.moveWindow("window", 900, 900)
    # cv.setWindowProperty("window", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)

    cv.setMouseCallback('window', draw_circle)
    while len(man_points) < 4:
        cv.imshow("window", man_img)
        key = cv.waitKey(10)
        if key == 27:
            break

    man_img = 255 * np.ones((SCREEN_RES[0], SCREEN_RES[1], 3))
    for p in man_points:
        cv.circle(man_img, p, 100, (0, 0, 255), 4)
        cv.circle(man_img, p, 10, (0, 0, 255), -1)
    cv.polylines(man_img, [np.reshape(np.array(man_points, dtype=np.int32), (-1, 1, 2))], True, (0, 0, 0), 30)
    cv.imshow("window", man_img)
    while True:
        key = cv.waitKey()
        if key == 27:
            break
    cv.destroyWindow("window")
    return man_points


def draw_circle(event, x, y, flags, param):
    global man_img, man_points
    if event == cv.EVENT_MOUSEMOVE:
        man_img = 255 * np.ones((SCREEN_RES[0], SCREEN_RES[1], 3))
        cv.circle(man_img, (x, y), 100, (255, 0, 0), -1)
        for p in man_points:
            cv.circle(man_img, p, 100, (0, 0, 255), 4)
            cv.circle(man_img, p, 10, (0, 0, 255), -1)
    elif event == cv.EVENT_LBUTTONUP:
        man_points.append([x, y])


""" 
    NOT USED 
"""


def calib_worker_old(queue):
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
    sz = (200, 500)
    r = (np.random.randint(screen.shape[0]-sz[0]),  np.random.randint(screen.shape[1] - sz[1]))
    crop = screen[r[0]:(r[0]+sz[0]), r[1]:(r[1]+sz[1])]
    img = cv.resize(crop, dsize=(crop.shape[1]*6, crop.shape[0]*6))
    img = cv.blur(img, (31, 31))
    rot = [None, cv.ROTATE_90_CLOCKWISE, cv.ROTATE_180, cv.ROTATE_90_COUNTERCLOCKWISE][np.random.randint(4)]
    img = img if img is None else cv.rotate(img, rot)
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



if __name__ == '__main__':
    main()


