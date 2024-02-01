import sys

import numpy as np
import cv2 as cv

"""
CONTROLS

    Esc : exit
    Back: reset
    f   : flip image
    m   : move left/right
    s   : sensitivity
    h   : height 
    w   : width
    b   : blur 
    k   : curvature
    c   : contrast
    .   : toggle dot pattern

    NOTE: shift + <key> will perform the inverse operation (when available)
"""


class Cylinder:
    def __init__(self, resolution=(920, 800)):
        self.resolution = resolution
        self.sensitivity = int(resolution[1] / 500)
        self.center = resolution[1] // 2
        self.width = resolution[1] // 3
        self.height = int(resolution[0] * 3 / 4)
        self.blur = 0
        self.curvature = self.width // 2
        self.flipped = False
        self.contrast = 1.0
        self.color_idx = 0
        self.color = (255, 255, 255)
        self.random_dot = False

    def move_left(self):
        self.center -= self.sensitivity

    def move_right(self):
        self.center += self.sensitivity

    def increase_width(self):
        self.width += self.sensitivity

    def decrease_width(self):
        self.width -= self.sensitivity

    def increase_height(self):
        self.height += self.sensitivity

    def decrease_height(self):
        self.height -= self.sensitivity

    def increase_blur(self):
        self.blur += 1

    def decrease_blur(self):
        self.blur = max(0, self.blur - 1)

    def increase_curvature(self):
        self.curvature += self.sensitivity

    def decrease_curvature(self):
        self.curvature = max(0, self.curvature - self.sensitivity)

    def increase_contrast(self):
        self.contrast = min(max(0, self.contrast + 0.01), 1)

    def decrease_contrast(self):
        self.contrast = min(max(0, self.contrast - 0.01), 1)

    def increase_sensitivity(self):
        self.sensitivity = min(self.resolution[1] // 3, self.sensitivity + 1)
        print("set sensivity to {:d} px".format(self.sensitivity))

    def decrease_sensivity(self):
        self.sensitivity = max(1, self.sensitivity - 1)
        print("set sensivity to {:d} px".format(self.sensitivity))

    def flip(self):
        self.flipped = not self.flipped

    def change_color(self):
        colors = [(255, 255, 255), (255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
        self.color_idx = 0 if self.color_idx == len(colors)-1 else self.color_idx + 1
        self.color = colors[self.color_idx]

    def toggle_random_dot(self):
        self.random_dot = not self.random_dot

    def handle_key(self, key):
        char = chr(key)
        func_map = {"m": self.move_left, "M": self.move_right, "w": self.increase_width, "W": self.decrease_width,
                    "h": self.increase_height, "H": self.decrease_height, "b": self.increase_blur,
                    "B": self.decrease_blur, ".": self.toggle_random_dot,
                    "k": self.increase_curvature, "K": self.decrease_curvature, "f": self.flip,
                    "c": self.increase_contrast, "C": self.decrease_contrast, '\x08': self.__init__,
                    "s": self.increase_sensitivity, "S": self.decrease_sensivity, "o": self.change_color}
        if char in func_map:
            func_map[char]()

    def get_img(self):
        if self.random_dot:
            return generate_random_dot_pattern(self.resolution).astype(np.uint8)

        img = np.zeros((self.resolution[1], self.resolution[0], 3))
        slice0 = slice(self.center - self.width // 2, self.center + self.width // 2)
        slice1 = slice(0, self.height - self.curvature)
        img[slice0, slice1, :] = self.color
        if self.curvature > 0:
            img = cv.ellipse(img, (self.height - self.curvature, self.center), (self.curvature, self.width // 2 - 1), 0,
                             0, 360, self.color, -1)
        if self.blur > 0:
            img = cv.blur(img.astype(np.uint8), (self.blur, self.blur))
        if self.flipped:
            img = np.fliplr(img)
        img = (img * self.contrast).astype(np.uint8)
        return img


class Sphere:
    def __init__(self, resolution=(1920, 1080)):
        self.resolution = resolution
        self.sensitivity = int(resolution[1] / 500)
        self.center = [resolution[0] // 2, resolution[1] // 2]
        self.radius = resolution[1] // 4
        self.blur = 0
        self.contrast = 1.0
        self.color_idx = 0
        self.color = (255, 255, 255)
        self.random_dot = False

    def move_left(self):
        self.center[1] -= self.sensitivity

    def move_right(self):
        self.center[1] += self.sensitivity

    def move_up(self):
        self.center[0] -= self.sensitivity

    def move_down(self):
        self.center[0] += self.sensitivity

    def increase_radius(self):
        self.radius += self.sensitivity

    def decrease_radius(self):
        self.radius -= self.sensitivity

    def increase_blur(self):
        self.blur += 1

    def decrease_blur(self):
        self.blur = max(0, self.blur - 1)

    def increase_contrast(self):
        self.contrast = min(max(0, self.contrast + 0.01), 1)

    def decrease_contrast(self):
        self.contrast = min(max(0, self.contrast - 0.01), 1)

    def increase_sensitivity(self):
        self.sensitivity = min(self.resolution[1] // 3, self.sensitivity + 1)
        print("set sensivity to {:d} px".format(self.sensitivity))

    def decrease_sensivity(self):
        self.sensitivity = max(1, self.sensitivity - 1)
        print("set sensivity to {:d} px".format(self.sensitivity))

    def change_color(self):
        colors = [(255, 255, 255), (255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
        self.color_idx = 0 if self.color_idx == len(colors)-1 else self.color_idx + 1
        self.color = colors[self.color_idx]

    def toggle_random_dot(self):
        self.random_dot = not self.random_dot

    def handle_key(self, key):
        char = chr(key)
        func_map = {"m": self.move_left, "M": self.move_right, "n": self.move_up, "N": self.move_down,
                    "r": self.increase_radius, "R": self.decrease_radius, "b": self.increase_blur,
                    "B": self.decrease_blur, ".": self.toggle_random_dot,
                    "c": self.increase_contrast, "C": self.decrease_contrast, '\x08': self.__init__,
                    "s": self.increase_sensitivity, "S": self.decrease_sensivity, "o": self.change_color}
        if char in func_map:
            func_map[char]()

    def get_img(self):
        if self.random_dot:
            return generate_random_dot_pattern(self.resolution).astype(np.uint8)

        img = np.zeros((self.resolution[1], self.resolution[0], 3))
        img = cv.ellipse(img, self.center, (self.radius, self.radius), 0, 0, 360, self.color, -1)
        if self.blur > 0:
            img = cv.blur(img.astype(np.uint8), (self.blur, self.blur))
        img = (img * self.contrast).astype(np.uint8)
        return img


def generate_random_dot_pattern(resolution):
    # random pixels
    m = np.random.randint(0, 255, (resolution[1], resolution[0]))
    img = np.stack([m, m, m], axis=2)

    # random dots
    img = 255 * np.ones((resolution[1], resolution[0], 3))
    dot_frac = 0.3
    min_r, max_r = 5, 10
    while np.sum(img == 0) / np.prod(img.shape) < dot_frac:
        r = np.random.randint(min_r, max_r)
        c = (np.random.randint(0, img.shape[1]), np.random.randint(0, img.shape[0]))
        cv.circle(img, c, r, (0, 0, 0), -1)

    return img


def main(args):
    if len(args) < 2 or args[1] not in ['cylinder', 'sphere']:
        print("\033[95m warning: Neither 'cylinder' nor 'sphere' given as argument, defaulting to 'cylinder'. \033[0m")
        args = ['', 'cylinder']
    obj = {'cylinder': Cylinder, 'sphere': Sphere}[args[1]]()

    cv.namedWindow("window", cv.WINDOW_NORMAL)
    # cv.moveWindow("window", 900, 900)
    # cv.setWindowProperty("window", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
    while True:
        cv.imshow("window", obj.get_img())
        key = cv.waitKey()
        if key == 27:
            break
        obj.handle_key(key)
    cv.destroyWindow("window")


if __name__ == "__main__":
    main(sys.argv)

