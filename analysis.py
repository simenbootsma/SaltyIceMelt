import numpy as np
import pickle
from load_settings import load_settings, ALL_KEYS
from Seawater import Seawater
import cv2 as cv
import glob
from PIL import Image
from scipy.stats import linregress

SETTINGS = load_settings('all')             # load processing settings and experimental details from processing_settings.xlsx
PIV_FOLDER = "/path/to/piv/exp_{:s}"        # folder containing DAT and TIF files for PIV
DATA_FOLDER = "/path/to/data/exp_{:s}/JPG"  # folder containing JPG images for boundary tracking
DEFAULT_CACHE = True                        # whether to load values from cache by default


""""
    UTILS
"""


def dump_to_cache(obj, name):
    with open('cache/{:s}.pkl'.format(name), 'wb') as f:
        pickle.dump(obj, f)


def get_from_cache(name):
    try:
        with open('cache/{:s}.pkl'.format(name), 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        print("Could not find '{:s}' in cache.".format(name))
        return


def get_contours(k):
    """ Returns contours for experiment <k> """
    with open('contours/contours_{:s}.pkl'.format(k), 'rb') as f:
        cntr = pickle.load(f)
    return cntr


def smoothen_contour(contour, n=50):
    # NOTE: requires contour to be ordered
    X, Y = list(zip(*contour))
    X = [np.mean(X[max(0, i - n // 2):min(len(X), i + n // 2)]) for i in range(len(X))]
    Y = [np.mean(Y[max(0, i - n // 2):min(len(Y), i + n // 2)]) for i in range(len(Y))]
    return list(zip(X, Y))


def smoothen_array(Y, n=50):
    Y = [np.nanmean(Y[max(0, i - n // 2):min(len(Y), i + n // 2)]) for i in range(len(Y))]
    return np.array(Y)


def bin_contour(contour, n=50):
    # NOTE: requires contour to be ordered
    X, Y = list(zip(*contour))
    X = [np.mean(X[i:i+n]) for i in range(0, len(X), n)]
    Y = [np.mean(Y[i:i+n]) for i in range(0, len(Y), n)]
    return list(zip(X, Y))


def find_time_offset(k, from_cache=DEFAULT_CACHE):
    """ Computes index at which experiment <k> should have its t=0 """
    if from_cache:
        return get_from_cache("time_offset_" + k)
    all_keys = [k2 for k2 in ALL_KEYS if k2[0] == k[0]]
    min_radius = np.min([compute_average_radius(get_contours(k)[0]) / SETTINGS[k]["Ccal"] for k in all_keys])
    radii = np.array([compute_average_radius(c) for c in get_contours(k)]) / SETTINGS[k]["Ccal"]
    toff = np.argmin(np.abs(radii - min_radius))
    dump_to_cache(toff, "time_offset_"+k)
    return toff


def find_half_volume_index(k, from_cache=DEFAULT_CACHE):
    """ Computes contour index at which the cylinder's volume has been halved since t=0 """
    if from_cache:
        return get_from_cache("half_volume_index_" + k)
    toff = find_time_offset(k)
    volumes = [compute_volume(c, ccal=SETTINGS[k]["Ccal"]) for c in get_contours(k)]
    normV = np.array(volumes) / volumes[toff]
    ih = len(normV[normV > 0.5]) - 1  # index at which volume is halved
    dump_to_cache(ih, "half_volume_index_" + k)
    return ih


def find_half_volume_time(k, from_cache=DEFAULT_CACHE):
    """ Computes time in seconds at which the cylinder's volume has been halved since t=0 """
    if from_cache:
        return get_from_cache("half_volume_time_" + k)
    toff = find_time_offset(k)
    dt = SETTINGS[k]['dt']
    volumes = [compute_volume(c, ccal=SETTINGS[k]["Ccal"]) for c in get_contours(k)]
    normV = np.array(volumes) / volumes[toff]
    ih = len(normV[normV > 0.5]) - 1  # index at which volume is halved

    if k == 'c2':
        # camera died for 1316 seconds between frames 532 and 533 :'(
        t_half = (ih - toff) * dt + (normV[ih] - 0.5) / (normV[ih] - normV[ih + 1]) * 1316
    else:
        t_half = (ih - toff + (normV[ih] - 0.5) / (normV[ih] - normV[ih + 1])) * dt  # time at which volume is halved
    dump_to_cache(t_half, "half_volume_time_" + k)
    return t_half


def find_times(k, from_cache=DEFAULT_CACHE):
    if from_cache:
        return get_from_cache('time_' + k)

    folder = DATA_FOLDER.format(k)
    filepaths = glob.glob(folder + "/*.JPG")

    t = []
    for fn in filepaths:
        if 'c5' in folder or 'b1' in folder:
            # image number exceeded 9999
            fn = fn.replace('99990', '0')
            fn = fn.lower().replace('jpg', 'NEF')
            exif = Image.open(fn).getexif()
        else:
            exif = Image.open(fn)._getexif()

        t_str = exif[36867].split(' ')[1]
        seconds = np.sum([int(s) * 60 ** (2 - i) for i, s in enumerate(t_str.split(':'))])
        t.append(seconds)
    t = np.array(t) - t[0]
    dump_to_cache(t, 'time_' + k)
    return t


""" 
    SHAPE EVOLUTION
"""


def compute_volume(contour, ccal):
    height = 16.5  # fix height (measured from bottom) of cylinder to 16.5 cm, ignore anything above this height

    X, Y = list(zip(*contour))
    X = np.array(X)
    Y = np.array(Y)
    V = 0
    for y in np.arange(np.max(Y)-int(height * ccal), np.max(Y)):
        if np.any(Y == y):
            diam = np.max(X[Y==y]) - np.min(X[Y==y])
            V += np.pi * diam**2 / 4
    return V / (100 * ccal)**3  # volume in m^3


def compute_surface_area(contour, ccal):
    height = 16.5  # fix height (measured from bottom) of cylinder to 16.5 cm, ignore anything above this height

    X, Y = list(zip(*contour))
    X = np.array(X)
    Y = np.array(Y)
    A = 0
    for y in np.arange(np.max(Y)-int(height*ccal), np.max(Y)):
        if np.any(Y == y):
            diam = np.max(X[Y==y]) - np.min(X[Y==y])
            A += np.pi * diam
    return A / (100 * ccal)**2  # surface area in m^2


def compute_average_radius(contour):
    X, Y = list(zip(*contour))

    # filter for upper 80%
    y80 = np.sort(Y)[int(len(Y) * 0.8)]
    X = [x for x, y in contour if y < y80]

    r = np.mean(np.abs(np.array(X) - np.mean(X)))
    return r  # radius in pixels


def compute_Nusselt_number(k):
    """ Compute the Nusselt number from a linear fit between sqrt(volume) and time """
    L = 334 * 1e3  # latent heat of fusion [J/kg]
    H = 0.3  # height of cylinder [m]
    c_ice = 2090  # specific heat capacity of ice [J/kg/K]
    T0_ice = 0  # temperature of ice core [degrees C]
    T = SETTINGS[k]["T"]  # ambient temperature [degrees C]
    S = SETTINGS[k]["SP"]  # ambient salinity [g/kg]
    rho_ice = Seawater(0, 0).density()
    sw = Seawater(T, S)

    toff = find_time_offset(k)
    # vol = np.array([compute_volume(c, ccal=SETTINGS[k]["Ccal"]) for c in get_contours(k)][toff:])
    vol = np.array(get_from_cache('volume_'+k))[toff:]
    t = find_times(k)[toff:toff + len(vol)]
    t = t - t[0]
    tau = t / find_half_volume_time(k)
    alpha = sw.thermal_conductivity() * T / (rho_ice * (c_ice * T0_ice - L) * H)

    slope, intercept, r, p, se = linregress(t[tau < 2], np.sqrt(vol[tau < 2]))
    Nu = slope / (alpha * np.sqrt(np.pi * H))
    Nu_se = np.abs(se / (alpha * np.sqrt(np.pi * H)))   # standard error
    return Nu, Nu_se


def compute_RayleighT_number(k):
    H = 0.3  # height of cylinder [m]
    g = 9.81  # [m^2/s]
    T = SETTINGS[k]["T"]  # ambient temperature [degrees C]
    S = SETTINGS[k]["SP"]  # ambient salinity [g/kg]
    sw = Seawater(T, S)
    RaT = -g * sw.density_derivative_t() * T * H**3 / (sw.dynamic_viscosity() * sw.thermal_diffusivity())
    return RaT


def compute_RayleighS_number(k):
    H = 0.3  # height of cylinder [m]
    g = 9.81  # [m^2/s]
    Ds = 1e-9  # haline diffusivity [m^2/s]
    T = SETTINGS[k]["T"]  # ambient temperature [degrees C]
    S = SETTINGS[k]["SP"]  # ambient salinity [g/kg]
    sw = Seawater(T, S)
    RaS = g * sw.density_derivative_s() * S * H ** 3 / (sw.dynamic_viscosity() * Ds)
    return RaS


def compute_GrashofT_number(k):
    H = 0.3  # height of cylinder [m]
    g = 9.81  # [m^2/s]
    T = SETTINGS[k]["T"]  # ambient temperature [degrees C]
    S = SETTINGS[k]["SP"]  # ambient salinity [g/kg]
    sw = Seawater(T, S)
    GrT = -g * sw.density_derivative_t() * sw.density() * T * H ** 3 / (sw.dynamic_viscosity()**2)
    return GrT


def compute_GrashofS_number(k):
    H = 0.3  # height of cylinder [m]
    g = 9.81  # [m^2/s]
    T = SETTINGS[k]["T"]  # ambient temperature [degrees C]
    S = SETTINGS[k]["SP"]  # ambient salinity [g/kg]
    sw = Seawater(T, S)
    GrS = g * sw.density_derivative_s() * sw.density() * S * H ** 3 / (sw.dynamic_viscosity() ** 2)
    return GrS


""""
    SCALLOPS
"""


def find_local_extremes(contour):
    X, Y = list(zip(*contour))
    c = bin_contour(contour, n=31)
    c = smoothen_contour(c, n=5)

    # replace two points with equal x-positions with one average point
    remove = []
    for i in range(1, len(c)):
        if c[i][0] == c[i-1][0]:
            c[i-1] = (c[i-1][0], (c[i-1][1]+c[i][1])/2)
            remove.append(i)
    c = [c[i] for i in range(len(c)) if i not in remove]

    Xc, Yc = list(zip(*c))
    Xac = np.abs(np.array(Xc) - np.mean(X))     # distance from vertical symmetry axis

    # find local minima/maxima
    extremes = []
    for i in range(1, len(Xac)-1):
        j = int(i * len(X)/len(Xac))  # index in original contour, before binning/smoothing
        if Xac[i-1] < Xac[i] > Xac[i+1]:  # crest
            extremes.append({"type": "high", "x": X[j], "y": Y[j], "idx": j})
        elif Xac[i-1] > Xac[i] < Xac[i+1] and Xac[i] > np.mean(Xac)/2:  # trough
            extremes.append({"type": "low", "x": X[j], "y": Y[j], "idx": j})
    return extremes


def filter_local_extremes(extremes):
    """ Only keep sets of peaks that have a significant dip in between them """

    amp_threshold = 5  # pixels
    filtered_points = []
    for i in range(1, len(extremes)-1):
        if extremes[i]["type"] == "low":
            cond1 = (extremes[i-1]["type"] == "high") and np.abs(extremes[i]["x"] - extremes[i-1]["x"]) > amp_threshold
            cond2 = (extremes[i + 1]["type"] == "high") and np.abs(extremes[i]["x"] - extremes[i+1]["x"]) > amp_threshold
            if cond1 and cond2:
                filtered_points.append([extremes[i-1], extremes[i+1]])
    return filtered_points


def compute_wavelengths(contour):
    local_extremes = find_local_extremes(contour)
    peak_pairs = filter_local_extremes(local_extremes)
    wavelengths = [np.abs(p1["y"] - p2["y"]) for p1, p2 in peak_pairs]
    return wavelengths  # in pixels


def compute_scallop_speed(k, from_cache=DEFAULT_CACHE):
    """ Computes downward speed of the scallops relative to average melt rate
        NOTE: a fit between dr/dt and dr/dy might be a more reliable method to obtain the scallop speed
     """

    if from_cache:
        return get_from_cache('scallop_speed_'+k)

    # Algorithm parameters
    outlier_dist = 0.06 if k=='f2' else 0.03 if k=='e3' else 0.04  # radius in which point density must be high
    outlier_density = 3  # minimum number of points inside circle with radius <outlier_dist>
    max_connection_dist = 0.05 if k=='f2' else 0.03  # maximum connection length
    min_path_length = .3  # minimum number of points in a path, fraction of total number of points

    cntrs = get_contours(k)
    settings = SETTINGS[k]

    # Find local maxima of both sides and save them separately
    t1, t2, y1, y2 = [], [], [], []
    x1, x2 = [], []
    for i, c in enumerate(cntrs):
        extr = find_local_extremes(c)
        mid_x = np.mean([p[0] for p in c])
        for p in extr:
            if p['type'] == 'high':
                if p["x"] < mid_x:
                    x1.append(p["x"] / settings["Ccal"])  # vertical position in cm
                    y1.append(p["y"] / settings["Ccal"])  # vertical position in cm
                    t1.append(i * settings["dt"])  # time in s
                else:
                    x2.append(p["x"] / settings["Ccal"])  # vertical position in cm
                    y2.append(p["y"] / settings["Ccal"])  # vertical position in cm
                    t2.append(i * settings["dt"])  # time in s

    speeds, path_lengths = [[], []], [[], []]
    for n, t, x, y in [(0, t1, x1, y1), (1, t2, x2, y2)]:
        # normalize height and time series
        yn = np.array(y) / np.max(y)
        tn = np.array(t) / np.max(t)

        # remove outliers
        keep = []
        for i in range(len(y)):
            dist = np.sqrt((yn - yn[i]) ** 2 + (tn - tn[i]) ** 2)
            if np.sum(dist < outlier_dist) > outlier_density:
                keep.append(i)

        # connect the dots
        connections = []
        yn = np.array([yn[i] for i in keep])
        tn = np.array([tn[i] for i in keep])
        for j in range(yn.size):
            dist = np.sqrt((yn - yn[j]) ** 2 + (tn - tn[j]) ** 2)
            dist[tn <= tn[j]] = 999  # next point must be later in time
            if np.min(dist) < max_connection_dist:
                connections.append([keep[j], keep[np.argmin(dist)]])

        # connections to paths
        paths = []
        while len(connections) > 0:
            c = connections[0]
            for i in range(len(paths)):
                if c[0] == paths[i][-1]:
                    paths[i].append(c[1])
                    break
            else:
                paths.append(c)
            connections.pop(0)

        # filter paths
        paths = [p for p in paths if len(p) > int(min_path_length * len(cntrs))]

        # apply linear fit to paths
        fits = []
        for p in paths:
            fit = np.polyfit([t[i] for i in p], [y[i] for i in p], deg=1)
            fits.append(fit)
        if k == 'b2' and n == 1:
            paths = [paths[i] for i in range(len(paths)) if fits[i][0] > 0]
            fits = [f for f in fits if f[0] > 0]
        speeds[n] = [f[0] * 60 * 10 for f in fits]  # cm/s -> mm/min.
        path_lengths[n] = [len(p) for p in paths]

    if len(speeds) == 0:
        return np.nan, np.nan

    mean_speed = np.average(speeds[0] + speeds[1], weights=path_lengths[0]+path_lengths[1])  # mm/min.
    std_speed = np.sqrt(np.average((np.array(speeds[0] + speeds[1]) - mean_speed)**2, weights=path_lengths[0]+path_lengths[1]))  # mm/min.

    avg_r = [compute_average_radius(c) for c in get_contours(k)]
    t = find_times(k)
    slope, intercept, r, p, se = linregress(t, avg_r)
    drdt = -slope / SETTINGS[k]["Ccal"] * 10 * 60  # mm/min.
    drdt_std = se / SETTINGS[k]["Ccal"] * 10 * 60  # mm/min.

    std_speed = np.abs(mean_speed / drdt) * np.sqrt((std_speed / mean_speed)**2 + (drdt_std/drdt)**2)
    mean_speed = mean_speed / drdt
    dump_to_cache([mean_speed, std_speed], 'scallop_speed_'+k)
    return mean_speed, std_speed


def compute_roughness_parameters(k, from_cache=DEFAULT_CACHE):
    """ computes histogram of deviation from a quadratic fit for experiment <k>, from tau=1 to tau=3
     returns sigma, kurtosis, skewness, Rk, Rpk, Rvk of this histogram """

    if from_cache:
        return get_from_cache('roughness_parameters_'+k)

    deg = 2  # polynomial degree for fit
    bins = np.arange(-50, 50, .2)
    hvi = find_half_volume_index(k)

    data1, data2 = np.array([]), np.array([])
    for c in get_contours(k)[hvi:]:
        try:
            x, y = list(zip(*c))
            x = np.array(x)
            y = np.array(y)

            # take selection
            y_max = np.max(y)
            x_mean = np.mean(x)
            x = x[y < 0.8 * y_max]
            y = y[y < 0.8 * y_max]

            # separate sides
            x1, y1 = x[x < x_mean], y[x < x_mean]  # left side
            x2, y2 = x[x > x_mean], y[x > x_mean]  # right side

            # perform polynomial fit
            p1 = np.polyfit(y1, x1, deg=deg)
            p2 = np.polyfit(y2, x2, deg=deg)
            fx1 = np.sum([p1[i] * y1 ** (deg - i) for i in range(deg + 1)], axis=0)
            fx2 = np.sum([p2[i] * y2 ** (deg - i) for i in range(deg + 1)], axis=0)

            data1 = np.hstack((data1, x1 - fx1))
            data2 = np.hstack((data2, x2 - fx2))
        except TypeError:
            print("type error at {:s}".format(k))

    params = []
    hx = (bins[1:] + bins[:-1]) / 2 / SETTINGS[k]["Ccal"] * 10

    for d in [data1, data2]:
        h, _ = np.histogram(d, bins=bins)
        sigma = np.sqrt(np.mean(d**2))
        Sk = np.mean(d ** 3) / sigma ** 3
        Ku = np.mean(d ** 4) / sigma ** 4

        sh = np.sum(h)
        b = np.array([np.sum(h[:i]) / sh for i in range(len(h))])  # bearing curve

        # takes the fit with the smallest slope
        fits = []
        for i in range(len(b)):
            if b[i] < b[-1] - 0.4:
                j = np.where(b - b[i] > 0.4)[0][0]  # +40% point
                p = np.polyfit(b[i:j], hx[i:j], deg=1)
                fits.append(p)
        idx = np.argmin([p[0] for p in fits])
        Rk = fits[idx][0]
        Rvk = fits[idx][1] - np.min(d)
        Rpk = np.max(d) - (fits[idx][1] + fits[idx][0])

        # takes the fit from 30% to 70%
        i, j = np.where(b > 0.3)[0][0], np.where(b > 0.7)[0][0]
        p_mid = np.polyfit(b[i:j], hx[i:j], deg=1)
        Rk_mid = p_mid[0]
        Rvk_mid = p_mid[1] - np.min(d)
        Rpk_mid = np.max(d) - (p_mid[1] + p_mid[0])

        params.append({"sigma": sigma, 'Sk': Sk, "Ku": Ku, "Rk": Rk, "Rpk": Rpk, "Rvk": Rvk, "Rk_mid": Rk_mid, "Rpk_mid": Rpk_mid, "Rvk_mid": Rvk_mid})
    dump_to_cache(params, 'roughness_parameters_' + k)
    return params


"""
    PIV
"""


def mask_to_edges(mask):
    """ Extract ice-water interface from a mask computed by tif_to_mask.py """
    mask = (255*mask).astype(np.uint8)

    cont, hierarchy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    edges = np.reshape(cont[0], (cont[0].shape[0], 2))

    # remove sides
    edges = edges[edges[:, 0] != 0]
    edges = edges[edges[:, 0] != 1023]
    edges = edges[edges[:, 1] != 0]
    edges = edges[edges[:, 1] != 1023]

    # invert y-coordinate
    edges[:, 1] = 1023 - edges[:, 1]
    return edges


def dat_to_vmat(filepath, dpx=16, flip_hor=False):
    """ Read .dat file and return velocity matrix.
    <dpx> is the distance in pixels between two velocity data points.
    This function takes roughly 0.2 seconds per frame """

    px_per_cm = 197.7
    fps = 50  # frames per second
    frames_skipped = 7  # number of frames skipped in processing, because of slow flow
    dt = 1. / (fps / frames_skipped)  # time interval between processed velocity fields

    vel = np.zeros((1024 // dpx + 1, 1024 // dpx + 1, 2))
    validity = np.zeros(vel.shape[:2])
    lines = open(filepath, 'r').readlines()
    for ln in lines[3:]:
        x, y, vx, vy, is_valid = ln.split(' ')
        j, i = int((int(x) - dpx / 2) / dpx), int((int(y) - dpx/2) / dpx)
        vel[i, j, :] = [float(vx), float(vy)]
        validity[i, j] = is_valid

    if flip_hor:
        # Exp PIV1 was flipped horizontally
        vel = np.fliplr(vel)
        validity = np.fliplr(validity)
        vel[:, :, 0] *= -1

    vel = vel / (100 * px_per_cm) / dt  # px per frame -> m/s
    vel[:, :, 1] *= -1  # flip y-direction
    return vel, validity


def combine_sets(folder):
    """ Rename all .dat files in <folder> and put them together in a new folder.
    Assumes .dat files are organized in folders named set1--7. """
    import shutil

    for sn in range(1, 8):
        files = glob.glob(folder + r"\set{:d}".format(sn) + r"\*.dat")
        for i, fn in enumerate(files):
            dst = folder + r'\all\B{:05d}.dat'.format(i * 7 + sn)
            shutil.copy(fn, dst)
            print("\r{:d}/{:d}".format(i+1+(sn-1)*len(files), len(files)*7), end='')


def vmat_relative_to_wall(vmat, mask):
    """ Reshape velocity matrix such that first column is velocity at the wall """
    N = int(mask.shape[0] / vmat.shape[0])
    smask = np.sum(mask, axis=1)
    idx = [np.mean(smask[i:i+N]) / N for i in range(0, len(smask), N)]

    mat = np.nan * np.zeros(vmat.shape)
    for i in range(mat.shape[0]):
        j, dj = int(idx[i]), idx[i] - int(idx[i])
        mat[i, :-j, :] = (1 - dj) * vmat[i, j:, :]
        mat[i, :-(j+1), :] += dj * vmat[i, (j+1):, :]
    return mat

#
# def coarsen_mask(mask, n):
#     m = np.zeros((mask.shape[0] // n, mask.shape[1] // n))
#     for i in range(m.shape[0]):
#         for j in range(m.shape[1]):
#             m[i, j] = np.median(mask[i*n:(i+1)*n, j*n:(j+1)*n])
#     return m


def compute_average_mask(k, from_cache=DEFAULT_CACHE):
    if from_cache:
        return get_from_cache('piv_average_mask_{:s}'.format(k))

    trans = {'piv1': '6a', 'piv2': '6b', 'piv3': '6c', 'piv4': '6d'}
    fnames = glob.glob(PIV_FOLDER + r'\masks\masks_{:s}\*.pkl'.format(trans[k]))
    if k == 'piv4':
        fnames = fnames[:1250]

    adv = piv_advection(k)
    melt = piv_melt_rate(k)
    avg_profile = np.zeros(1024)
    cnt = np.zeros(1024)
    for i, fn in enumerate(fnames):
        with open(fn, 'rb') as f:
            mask = pickle.load(f)
        edges = mask_to_edges(mask)
        for j in range(len(edges)):
            bin_num = int(edges[j, 1] - i * 10 * adv)
            if bin_num < len(avg_profile):
                avg_profile[bin_num] += edges[j, 0] - i * 10 * melt
                cnt[bin_num] += 1
    prof = np.array([[avg_profile[i] / cnt[i], i] for i in range(len(avg_profile))])
    dump_to_cache(prof, 'piv_average_mask_{:s}'.format(k))
    return prof


def piv_melt_rate(k, from_cache=DEFAULT_CACHE):
    if from_cache:
        return get_from_cache('piv_meltrate_'+k)

    trans = {'piv1': '6a', 'piv2': '6b', 'piv3': '6c', 'piv4': '6d'}
    fnames = glob.glob(r'E:\Thesis\exp6_piv\masks\masks_{:s}\*.pkl'.format(trans[k]))
    if k == 'piv4':
        fnames = fnames[:1250]

    mx = np.zeros(len(fnames))
    for i in range(len(fnames)):
        with open(fnames[i], 'rb') as f:
            mask = pickle.load(f)
        edges = mask_to_edges(mask)
        mx[i] = np.mean(edges[:, 0])
    melt_rate = 0.1 * np.polyfit(np.arange(len(mx)), mx, deg=1)[0]  # horizontal melt rate in px per frame
    print("[{:s}] melt-rate: {:.3f} mm/min.".format(k, melt_rate / load_settings(k)['Ccal'] * 50 * 60 * 10))
    dump_to_cache(melt_rate, 'piv_meltrate_'+k)
    return melt_rate


def piv_advection(k, from_cache=DEFAULT_CACHE):
    if from_cache:
        return get_from_cache('piv_advection_'+k)

    approx_y = {'piv1': [], 'piv2': [800], 'piv3': [400, 900], 'piv4': [320, 620]}
    trans = {'piv1': '6a', 'piv2': '6b', 'piv3': '6c', 'piv4': '6d'}
    fnames = glob.glob(r'E:\Thesis\exp6_piv\masks\masks_{:s}\*.pkl'.format(trans[k]))
    if k == 'piv4':
        fnames = fnames[:1250]

    crests = [[] for _ in approx_y[k]]
    for i, fn in enumerate(fnames):
        with open(fn, 'rb') as f:
            mask = pickle.load(f)
        edges = mask_to_edges(mask)
        for j in range(len(approx_y[k])):
            idx_max = approx_y[k][j]-100 + np.argmax(edges[approx_y[k][j]-100:approx_y[k][j]+100, 0])
            p = np.polyfit(edges[idx_max-50:idx_max+50, 1], edges[idx_max-50:idx_max+50, 0], deg=2)
            idx_max = int(-p[1] / (2*p[0]))
            crests[j].append(edges[idx_max])
    advection = 0.1 * np.mean([np.polyfit(np.arange(len(c)), [p[1] for p in c], deg=1)[0] for c in crests])  # vertical advection in px per frame
    if k == 'piv1':
        advection = 0
    print("[{:s}] advection: {:.3f} mm/min.".format(k, advection / load_settings(k)['Ccal'] * 50 * 60 * 10))
    dump_to_cache(advection, 'piv_advection_'+k)
    return advection


def compute_mean_velocity_field(k, n=500, from_cache=DEFAULT_CACHE, start_idx=0):
    """ Computes velocity field of experiment <k>, averaged over the first (<n> frames) """
    if from_cache:
        return get_from_cache('mean_vel_{:s}_n{:d}_si{:d}'.format(k, n, start_idx))
    adv = piv_advection(k)
    melt = piv_melt_rate(k)
    trans = {'piv1': '6a', 'piv2': '6b', 'piv3': '6c', 'piv4': '6d'}
    fname = PIV_FOLDER + r'\exp{:s}\DAT\all\B{:05d}.dat'
    velocity, validity = dat_to_vmat(fname.format(trans[k], start_idx+1))
    for i in range(2 + start_idx, n + start_idx):
        vm, val = dat_to_vmat(fname.format(trans[k], i), flip_hor=k=='piv1')

        # Shift to match frame 500
        jx, jy = int(abs((i-500) * melt / 16)), int(abs((i-500) * adv / 16))
        jx_end = vm.shape[0] if jx == 0 else -jx
        jy_end = vm.shape[1] if jy == 0 else -jy
        if i > 500:
            velocity[:jy_end, jx:] += vm[jy:, :jx_end]
            validity[:jy_end, jx:] += val[jy:, :jx_end]
        else:
            velocity[jy:, :jx_end] += vm[:jy_end, jx:]
            validity[jy:, :jx_end] += val[:jy_end, jx:]
        print("\r[mean vel field]: {:d}/{:d}".format(i - start_idx, n + start_idx), end='')
    velocity[:, :, 0] = velocity[:, :, 0] / validity
    velocity[:, :, 1] = velocity[:, :, 1] / validity
    dump_to_cache(velocity, 'mean_vel_{:s}_n{:d}_si{:d}'.format(k, n, start_idx))
    return velocity


def compute_rms_velocity_field(k, n=500, from_cache=DEFAULT_CACHE, start_idx=0):
    """ Computes RMS velocity field of experiment <k>, averaged over the first (<n> frames) """
    if from_cache:
        return get_from_cache('rms_vel_{:s}_n{:d}_si{:d}'.format(k, n, start_idx))
    adv = piv_advection(k)
    melt = piv_melt_rate(k)
    mean_vel = compute_mean_velocity_field(k, n=n, start_idx=start_idx)
    mean_speed = np.sqrt(mean_vel[:, :, 0]**2 + mean_vel[:, :, 1]**2)
    mean_speed[np.isnan(mean_speed)] = 0
    fname = PIV_FOLDER + r'\exp{:s}\DAT\all\B{:05d}.dat'
    trans = {'piv1': '6a', 'piv2': '6b', 'piv3': '6c', 'piv4': '6d'}
    velocity, validity = dat_to_vmat(fname.format(trans[k], start_idx + 1), flip_hor=k=='piv1')
    rms = (np.sqrt(velocity[:, :, 0]**2 + velocity[:, :, 1]**2) - mean_speed) ** 2
    for i in range(2 + start_idx, n + start_idx):
        vm, val = dat_to_vmat(fname.format(trans[k], i), flip_hor=k=='piv1')
        sm = np.sqrt(vm[:, :, 0]**2 + vm[:, :, 1]**2)

        # Shift to match frame 500
        jx, jy = int(abs((i-500) * melt / 16)), int(abs((i-500) * adv / 16))
        jx_end = sm.shape[0] if jx == 0 else -jx
        jy_end = sm.shape[1] if jy == 0 else -jy
        if i > 500:
            rms[:jy_end, jx:] += (sm[jy:, :jx_end] - mean_speed[jy:, :jx_end]) ** 2
            validity[:jy_end, jx:] += val[jy:, :jx_end]
        else:
            rms[jy:, :jx_end] += (sm[:jy_end, jx:] - mean_speed[:jy_end, jx:]) ** 2
            validity[jy:, :jx_end] += val[:jy_end, jx:]
        print("\r[rms vel field]: {:d}/{:d}".format(i - start_idx, n + start_idx), end='')
    rms = np.sqrt(rms / validity)
    dump_to_cache(rms, 'rms_vel_{:s}_n{:d}_si{:d}'.format(k, n, start_idx))
    return rms


def histogram_at_point(k, rows, dwall, n_frames=990, from_cache=DEFAULT_CACHE):
    if from_cache:
        return get_from_cache('histogram_{:s}_n{:d}'.format(k, n_frames))

    bin_edges = [np.linspace(-.3, .3, 60), np.linspace(-0.2, 2, 60)]  # bins for x, and y-velocity in mm/min.
    dpx = 16  # distance between values in vel field in pixels
    dy, dx = 1, 0  # number of points on either side to include in the average
    w_sz = (2*dx + 1) * (2*dy + 1)  # window size

    adv_rate = piv_advection(k)
    melt_rate = piv_melt_rate(k)
    edges = compute_average_mask(k)
    edges[:, 0] = edges[::-1, 0]

    data = np.zeros((len(rows), len(dwall), n_frames * w_sz, 2))
    xwall = [np.mean([ex for ex, ey in edges if r - dy < ey / dpx < r + dy + 1]) / dpx for r in rows]
    fname = PIV_FOLDER + r'\exp{:s}\DAT\all\B{:05d}.dat'
    trans = {'piv1': '6a', 'piv2': '6b', 'piv3': '6c', 'piv4': '6d'}
    for n in range(n_frames):
        vmat, val = dat_to_vmat(fname.format(trans[k], n + 1), flip_hor=k=='piv1')
        vmat[val == 0] = np.nan
        for i, r, c in zip(range(len(rows)), rows, xwall):
            r -= int(adv_rate / dpx * (n - 500))  # shift the velocity field to compensate for advection
            c += melt_rate / dpx * (n - 500)  # shift the velocity field to compensate for melt
            c1, c2 = int(c-0.5), int(c-0.5) + 1
            for j, dw in enumerate(dwall):
                v1 = vmat[r - dy:r + dy + 1, c1 - dx + dw:c1 + dx + 1 + dw, :]
                v2 = vmat[r - dy:r + dy + 1, c2 - dx + dw:c2 + dx + 1 + dw, :]
                v = v1 * (c - 0.5 - c1) - v2 * (c - 0.5 - c2)
                v[:, :, 1] *= -1  # define positive y-velocity as downward
                if v.size == w_sz * 2:
                    data[i, j, w_sz * n:w_sz * (n + 1), :] = np.reshape(v, (1, 1, w_sz, 2))
                else:
                    data[i, j, w_sz * n:w_sz * (n + 1), :] = np.nan
        print("\r[hist {:s}] {:d}/{:d}".format(k, n + 1, n_frames), end='')

    hist = [[[[] for _ in dwall] for _ in rows] for _ in 'xy']
    hist_x = [[[[] for _ in dwall] for _ in rows] for _ in 'xy']
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            for a in range(data.shape[3]):
                dat = data[i, j, :, a]
                dat = dat[~np.isnan(dat)] * 100  # m/s -> cm/s
                h, be = np.histogram(dat, bins=bin_edges[a])

                h = h / np.sum(h) / (be[1] - be[0])
                hist[a][i][j] = h
                hist_x[a][i][j] = (be[1:] + be[:-1]) / 2
    dump_to_cache([hist, hist_x], 'histogram_{:s}_n{:d}'.format(k, n_frames))
    return hist, hist_x


def time_series_at_point(k, row, dwall, n_frames=990, from_cache=DEFAULT_CACHE):
    if from_cache:
        return get_from_cache('velocity_time_series_'+k)

    dpx = 16
    data = np.nan * np.zeros((n_frames, 2))
    adv_rate = piv_advection(k)
    melt_rate = piv_melt_rate(k)
    edges = compute_average_mask(k)
    edges[:, 0] = edges[::-1, 0]
    xwall = edges[np.argmin(np.abs(edges[:, 1] / dpx - row)), 0] / dpx
    fname = PIV_FOLDER + r'\exp{:s}\DAT\all\B{:05d}.dat'
    trans = {'piv1': '6a', 'piv2': '6b', 'piv3': '6c', 'piv4': '6d'}
    for n in range(n_frames):
        vmat, val = dat_to_vmat(fname.format(trans[k], n + 1), flip_hor=k=='piv1')
        vmat[val == 0] = np.nan
        r = row - int(adv_rate / dpx * (n - 500))  # shift the velocity field to compensate for advection
        c = xwall + melt_rate / dpx * (n - 500)  # shift the velocity field to compensate for melt
        c1, c2 = int(c-0.5), int(c-0.5) + 1
        data[n, :] = vmat[r, c1 + dwall, :]
        print("\r[time series {:s}] {:d}/{:d}".format(k, n + 1, n_frames), end='')
    dump_to_cache(data, 'velocity_time_series_'+k)
    return data
