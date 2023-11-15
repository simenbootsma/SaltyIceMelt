import cv2 as cv
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyArrow
import numpy as np
import cmcrameri.cm as cmc
import analysis
from load_settings import load_settings, ALL_KEYS
from extract_contours import rotate_image, find_edges
from Seawater import Seawater
import pyperclip


SETTINGS = load_settings('all')


def show_figures():
    # refresh_cache()
    tab_3()   # experiment details
    fig_3()   # density in T-S diagram
    fig_5()   # boundary tracking processing
    fig_7()   # contours
    fig_8()   # volume over time
    fig_9()   # Nusselt - Rayleigh
    fig_10()  # both mean and RMS fields
    fig_11()   # velocity time series at points
    fig_12()   # velocity profiles at lines
    fig_13()   # location of lines for profiles and points for histograms
    fig_14_15()   # velocity histograms at points
    fig_16()   # scallop amplitude in Grashof plot
    fig_17()   # radius and r - <r> as function of height and time
    fig_18()  # wavelength and scallop migration as function of density ratio
    fig_20()  # all contours at 17 degrees
    fig_21_22_23()  # all contours at high temperature


def tab_3():
    """ Generate LaTeX table with all experiments """
    header = r"Exp. & $T_\infty$ & $S_\infty$ & $t_{1/2}$ & $R_\rho$ & $Gr_T$ & $Gr_S$ & Nu\\"
    units = r" & ($\degree$C) & (g/kg)  & (min.) & & & & \\"
    row_f = r"{:s}$_{:s}$ & {:.1f} & {:.1f} & {:.1f} & {:.3f} & ${:.1f} \times 10^{:d}$ & ${:.1f} \times 10^{:d}$ & {:.0f}\\"
    row_f2 = r"{:s}$_{:s}$ & {:.1f} & {:.1f} & {:.1f} & $\infty$ & ${:.1f} \times 10^{:d}$ & {:.1f} & {:.0f}\\"
    rows = {k: [] for k in 'abcdef'}

    # Gather data
    for k in ALL_KEYS:
        exp_let, exp_num = k[0], k[1]
        T, S = SETTINGS[k]['T'], SETTINGS[k]['SP']
        th = analysis.find_half_volume_time(k) / 60
        R = Seawater(T, S).density_ratio()
        GrT = analysis.compute_GrashofT_number(k)
        GrT_val, GrT_exp = GrT / 10**int(np.log10(GrT)), int(np.log10(GrT))
        Nu = analysis.compute_Nusselt_number(k)[0]
        GrS = analysis.compute_GrashofS_number(k)
        if GrS > 0:
            GrS_val, GrS_exp = GrS / 10 ** int(np.log10(GrS)), int(np.log10(GrS))
            row = row_f.format(exp_let, exp_num, T, S, th, R, GrT_val, GrT_exp, GrS_val, GrS_exp, Nu)
        else:
            row = row_f2.format(exp_let, exp_num, T, S, th, GrT_val, GrT_exp, GrS, Nu)
        rows[exp_let].append(row)

    # Put it all together and copy to clipboard
    txt = 2*"\hline\n" + header + "\n" + units + "\n"
    for k in 'abcdef':
        txt += "\hline\n"
        for r in rows[k]:
            txt += r + "\n"
    txt += 2 * "\hline\n"
    pyperclip.copy(txt)
    print("Table is copied to clipboard!")


def fig_3():
    """ Show density as function of T and S """
    CMAP = ListedColormap([cmc.devon_r(val) for val in np.linspace(0, .8, 250)])
    contour_levels = [990, 1000, 1010, 1020, 1030, 1040, 1050, 1060, 1070, 1080]
    MANUAL_CLABELS = [(x, -0.5 * x + 57.5) for x in np.linspace(5, 102, len(contour_levels))]

    min_t, min_sal = -10, 0
    max_t, max_sal = 60.1, 110.1
    t, sal = np.meshgrid(np.arange(min_t, max_t, .05), np.arange(min_sal, max_sal, .05))
    sw = Seawater(t, sal)
    rho = sw.density()
    # t_arr, sal_arr = np.arange(np.min(sw.t), np.max(sw.t), .01), np.arange(np.min(sw.s), np.max(sw.s), .01)

    t_fr = sw.freezing_temperature()

    plt.figure()
    plt.imshow(np.flipud(rho.T), extent=[np.min(sw.s), np.max(sw.s), np.min(sw.t), np.max(sw.t)], cmap=CMAP, aspect='auto', vmin=990, vmax=1080)
    plt.xlabel('Salinity (g/kg)', fontsize=12)
    plt.ylabel('Temperature ($\degree$C)', fontsize=12)
    plt.tick_params(labelsize=12)
    cb = plt.colorbar(extend='both')
    cb.ax.tick_params(labelsize=12)
    cb.ax.set_title(r"$\rho$ (kg/m$^3$)", fontsize=12)

    t1 = plt.Polygon(np.array([[np.min(sw.s), np.min(sw.t)], [np.min(sw.s), 0], [np.max(sw.s), np.min(t_fr)], [np.max(sw.s), np.min(sw.t)]]), color=[.6, .6, .6])
    plt.gca().add_patch(t1)

    crho = rho + 0.0
    crho[sw.t < t_fr] = np.nan

    plt.plot([np.min(sw.s), np.max(sw.s)], [0, 0], '--k', lw=1)
    cont = plt.contour(np.flipud(sw.s.T), np.flipud(sw.t.T), np.flipud(crho.T), levels=contour_levels, colors='black', linewidths=[1, 2] + [1 for _ in range(len(contour_levels)-2)])
    plt.clabel(cont, cont.levels, manual=MANUAL_CLABELS, fmt=lambda x: "{:.0f}".format(x), fontsize=10)

    # maximum density
    s_col = sw.s[:, 0]
    s_col = s_col[s_col < 26.25]
    plt.plot(s_col[s_col < 7.6], 4 - 0.216 * s_col[s_col < 7.6], color=(.3, .3, .3), lw=1.5)
    plt.plot(s_col[s_col > 10], 4 - 0.216 * s_col[s_col > 10], color=(.3, .3, .3), lw=1.5)
    plt.text(7.2, 1.3, 'T$*$', color=(.3, .3, .3), fontsize=11)

    # freezing temperature
    plt.plot(sw.s[:, 0], t_fr[:, 0], '-', color=(.2, .2, .2), lw=2)
    plt.text(30, -5, 'T$_{fp}$', color=(.2, .2, .2), fontsize=11)
    plt.text(5, -7, 'Ice', color='k', fontsize=12)

    red = plt.get_cmap('Reds_r', 10)(3)
    blue = plt.get_cmap('Blues_r', 10)(3)
    purple = plt.get_cmap('Purples_r', 10)(3)
    green = plt.get_cmap('Greens_r', 10)(5)

    colors = {"a": red, "c": blue, 'b': purple, 'd': green, 'e': green, 'f': green}
    for k in ALL_KEYS:
        plt.plot(SETTINGS[k]["SP"], SETTINGS[k]["T"], 'o', mec='k', mfc=colors[k[0]], markersize=6 if k[0] in 'abc' else 5, mew=1, clip_on=False, zorder=100)
    plt.show()


def fig_5():
    """ Example of processing for boundary tracking """
    filepath = r"E:\Thesis\exp3\experiment3e\JPG\DSC_4828.jpg"
    settings = SETTINGS["a5"]

    img = cv.imread(filepath)
    img = rotate_image(img, angle=settings["rot"])

    gray = cv.cvtColor(img, cv.COLOR_RGBA2GRAY)

    HOR_SLICE = slice(settings["hcrop"][0], settings["hcrop"][1])
    VERT_SLICE = slice(settings["vcrop"][0], settings["vcrop"][1])
    crop = gray[VERT_SLICE, HOR_SLICE]

    ret, glob_bin = cv.threshold(crop, settings["binthresh"], 255, cv.THRESH_BINARY)
    inverted = 255 - glob_bin
    kernel = cv.getStructuringElement(cv.MORPH_RECT, ksize=(5, 5))
    closed = cv.morphologyEx(inverted, cv.MORPH_CLOSE, kernel, iterations=settings["CL_ITER"])
    edges = find_edges(closed)

    binary = np.zeros((glob_bin.shape[0], glob_bin.shape[1], 3))
    binary[:, :, 0] = inverted
    binary[:, :, 1] = inverted
    binary[:, :, 2] = inverted

    cntr = np.zeros((closed.shape[0], closed.shape[1], 3))
    cntr[:, :, 0] = closed
    cntr[:, :, 1] = closed
    cntr[:, :, 2] = closed

    sz = 10
    for x, y in edges:
        cntr[max(0, int(y)-sz):int(y)+sz+1, int(x)-sz:int(x)+sz+1, :] = [255, 0, 0]

    plt.figure()
    plt.imshow(img[VERT_SLICE, HOR_SLICE])
    plt.title('raw')

    plt.figure()
    plt.imshow(binary/255)
    plt.title('binary')

    plt.figure()
    plt.imshow(cntr/255)
    plt.title('contoured')
    plt.show()


def fig_7():
    """ Contours over time """
    for exp_id in 'abc':
        cmap = {"a": "Reds_r", "c": "Blues_r", "b": "Purples_r"}[exp_id]
        temp = {"a": "17", "c": "5", "b": "11"}[exp_id]
        exps = {"a": ['a' + k for k in '13579'], "c": ['c' + k for k in '12345'], "b": ['b' + k for k in '12345']}[exp_id]
        height = {"a": 18, "c": 17.5, "b": 16.5}[exp_id]  # fixed height from bottom of cylinder

        intervals = np.arange(0, 3, .5)
        colors = [plt.get_cmap(cmap, len(intervals) + 1)(i) for i in range(len(intervals))]

        fig, ax = plt.subplots(ncols=len(exps), figsize=[8, 5])
        plt.subplots_adjust(left=0.18, right=0.902, wspace=0.04)
        # plt.subplots_adjust(left=0.02, right=0.99, wspace=0.04)
        for j, k in enumerate(exps):
            exp = load_settings(k)
            i_half = analysis.find_half_volume_index(k)
            i_off = analysis.find_time_offset(k)
            idx = [i_off + int((i_half - i_off) * intv) for intv in intervals]
            contours = analysis.get_contours(k)
            cx = np.mean(contours[i_off][:, 0])  # center x-location in pixels
            by = np.max(contours[i_off][:, 1]) / exp["Ccal"]  # bottom y-location in cm
            print(k + ": " + ", ".join([str(i) for i in idx]) + " | max: " + str(len(contours)))

            # exceptions
            match k:
                case "a1":
                    idx[4] -= 1
                case "a5":
                    idx[5] -= 6
                case "c3":
                    idx[4] -= 38
                case "c4":
                    idx[4] -= 32
                case "c5":
                    idx[5] -= 9
                case "b1":
                    idx[4] -= 14

            for i in range(len(intervals)):
                if idx[i] < len(contours):
                    if k == "c2":
                        cx = np.mean(contours[idx[i]][:, 0])  # 4b drifted too much
                    x = np.array(analysis.smoothen_array(contours[idx[i]][:, 0]) - cx) / exp["Ccal"]
                    y = np.array(analysis.smoothen_array(contours[idx[i]][:, 1])) / exp["Ccal"]
                    x = x[y > by-height]
                    y = y[y > by-height] - (by - height)
                    xy = np.array([x, y]).T
                    ax[j].add_artist(plt.Polygon(xy, closed=True, ec='k', fc=colors[i], lw=0.5))
            ax[j].set_xlim([-3, 3])
            ax[j].set_ylim([0, 20])
            ax[j].set_aspect('equal')
            ax[j].invert_yaxis()
            ax[j].tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
            if exp_id == 'a':
                ax[j].set_title("{:.0f} g/kg".format(exp["SP"]))
            panel_label = r"({:s}$_{:s}$)".format(k[0], k[1])
            ax[j].text(2.5, 19.5, panel_label, ha='right', va='bottom', fontsize=14)
        if exp_id == 'a':
            ax[0].set_title("$S_\infty = 0$ g/kg")
        ax[0].add_artist(plt.Rectangle((-2.5, 19.5), 2, 0.2, ec='k', fc='k'))
        ax[0].text(-1.5, 19.2, '2 cm', ha='center', fontsize=14)
        ax[0].set_ylabel(r'$T_\infty = {:s}\degree$C'.format(temp), fontsize=14)
        cb = plt.colorbar(plt.cm.ScalarMappable(plt.Normalize(vmin=0, vmax=3),
                                                cmap=ListedColormap(colors)), ax=ax, fraction=0.1, shrink=1, extend='max')
        cb.ax.tick_params(labelsize=14)
        if exp_id == 'a':
            cb.ax.set_title(r'$\tau$ (-)', fontsize=14)
    plt.show()


def fig_8():
    """ Volume over time """
    cmap = cmc.roma_r
    fontsize = 14

    end_time = {"a": 30, "b": 50, "c": 250}
    fig, axes = plt.subplots(ncols=3)
    titles = {"a": "$T_\infty = 17\degree$C", "c": "$T_\infty = 5 \degree$C", "b": "$T_\infty = 11 \degree$C"}
    for a, n in enumerate("cba"):
        keys = [k for k in ALL_KEYS if k[0] == n]
        for i, k in enumerate(keys):
            toff = analysis.find_time_offset(k)
            end_ind = 200 if k == "b4" else SETTINGS[k]["N"]
            V = analysis.get_from_cache('volume_' + k)
            # V = [analysis.compute_volume(c, ccal=SETTINGS[k]['Ccal']) for c in analysis.get_contours(k)]
            V = np.array(V[toff:end_ind])
            normV = V / V[0]  # normalize
            t = analysis.find_times(k)[toff:end_ind] / 60  # time in minutes
            t = t - t[0]

            if k == 'c2':
                normV[964-toff] = np.nan  # failed contour
            axes[a].plot(t, normV, '-', label="{:.0f} g/kg".format(SETTINGS[k]["SP"]), color=cmap(i * cmap.N // len(keys)), zorder=20-i, markersize=2)
        axes[a].set_xlabel('Time (min.)', fontsize=fontsize)
        axes[a].grid()
        axes[a].set_xlim([0, end_time[n]])
        axes[a].set_ylim([0, 1])
        axes[a].set_title(titles[n], fontsize=fontsize)
        axes[a].tick_params(labelleft=a==0, right=a<2, labelsize=fontsize)
        axes[a].set_yticks([0, .5, 1])
        axes[a].text(.9, .9, '({:s})'.format('abc'[a]), fontsize=fontsize+2, transform=axes[a].transAxes, ha='center', va='center')

    axes[-1].legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=fontsize)
    axes[0].set_ylabel(r'$V(t)/V_0$', fontsize=fontsize)
    plt.show()


def fig_9():
    """ Nusselt-Rayleigh plots """
    fontsize = 14
    keysA = ["a" + n for n in "123456789"]
    keysB = ["b" + n for n in "12345"]
    keysC = ["c" + n for n in "12345"]
    all_keys = keysA + keysB + keysC

    Nusselt = {k: analysis.compute_Nusselt_number(k) for k in all_keys}
    RayleighT = {k: analysis.compute_RayleighT_number(k) for k in all_keys}
    RayleighS = {k: analysis.compute_RayleighS_number(k) for k in all_keys}

    red = plt.get_cmap("Reds_r", 10)(3)
    blue = plt.get_cmap("Blues_r", 10)(3)
    purple = plt.get_cmap("Purples_r", 10)(3)

    # Rayleigh S
    fig, axes = plt.subplots(ncols=2)
    colors = [blue, purple, red]
    keys = [keysC, keysB, keysA]
    temps = [5, 11, 17]
    for i in range(3):
        x = np.array([RayleighS[k] for k in keys[i]])
        y = np.array([Nusselt[k][0] for k in keys[i]])
        label = r"$T_\infty$ = " + "{:.0f}".format(temps[i]) + r"$\degree$C"
        axes[0].plot(x, y, '-o', color=colors[i], lw=2, markersize=4, label=label)

    # Rayleigh T
    cmap = cmc.roma_r
    salinities = [0, 5, 10, 15, 20]
    for i in range(5):
        ks = [keysC[i], keysB[i], keysA[i * 2]]
        x = np.array([RayleighT[k] for k in ks])
        y = np.array([Nusselt[k][0] for k in ks])
        ye = np.array([Nusselt[k][1] for k in ks])
        label = '$S_\infty$ = {:.0f} g/kg'.format(salinities[i])
        axes[1].plot(x, y, '-o', color=cmap(int(i * 255 / 5)), lw=2, markersize=4, label=label)
    axes[1].plot([2e8, 9e8], [.7 * (2e8) ** 0.25, .7 * (9e8) ** 0.25], '-k')
    axes[1].text(1.9e8, 88, '$Nu \propto Ra_T^{1/4}$', rotation=45, fontsize=fontsize)

    axes[0].set_xlabel("$Ra_S$", fontsize=fontsize+2)
    axes[0].set_ylabel("$Nu$", fontsize=fontsize+2)
    axes[0].legend(fontsize=fontsize, loc='lower right')
    axes[0].grid()
    axes[0].set_ylim([0, 130])
    axes[0].tick_params(labelsize=fontsize)

    axes[1].set_xlabel("$Ra_T$", fontsize=fontsize+2)
    axes[1].set_ylabel('$Nu$', fontsize=fontsize+2)
    # axes[1].tick_params(labelleft=False)
    axes[1].set_yscale('log')
    axes[1].set_xscale('log')
    axes[1].grid()
    # axes[1].grid(which='both', axis='y')
    # axes[1].grid(which='major', axis='x')
    axes[1].legend(fontsize=fontsize)
    axes[1].set_ylim([35, 135])
    axes[1].set_xlim([1e8, 1e10])
    axes[1].tick_params(labelsize=fontsize)

    for i in range(2):
        axes[i].text(.1, .93, '({:s})'.format('ab'[i]), fontsize=fontsize + 2, transform=axes[i].transAxes, ha='center',
                     va='center')
    plt.show()


def fig_10():
    """ Mean and RMS velocity fields """
    fig, axes = plt.subplots(2, 4)
    n = {'piv1': 990, 'piv2': 8990, 'piv3': 8990, 'piv4': 8990}
    cmap = cmc.vik
    cmap_rms = cmc.vik
    dx, dy = 3, 6  # vector spacing in horizontal, vertical direction
    vmin, vmax = 0, 1.5
    rms_min, rms_max = 0, 0.8
    ims, ims_rms = [], []
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    for i in range(4):
        k = "piv{:d}".format(i+1)
        # put mask on top of mean field
        edges = analysis.compute_average_mask(k)
        edges[:, 0] = analysis.smoothen_array(edges[:, 0], n=50)

        vel = analysis.compute_mean_velocity_field(k, n=n[k])
        rms = analysis.compute_rms_velocity_field(k, n=n[k])
        vel = vel * 100  # m/s -> cm/s
        rms = rms * 100  # m/s -> cm/s

        # remove sides
        vel = vel[1:-3, 1:-1, :]
        rms = rms[1:-3, 1:-1]
        edges[:, 0] -= 1 * 16
        edges[:, 1] -= 3 * 16

        # transform to make all tabs look the same
        xm = int(np.nanmean(edges[:, 0]) / 16)
        xedge = 15
        xlim = 5
        if xm < xedge:
            vel[:, (xedge-xm):] = vel[:, :-(xedge-xm)]
            rms[:, (xedge - xm):] = rms[:, :-(xedge - xm)]
        elif xm > xedge:
            vel[:, :-(xm - xedge)] = vel[:, (xm - xedge):]
            rms[:, :-(xm - xedge)] = rms[:, (xm - xedge):]
        vel = vel[:, :-xlim]
        rms = rms[:, :-xlim]
        edges[:, 0] += 16 * (xedge - xm)

        sz = [d for d in vel.shape[:2]][::-1]
        v = np.sqrt(vel[:, :, 0]**2 + vel[:, :, 1]**2)

        ax = axes[0, i]
        X, Y = np.meshgrid(np.linspace(0, sz[0], v.shape[1]), np.linspace(0, sz[1], v.shape[0]))
        im = ax.imshow(v, extent=[0, sz[0], 0, sz[1]], cmap=cmap, vmin=vmin, vmax=vmax)
        ims.append(im)
        ax.quiver(X[::dy, ::dx], Y[::dy, ::dx], vel[::dy, ::dx, 0], vel[::dy, ::dx, 1], scale=11, width=0.005)

        ax = axes[1, i]
        im = ax.imshow(rms, extent=[0, sz[0], 0, sz[1]], cmap=cmap_rms, vmin=rms_min, vmax=rms_max)
        ims_rms.append(im)

        # draw ice area
        edges = edges / 16
        edges = np.vstack([[-1, -100], edges, [-1, 100]])
        for ax in axes[:, i]:
            ax.add_artist(plt.Polygon(edges, facecolor=(.7, .7, .7)))
            ax.plot(edges[:, 0], edges[:, 1], color=(.3, .3, .3), lw=1.5)

            ax.set_xlim([0, sz[0]])
            ax.set_ylim([0, sz[1]])
            ax.tick_params(labelleft=False, labelbottom=False, left=False, bottom=False)
        axes[0, i].text(0.05, 0.95, "({:s})".format(alphabet[i]), ha='left', va='top', fontsize=12, transform=axes[0, i].transAxes)
        axes[1, i].text(0.05, 0.95, "({:s})".format(alphabet[4+i]), ha='left', va='top', fontsize=12, transform=axes[1, i].transAxes)
    axes[0, 1].add_artist(FancyArrow(50, 25, 0, -12, width=0.2, head_width=2.5, color='w'))
    axes[0, 1].text(45, 20, 'g', color='w', fontsize=18, ha='center', va='center', fontname='Calibri')
    cb = plt.colorbar(ims[0], ax=axes[0, :])
    cb2 = plt.colorbar(ims_rms[0], ax=axes[1, :])
    cb.set_ticks([0, .5, 1, 1.5])
    cb.set_label('Flow speed (cm/s)', fontsize=12)
    cb2.set_label('RMS (cm/s)', fontsize=12)
    plt.show()


def fig_11():
    """ Velocity time series at crests, 4mm from the wall """
    fontsize = 14
    rows = {'piv1': 30, 'piv2': 12, 'piv3': 35, 'piv4': 24}
    dwall = 5  # distance from wall in PIV window size units
    nframes = {'piv1': 990, 'piv2': 8990, 'piv3': 8990, 'piv4': 8990}
    dpx = 16  # distance between values in vel field in pixels
    bsz = 50  # binning size
    cmap = cmc.roma_r
    colors = {'piv1': cmap(0), 'piv2': cmap(0.3), 'piv3': cmap(.7), 'piv4': cmap(.99)}
    sal = {'piv1': 0, 'piv2': 5, 'piv3': 10, 'piv4': 20}

    fig, ax = plt.subplots(2, 1)
    ax[0].plot([0, 3], [0, 0], '-k')
    for k in rows:
        data = analysis.time_series_at_point(k, rows[k], dwall, nframes[k])
        data *= 100  # m/s -> cm/s
        vx = [np.nanmean(data[i:i+bsz, 0]) for i in range(0, nframes[k], bsz)]
        vy = [-np.nanmean(data[i:i + bsz, 1]) for i in range(0, nframes[k], bsz)]
        ax[0].plot(np.arange(len(vx)) * bsz / 3000, vx, color=colors[k], label='${:.0f}$ g/kg'.format(sal[k]))
        ax[1].plot(np.arange(len(vy)) * bsz / 3000, vy, color=colors[k], label='${:.0f}$ g/kg'.format(sal[k]))
    ax[0].grid()
    ax[1].grid()
    ax[1].set_ylim([0, 1.6])
    ax[0].tick_params(labelbottom=False, labelsize=fontsize)
    ax[1].tick_params(top=True, labelsize=fontsize)
    ax[0].legend(fontsize=fontsize, loc='lower right', ncols=2)
    ax[0].set_ylabel('$v_x$ (cm/s)', fontsize=fontsize)
    ax[1].set_ylabel('$v_y$ (cm/s)', fontsize=fontsize)
    ax[1].set_xlabel('Time (min.)', fontsize=fontsize)
    ax[0].set_xlim([0, 3])
    ax[1].set_xlim([0, 3])
    ax[0].set_ylim([-0.3, 0.3])
    ax[1].set_yticks([0, .5, 1, 1.5])
    ax[0].text(0.98, 0.95, '(a)', ha='right', va='top', transform=ax[0].transAxes, fontsize=fontsize)
    ax[1].text(0.98, 0.95, '(b)', ha='right', va='top', transform=ax[1].transAxes, fontsize=fontsize)
    plt.show()


def fig_12():
    """ x- and y-velocity profiles from wall """
    fontsize = 14
    rows = {'piv1': [30], 'piv2': [12, 30], 'piv3': [35, 53], 'piv4': [24, 34]}
    dwall = [1, 5, 9, 18]  # distance from wall
    n = {'piv1': 990, 'piv2': 8990, 'piv3': 8990, 'piv4': 8990}
    dpx = 16  # distance between values in vel field in pixels
    dy = 1  # number of rows on either side to include in average
    linestyles = ['-', '--']
    cmap = cmc.roma_r
    colors = {'piv1': cmap(0), 'piv2': cmap(0.3), 'piv3': cmap(.7), 'piv4': cmap(.99)}
    labels = {'piv1': ['0 g/kg'], 'piv2': ['5 g/kg, crest', '5 g/kg, trough'],
              'piv3': ['10 g/kg, crest', '10 g/kg, trough'], 'piv4': ['20 g/kg, crest', '20 g/kg, trough']}
    labels = {'piv1': ['0 g/kg'], 'piv2': ['5 g/kg', '5 g/kg, trough'],
              'piv3': ['10 g/kg', '10 g/kg, trough'], 'piv4': ['20 g/kg', '20 g/kg, trough']}
    ccal = load_settings('piv1')['Ccal']

    # profiles
    figS, axS = plt.subplots()
    figXY, axesXY = plt.subplots(nrows=2)

    axS.plot([-1, 5], [0, 0], color='k', lw=1)
    axS.plot([0, 0], [-.1, 1.5], color='k', lw=1)
    for ax in axesXY:
        ax.plot([-1, 5], [0, 0], color='k', lw=1)
        ax.plot([0, 0], [-2, 2], color='k', lw=1)
        for dw in dwall:
            ax.plot([dw*dpx/ccal, dw*dpx/ccal], [-1, 2], '-', color=(.7, .7, .7), lw=1)
            ax.plot(dw*dpx/ccal, 0.2 if ax==axesXY[0] else 1.5, 'k', marker=7, markersize=6, clip_on=False)
            ax.plot(dw * dpx / ccal, -0.15 if ax == axesXY[0] else -0.1, 'k', marker=6, markersize=6, clip_on=False)
        ax.grid()
    for k in '1234':
        k = 'piv' + k
        mvel = analysis.compute_mean_velocity_field(k, n=n[k])
        edges = analysis.compute_average_mask(k)
        edges[:, 0] = edges[::-1, 0]

        for j, i in enumerate(rows[k]):
            vx = np.mean(mvel[i - dy:i + dy + 1, :, 0], axis=0) * 100  # x-velocity in cm/s
            vy = -1 * np.mean(mvel[i - dy:i + dy + 1, :, 1], axis=0) * 100  # y-velocity in cm/s (downward)
            v = np.sqrt(vx**2 + vy**2)  # speed in cm/s
            xw = np.mean([ex for ex, ey in edges if i-dy < ey/dpx < i+dy+1]) / dpx  # average wall x-coordinate
            # xw = int(xw)
            xw -= 1  # account for 1 window size unit inaccuracy in the wall location
            x = (np.arange(v.size) - xw) * dpx / ccal  # distance from wall in cm

            axS.plot(x, v, linestyles[j], color=colors[k], label=labels[k][j])
            axesXY[0].plot(x, vx, linestyles[j], color=colors[k], label=labels[k][j])
            axesXY[1].plot(x, vy, linestyles[j], color=colors[k], label=labels[k][j] if j == 0 else None)
    axS.set_xlabel('$x - x_w$ (cm)', fontsize=14)
    axS.set_ylabel('$|v|$ (cm s$^{-1}$)', fontsize=14)
    axS.set_xlim([-.5, 5])
    axS.set_ylim([-.1, 1.5])
    axS.grid()
    axS.legend()

    axesXY[1].set_xlabel('$x - x_w$ (cm)', fontsize=14)
    axesXY[0].set_ylabel('$v_x$ (cm/s)', fontsize=14)
    axesXY[1].set_ylabel('$v_y$ (cm/s)', fontsize=14)
    axesXY[0].set_xlim([0, 5])
    axesXY[0].set_ylim([-.15, .2])
    axesXY[1].set_xlim([0, 5])
    axesXY[1].set_ylim([-.1, 1.5])
    axesXY[1].legend(fontsize=fontsize, loc='upper left', bbox_to_anchor=(0.6, 0.99))
    axesXY[0].tick_params(labelbottom=False, labelsize=fontsize)
    axesXY[1].tick_params(top=True, labelsize=fontsize)
    axesXY[0].set_yticks([-.1, 0, .1, .2])
    axesXY[1].set_yticks([0, .5, 1, 1.5])
    axesXY[0].text(0.98, 0.95, '(a)', ha='right', va='top', transform=axesXY[0].transAxes, fontsize=fontsize)
    axesXY[1].text(0.98, 0.95, '(b)', ha='right', va='top', transform=axesXY[1].transAxes, fontsize=fontsize)

    plt.show()


def fig_13():
    """ Location of lines and points for profiles and histograms, respectively """
    rows = {'piv1': [30], 'piv2': [12, 30], 'piv3': [35, 53], 'piv4': [24, 34]}
    dwall = [1, 5, 9, 18]  # distance from wall
    n = {'piv1': 990, 'piv2': 8990, 'piv3': 8990, 'piv4': 8990}
    dpx = 16  # distance between values in vel field in pixels
    dy = 1  # number of rows on either side to include in average
    vmin, vmax = 0, 1.4
    cmap = cmc.roma_r
    colors = {'piv1': cmap(0), 'piv2': cmap(0.3), 'piv3': cmap(.7), 'piv4': cmap(.99)}

    fig, axes = plt.subplots(2, 2)
    for j in range(4):
        k = 'piv{:d}'.format(j+1)
        ax = axes[j // 2, j % 2]

        # put mask on top of mean field
        edges = analysis.compute_average_mask(k)
        edges[:, 0] = analysis.smoothen_array(edges[:, 0], n=50)

        vel = analysis.compute_mean_velocity_field(k, n=n[k])
        vel = vel * 100  # m/s -> cm/s

        # remove sides
        vel = vel[1:-3, 1:-1, :]
        edges[:, 0] -= 1 * dpx
        edges[:, 1] -= 3 * dpx

        # transform to make all tabs look the same
        xm = int(np.nanmean(edges[:, 0]) / dpx)
        xedge = 15
        xlim = 5
        if xm < xedge:
            vel[:, (xedge - xm):] = vel[:, :-(xedge - xm)]
        elif xm > xedge:
            vel[:, :-(xm - xedge)] = vel[:, (xm - xedge):]
        vel = vel[:, :-xlim]
        edges[:, 0] += dpx * (xedge - xm)

        sz = [d for d in vel.shape[:2]][::-1]
        v = np.sqrt(vel[:, :, 0] ** 2 + vel[:, :, 1] ** 2)
        ax.imshow(v, extent=[0, sz[0], 0, sz[1]], cmap=cmc.vik, vmin=vmin, vmax=vmax)

        # draw ice area
        edges = edges / dpx
        edges = np.vstack([[-1, -100], edges, [-1, 100]])
        ax.add_artist(plt.Polygon(edges, facecolor=(.7, .7, .7), zorder=5))
        ax.plot(edges[:, 0], edges[:, 1], color=(.3, .3, .3), lw=1.5, zorder=6)

        ax.set_xlim([0, sz[0]])
        ax.set_ylim([0, sz[1]])
        ax.tick_params(labelleft=False, labelbottom=False, left=False, bottom=False)
        ax.text(0.05, 0.95, "({:s})".format(k[1]), ha='left', va='top', fontsize=14, transform=ax.transAxes, zorder=10)

        for i, yr in enumerate(rows[k]):
            yr = v.shape[0] - yr
            ax.plot([0, v.shape[1]], [yr, yr], ['-', '--'][i], lw=6, color=colors[k])
            xw = np.mean([ex for ex, ey in edges if yr-dy < ey < yr+dy+1])  # average wall x-coordinate
            for dw in dwall:
                marker = 'D' if dw == 5 and i == 0 else 'o'
                ax.plot(xw + dw, yr, 'k', markersize=6, mfc='w', mew=1.5, marker=marker)
    axes[1, 0].add_artist(FancyArrow(40, 50, 13, 0, width=0.2, head_width=1.7, color='w'))
    axes[1, 0].add_artist(FancyArrow(40, 50, 0, -13, width=0.2, head_width=1.7, color='w'))
    axes[1, 0].text(35, 38, 'y', color='w', fontsize=14, ha='center', va='center')
    axes[1, 0].text(52, 55, 'x', color='w', fontsize=14, ha='center', va='center')
    plt.show()


def fig_14_15():
    """ Compute velocity histograms at given points """
    nframes = {'piv1': 990, "piv2": 8990, "piv3": 8990, "piv4": 8990}
    # rows = {'piv1': [30], 'piv2': [15, 32], 'piv3': [38, 56], 'piv4': [27, 35]}
    rows = {'piv1': [30], 'piv2': [12, 30], 'piv3': [35, 53], 'piv4': [24, 34]}
    dwall = [1, 5, 9, 18]  # distance from wall
    dpx = 16  # distance between values in vel field in pixels
    dy, dx = 1, 0  # number of points on either side to include in the average
    ccal = load_settings('piv1')['Ccal']  # px per cm
    linestyles = ['-', '--']
    cmap = cmc.roma_r
    colors = {'piv1': cmap(0), 'piv2': cmap(0.3), 'piv3': cmap(.7), 'piv4': cmap(.99)}
    labels = {'piv1': ['$x - x$ = {:.0f} mm'], 'piv2': ['$x - x_w$ = {:.0f} mm, crest', '$x - x_w$ = {:.0f} mm, trough'],
              'piv3': ['$x - x_w$ = {:.0f} mm, crest', '$x - x_w$ = {:.0f} mm, trough'], 'piv4': ['$x - x_w$ = {:.0f} mm, crest', '$x - x_w$ = {:.0f} mm, trough']}
    titles = {'piv1': '$S_\infty = $ 0 g/kg', 'piv2': '5 g/kg', 'piv3': '10 g/kg', 'piv4': '20 g/kg'}
    alphabet = 'abcdefghijklmnopqrstuvwxyz'

    # make and show histograms
    fig_x, axes_x = plt.subplots(len(dwall), 4)
    fig_y, axes_y = plt.subplots(len(dwall), 4)
    axes = [axes_x, axes_y]
    for ki, k in enumerate('1234'):
        k = 'piv' + k
        hist, hist_x = analysis.histogram_at_point(k, rows[k], dwall, n_frames=nframes[k])
        for a in range(len(hist)):
            for i in range(len(hist[a])):
                for j in range(len(hist[a][i])):
                    dw = dwall[j] * dpx / ccal * 10  # distance from wall in mm
                    h = analysis.smoothen_array(hist[a][i][j], n=3)
                    hx = hist_x[a][i][j]
                    axes[a][j, ki].plot([0, 0], [0, 100], '-k', lw=1, zorder=1)
                    axes[a][j, ki].plot(hx, h, linestyles[i], color=colors[k], label=labels[k][i].format(dw))
                    axes[a][j, ki].grid(visible=True)
                    axes[a][j, ki].tick_params(labelbottom=j==len(dwall)-1, labelleft=ki==0, top=j>0, right=ki<3)
                    if a == 1:
                        axes[a][j, ki].set_xticks([0, 1, 2])
                        axes[a][j, ki].text(0.95, 0.95, "({:s})".format(alphabet[4 * j + ki]), ha='right', va='top',
                                            transform=axes[a][j, ki].transAxes)
                    else:
                        axes[a][j, ki].text(0.15, 0.95, "({:s})".format(alphabet[4 * j + ki]), ha='left', va='top',
                                            transform=axes[a][j, ki].transAxes)
        for j in range(len(dwall)):
            axes[0][j, ki].set_ylim([0, 30])
            axes[0][j, ki].set_xlim([-.24, .24])
            axes[1][j, ki].set_ylim([0, 10])
            axes[1][j, ki].set_xlim([-0.4, 2])
        # axes[1][2, ki].set_ylim([0, 15])
        axes[0][0, ki].set_title(titles[k])
        axes[1][0, ki].set_title(titles[k])

    for j in range(len(dwall)):
        for a in range(len(axes)):
            axes[a][j, -1].yaxis.set_label_position("right")
            axes[a][j, -1].set_ylabel("$\Delta x = {:.0f}$ mm".format(dwall[j] * dpx / ccal * 10), rotation=270, labelpad=15)
    fig_x.supxlabel('$v_x$ (cm s$^{-1}$)', fontsize=14)
    fig_y.supxlabel('$v_y$ (cm s$^{-1}$)', fontsize=14)
    fig_x.supylabel('PDF (s cm$^{-1}$)', fontsize=14)
    fig_y.supylabel('PDF (s cm$^{-1}$)', fontsize=14)
    fig_x.text(-0.3, 1.3, 'I)', fontsize=18, transform=axes_x[0, 0].transAxes, ha='right')
    fig_y.text(-0.3, 1.3, 'II)', fontsize=18, transform=axes_y[0, 0].transAxes, ha='right')
    plt.show()


def fig_16():
    """ Scallop amplitude in T-S and GrT-GrS plot """
    from matplotlib.markers import MarkerStyle

    def T_for_SR(sal, r):
        # compute temperature T, given salinity S and density ratio R
        tp = np.linspace(0, 60, 1000)
        w = Seawater(tp, np.ones(tp.size) * sal)
        return tp[np.argmin(np.abs(w.density_ratio() - r))]

    GrT, GrS = analysis.compute_GrashofT_number, analysis.compute_GrashofS_number
    px = 1 / plt.rcParams['figure.dpi']
    fig, ax = plt.subplots(1, 2, figsize=(1900*px, 500*px))
    s = np.linspace(0, 160, 1000)
    for k in ALL_KEYS:
        rp = analysis.compute_roughness_parameters(k)
        T, S = SETTINGS[k]["T"], SETTINGS[k]["SP"]

        ampL, ampR = rp[0]['sigma'], rp[1]['sigma']
        ampL = ampL / SETTINGS[k]["Ccal"] * 10  # px -> mm
        ampR = ampR / SETTINGS[k]["Ccal"] * 10  # px -> mm
        zorder = 5 + int(10*(ampL + ampR))

        markerL = MarkerStyle('o', fillstyle='left')
        markerR = MarkerStyle('o', fillstyle='right')

        if S > 0:
            ax[0].scatter(S, T, s=70, c=ampL, marker=markerL, cmap='plasma', edgecolors='k', linewidths=.5, vmin=0, vmax=1, zorder=zorder)
            ax[0].scatter(S, T, s=70, c=ampR, marker=markerR, cmap='plasma', edgecolors='k', linewidths=.5, vmin=0, vmax=1, zorder=zorder)
            ax[1].scatter(GrS(k), GrT(k), s=70, c=ampL, marker=markerL, cmap='plasma', edgecolors='k', linewidths=.5, vmin=0, vmax=1, zorder=zorder)
            ax[1].scatter(GrS(k), GrT(k), s=70, c=ampR, marker=markerR, cmap='plasma', edgecolors='k', linewidths=.5, vmin=0, vmax=1, zorder=zorder)

            # if k == '4e':
            #     ax[0].plot(S, T, 'o', markersize=20, color=(.8, 0, 0), mfc='none')
            #     ax[1].plot(GrS(k), GrT(k), 'o', markersize=20, color=(.8, 0, 0), mfc='none')

    # add R = 1
    ax[0].plot(s, [T_for_SR(sl, 1) for sl in s], '-k', lw=1.5)
    ax[1].plot([1e5, 2e11], [1 * 1e5, 1 * 2e11], '-k', lw=1.5)

    # add lower bound from Josberger 1981
    t = np.array([T_for_SR(sl, 0.157) for sl in s])
    ax[0].plot(s[t>10], t[t>10], '--k', lw=1.5, label='Josberger & Martin (1981)')
    sw = Seawater(10, 10)
    min_GrT = -9.81 * sw.density_derivative_t() * sw.density() * sw.t * 0.3 ** 3 / (sw.dynamic_viscosity()**2)
    ax[1].plot([min_GrT / 0.157, 2e11], [min_GrT, .157 * 2e11], '--k', lw=1.5, label='Josberger & Martin (1981)')

    # add lower bound from CG 1982b
    sw = Seawater(10 + s / 3, s)
    gt = -9.81 * sw.density_derivative_t() * sw.density() * sw.t * 0.3 ** 3 / (sw.dynamic_viscosity()**2)
    gs = 9.81 * sw.density_derivative_s() * sw.density() * sw.s * 0.3 ** 3 / (sw.dynamic_viscosity()**2)
    gs[gs < gt] = np.nan
    gt[np.isnan(gs)] = np.nan
    ax[0].plot(s[sw.density_ratio() <= 1], 10 + s[sw.density_ratio() <= 1] / 3, '-.k', label='Carey & Gebhart (1982a)')
    ax[1].plot(gs, gt, '-.k', label='Carey & Gebhart (1982a)')

    # add lower bound from Sammakia 1983
    sw = Seawater(11 + (s-3) * 5./32, s)
    gt = -9.81 * sw.density_derivative_t() * sw.density() * sw.t * 0.3 ** 3 / (sw.dynamic_viscosity() ** 2)
    gs = 9.81 * sw.density_derivative_s() * sw.density() * sw.s * 0.3 ** 3 / (sw.dynamic_viscosity() ** 2)
    gs[gs < gt] = np.nan
    gt[np.isnan(gs)] = np.nan
    ax[0].plot(s[sw.density_ratio() <= 1], 11 + (s[sw.density_ratio() <= 1] - 3) * 5. / 32, ':k', label='Sammakia & Gebhart (1983)')
    ax[1].plot(gs, gt, ':k', label='Sammakia & Gebhart (1983)')

    ax[0].grid()
    ax[0].set_xlabel(r"$S_\infty$ (g/kg)", fontsize=12)
    ax[0].set_ylabel(r"$T_\infty$ ($\degree$C)", fontsize=12)
    ax[0].set_xlim([0, 120])
    ax[0].set_ylim([0, 60])
    ax[0].legend(fontsize=10, loc='lower right')
    ax[0].tick_params(labelsize=12)
    ax[0].text(20, 43, r'$R_\rho = 1.0$', fontsize=12, rotation=60, ha='center', va='center')
    ax[0].text(0.05, 0.95, '(a)', ha='left', va='top', transform=ax[0].transAxes, fontsize=14)

    ax[1].grid()
    ax[1].set_xlabel('$Gr_S$ (-)', fontsize=12)
    ax[1].set_ylabel('$Gr_T$ (-)', fontsize=12)
    ax[1].set_xscale('log')
    ax[1].set_yscale('log')
    ax[1].set_ylim([1e7, 3e10])
    ax[1].set_xlim([1e8, 1e11])
    ax[1].tick_params(labelsize=12)
    ax[1].text(7.8e8, 1.13e9, r'$R_\rho = 1.0$', fontsize=12, rotation=35, ha='center', va='center')
    ax[1].text(0.05, 0.95, '(b)', ha='left', va='top', transform=ax[1].transAxes, fontsize=14)
    # ax[1].text(2.28e9, 2e8, r'$R_\rho = 0.157$', fontsize=12, rotation=35, ha='center', va='center')
    plt.subplots_adjust(left=0.048, right=0.99, top=.95, bottom=0.106)
    cb = plt.colorbar(plt.cm.ScalarMappable(plt.Normalize(vmin=0, vmax=1), cmap=plt.get_cmap('plasma')), ax=ax)
    cb.set_label('Scallop amplitude (mm)', fontsize=12)
    cb.ax.tick_params(labelsize=12)

    plt.show()


def fig_17():
    """ Radius, r - <r> and dr/dt as function of height and time """
    exps = ['a3', 'a5']
    side = 0  # 0 = left, 1 = right
    dn = 6  # number of time steps to skip for time derivative
    # cmc.show_cmaps()

    fig, axes = plt.subplots(2, 2)
    cmaps = [cmc.nuuk, cmc.broc, cmc.nuuk]
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    for i, k in enumerate(exps):
        cntrs = analysis.get_contours(k)
        toff = analysis.find_time_offset(k)
        x0 = np.mean([x for x, y in cntrs[toff]])
        mat = np.zeros((5000, len(cntrs)-toff, 2))
        for j, c in enumerate(cntrs[toff:]):
            X = np.array([x for x, y in c])
            Y = np.array([y for x, y in c])
            X1, X2 = np.abs(X[X <= x0] - x0), np.abs(X[X > x0] - x0)
            Y1, Y2 = Y[X <= x0], Y[X > x0]
            mat[Y1, j, 0] = X1
            mat[Y2, j, 1] = X2

        mat = mat / SETTINGS[k]["Ccal"] * 10  # px -> mm
        tmax, ymax = mat.shape[1] * (SETTINGS[k]["dt"] / 60), mat.shape[0] / SETTINGS[k]["Ccal"]

        # r(y, t)
        mat[mat == 0] = np.nan
        im1 = axes[0, i].imshow(mat[:, :, side], aspect='auto', extent=[0, tmax, ymax, 0], vmin=0, vmax=25, cmap=cmaps[0])
        axes[0, i].tick_params(labelbottom=False)
        axes[0, i].set_facecolor((.7, .7, .7))

        # r - <r>
        m1 = mat[:, :, side]
        m1[m1==0] = np.nan
        im2 = axes[1, i].imshow(m1 - np.nanmean(m1, axis=0), aspect='auto', vmin=-3, vmax=3, extent=[0, tmax, ymax, 0], cmap=cmaps[1])
        axes[1, i].tick_params(labelbottom=False)
        axes[1, i].set_facecolor((.7, .7, .7))

        # # dr/dt
        # drdt = -(mat[:, dn:] - mat[:, :-dn]) / (dn * SETTINGS[k]["dt"] / 60)  # mm/min. (inward is positive)
        # im3 = axes[2, i].imshow(drdt[:, :, side], aspect='auto', vmin=0, vmax=2, extent=[0, tmax, ymax, 0], cmap=cmaps[2])
        # axes[2, i].set_facecolor((.7, .7, .7))

        for j in range(axes.shape[0]):
            axes[j, i].set_xlim([0, 25])
            axes[j, i].set_ylim([20, 0])
            axes[j, i].tick_params(labelbottom=j==axes.shape[0]-1, top=j>0, labelleft=i==0, right=i==0, labelsize=12)
            axes[j, i].text(0.02, 0.95, '({:s})'.format(alphabet[2*j+i]), transform=axes[j, i].transAxes, fontsize=14, ha='left', va='top')
            axes[j, 0].set_ylabel('y (cm)', fontsize=14)
    axes[0, 0].set_title('$S_\infty = {:.0f}$ g/kg'.format(SETTINGS[exps[0]]["SP"]), fontsize=14)
    axes[0, 1].set_title('${:.0f}$ g/kg'.format(SETTINGS[exps[1]]["SP"]), fontsize=14)
    axes[-1, 0].set_xlabel('Time (min.)', fontsize=14)
    axes[-1, 1].set_xlabel('Time (min.)', fontsize=14)
    cb1 = plt.colorbar(im1, ax=axes[0, :], aspect=10)
    cb1.set_label('r(y, t) (mm)', fontsize=14)
    cb1.ax.tick_params(labelsize=12)
    cb2 = plt.colorbar(im2, ax=axes[1, :], aspect=10)
    cb2.set_label(r'$r - \langle r\rangle_y$ (mm)', fontsize=14)
    cb2.ax.tick_params(labelsize=12)
    # cb3 = plt.colorbar(im3, ax=axes[2, :], aspect=10)
    # cb3.set_label(r'$\dot{r}$ (mm/min.)', fontsize=14)
    # cb3.ax.tick_params(labelsize=12)
    # cb3.ax.set_yticks([0, 1, 2])

    # # reference lines
    # a1 = (13.65 - 11.4) / (20 - 10)
    # b1 = 11.4 - a1 * 10 + 0.9
    # a2 = (7.65 - 6.4) / (20 - 10)
    # b2 = 6.4 - a2 * 10 + 0.6
    # x = np.array([12, 17])
    # axes[1, 0].plot(x, a1*x + b1, '-r')
    # axes[1, 1].plot(x, a2*x + b2, '-r')
    # axes[2, 0].plot(x, a1*x + b1, '-r')
    # axes[2, 1].plot(x, a2*x + b2, '-r')
    plt.show()


def fig_18():
    """ Wavelength and scallop speed as function of density ratio """
    sal, vel, vel_std, temp, d_ratio, wl, wl_std = [], [], [], [], [], [], []
    keys = ["a" + k for k in "3456789"] + ['b2', 'd1', 'd2', 'e2', 'e3', 'f2']
    for i, k in enumerate(keys):
        V_avg, V_std = analysis.compute_scallop_speed(k)
        w = Seawater(SETTINGS[k]["T"], SETTINGS[k]["SP"])
        R = -w.density_derivative_t() * w.t / (w.density_derivative_s() * w.s)

        L = []
        contours = analysis.get_contours(k)
        hvi, toff = analysis.find_half_volume_index(k), analysis.find_time_offset(k)
        start_ind = hvi
        end_ind = toff + int(2.5 * (hvi - toff))
        end_ind = min(len(contours), end_ind)
        for j in range(start_ind, end_ind):
            L += analysis.compute_wavelengths(contours[j])
        analysis.dump_to_cache(L, 'wavelengths_'+k)
        # L = analysis.get_from_cache('wavelengths_' + k)
        if k == 'd1':
            L = [val for val in L if val > SETTINGS[k]["Ccal"]]
        elif k == 'd2':
            L = [val for val in L if val < 5 * SETTINGS[k]["Ccal"]]

        sal.append(SETTINGS[k]["SP"])
        d_ratio.append(R)
        vel.append(V_avg)
        vel_std.append(V_std)
        temp.append(SETTINGS[k]['T'])
        wl.append(np.mean(L) / SETTINGS[k]["Ccal"])
        wl_std.append(np.std(L) / SETTINGS[k]["Ccal"])

    fig, axes = plt.subplots(2, 1)
    axes[1].plot([0, 40], [0, 0], '-', lw=1, color=[.3, .3, .3])
    fmt = ['o', '^', 's', 'p']
    msize = [7, 8, 7, 9]
    cmap = cmc.bamako
    # acton, nuuk, bamako
    for i in range(len(sal)):
        j = 3 if temp[i] > 25 else 2 if temp[i] > 20 else 1 if temp[i] > 15 else 0
        c = cmap(sal[i] / 35)
        axes[1].errorbar(d_ratio[i], vel[i], yerr=vel_std[i], fmt=fmt[j], capsize=3, color='k', mfc=c, markersize=msize[j])
        axes[0].errorbar(d_ratio[i], wl[i], yerr=wl_std[i], fmt=fmt[j], capsize=3, color='k', mfc=c, markersize=msize[j])
    handles = [axes[1].errorbar(2, 1, yerr=1, fmt=fmt[j], capsize=3, color='k', mfc=(.9, .9, .9), markersize=msize[j] - 1)
               for j in range(4)]
    axes[1].legend(handles, ['$10 < T_\infty \leq 15 \degree$C', '$15 < T_\infty \leq 20 \degree$C',
                         '$20 < T_\infty \leq 25\degree$C', '$T_\infty = 50 \degree$C'], loc='upper left')
    axes[0].grid()
    axes[0].set_xlim([0, 1.2])
    axes[0].set_ylim([0, 4])
    axes[0].set_ylabel('$\lambda$ (cm)', fontsize=14)
    axes[0].tick_params(labelbottom=False)
    axes[0].text(0.98, 0.95, '(a)', ha='right', va='top', fontsize=14, transform=axes[0].transAxes)

    axes[1].grid()
    axes[1].set_xlim([0, 1.2])
    axes[1].set_ylim([-1, 5])
    axes[1].set_xlabel(r'$R_\rho$', fontsize=14)
    # axes[1].set_ylabel(r'$\dot{h}/\dot{r}$', fontsize=14)
    axes[1].set_ylabel(r'$u_m$ (-)', fontsize=14)
    axes[1].tick_params(top=True)
    axes[1].text(0.98, 0.95, '(b)', ha='right', va='top', fontsize=14, transform=axes[1].transAxes)

    cb = plt.colorbar(plt.cm.ScalarMappable(plt.Normalize(0, 35), cmap=cmap), ax=axes)
    cb.set_label("$S_\infty$ (g/kg)", fontsize=14)
    plt.show()


""" 
    APPENDIX 
"""


def fig_20():
    """ All contours of exp3 over time """
    cmap = "Reds_r"
    temp = "17"
    exps = ['a' + k for k in '123456789']
    height = 16.5  # fixed height from bottom of cylinder

    intervals = np.arange(0, 3, .5)
    colors = [plt.get_cmap(cmap, len(intervals) + 1)(i) for i in range(len(intervals))]

    fig, ax = plt.subplots(ncols=len(exps), figsize=[8, 5])
    plt.subplots_adjust(left=0.18, right=0.902, wspace=0.04)
    # plt.subplots_adjust(left=0.02, right=0.99, wspace=0.04)
    for j, k in enumerate(exps):
        i_half = analysis.find_half_volume_index(k)
        i_off = analysis.find_time_offset(k)
        idx = [i_off + int((i_half - i_off) * intv) for intv in intervals]
        contours = analysis.get_contours(k)
        cx = np.mean(contours[i_off][:, 0])  # center x-location in pixels
        by = np.max(contours[i_off][:, 1]) / SETTINGS[k]["Ccal"]  # bottom y-location in cm
        print(k + ": " + ", ".join([str(i) for i in idx]) + " | max: " + str(len(contours)))

        # exceptions
        match k:
            case "a1":
                idx[4] -= 1
            case "a2":
                idx[4] -= 2
            case "a5":
                idx[5] -= 6
            case "a8":
                idx[5] -= 2

        for i in range(len(intervals)):
            if idx[i] < len(contours):
                x = np.array(analysis.smoothen_array(contours[idx[i]][:, 0]) - cx) / SETTINGS[k]["Ccal"]
                y = np.array(analysis.smoothen_array(contours[idx[i]][:, 1])) / SETTINGS[k]["Ccal"]
                x = x[y > by-height]
                y = y[y > by-height] - (by - height)
                xy = np.array([x, y]).T
                ax[j].add_artist(plt.Polygon(xy, closed=True, ec='k', fc=colors[i], lw=0.5))
        ax[j].set_xlim([-3, 3])
        ax[j].set_ylim([0, 20])
        ax[j].set_aspect('equal')
        ax[j].invert_yaxis()
        ax[j].tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
        ax[j].set_title("{:.1f} g/kg".format(SETTINGS[k]["SP"]))
        panel_label = r"(a$_{:d}$)".format(j+1)
        ax[j].text(2.5, 19.5, panel_label, ha='right', va='bottom', fontsize=14)
    ax[0].add_artist(plt.Rectangle((-2.5, 19.5), 2, 0.2, ec='k', fc='k'))
    ax[0].text(-1.5, 19.2, '2 cm', ha='center', fontsize=14)
    ax[0].set_title("$S_\infty = 0$ g/kg")
    ax[0].set_ylabel(r'$T_\infty = {:s}\degree$C'.format(temp), fontsize=14)
    cb = plt.colorbar(plt.cm.ScalarMappable(plt.Normalize(vmin=0, vmax=2 * intervals[-1] - intervals[-2]),
                                            cmap=ListedColormap(colors)), ax=ax, fraction=0.1, shrink=1, extend='max')
    cb.ax.set_title(r'$\tau$ (-)', fontsize=14)
    plt.show()


def fig_21_22_23():
    """ All contours of expDEF over time """
    cmap = "Greens_r"
    experiments = [['d' + k for k in '1234'], ['e' + k for k in '1234'], ['f' + k for k in '1234']]
    height = 16.5  # fixed height from bottom of cylinder
    for exp_label, exps in zip('def', experiments):
        intervals = np.arange(0, 3, .5)
        colors = [plt.get_cmap(cmap, len(intervals) + 1)(i) for i in range(len(intervals))]

        fig, ax = plt.subplots(ncols=len(exps), figsize=[8, 5])
        plt.subplots_adjust(left=0.18, right=0.902, wspace=0.04)
        # plt.subplots_adjust(left=0.02, right=0.99, wspace=0.04)
        for j, k in enumerate(exps):
            i_half = analysis.find_half_volume_index(k)
            i_off = analysis.find_time_offset(k)
            idx = [i_off + int((i_half - i_off) * intv) for intv in intervals]
            contours = analysis.get_contours(k)
            cx = np.mean(contours[i_off][:, 0])  # center x-location in pixels
            by = np.max(contours[i_off][:, 1]) / SETTINGS[k]["Ccal"]  # bottom y-location in cm
            print(k + ": " + ", ".join([str(i) for i in idx]) + " | max: " + str(len(contours)))

            # exceptions
            match k:
                case "f2":
                    idx[4] -= 4

            for i in range(len(intervals)):
                if idx[i] < len(contours):
                    x = np.array(analysis.smoothen_array(contours[idx[i]][:, 0]) - cx) / SETTINGS[k]["Ccal"]
                    y = np.array(analysis.smoothen_array(contours[idx[i]][:, 1])) / SETTINGS[k]["Ccal"]
                    x = x[y > by-height]
                    y = y[y > by-height] - (by - height)
                    xy = np.array([x, y]).T
                    ax[j].add_artist(plt.Polygon(xy, closed=True, ec='k', fc=colors[i], lw=0.5))
            ax[j].set_xlim([-3, 3])
            ax[j].set_ylim([0, 20])
            ax[j].set_aspect('equal')
            ax[j].invert_yaxis()
            ax[j].tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
            if j == 0:
                ax[j].set_title("$T_\infty = ${:.1f}$\degree$C\n$S_\infty = ${:.1f} g/kg".format(SETTINGS[k]["T"], SETTINGS[k]["SP"]))
            else:
                ax[j].set_title("{:.1f}$\degree$C\n{:.1f} g/kg".format(SETTINGS[k]["T"], SETTINGS[k]["SP"]))
            panel_label = r"({:s}$_{:d}$)".format(exp_label, j+1)
            ax[j].text(2.5, 19.5, panel_label, ha='right', va='bottom', fontsize=14)
        ax[0].add_artist(plt.Rectangle((-2.5, 19.5), 2, 0.2, ec='k', fc='k'))
        ax[0].text(-1.5, 19.2, '2 cm', ha='center', fontsize=14)
        cb = plt.colorbar(plt.cm.ScalarMappable(plt.Normalize(vmin=0, vmax=2 * intervals[-1] - intervals[-2]),
                                                cmap=ListedColormap(colors)), ax=ax, fraction=0.1, shrink=1, extend='max')
        cb.ax.set_title(r'$\tau$ (-)', fontsize=14)
    plt.show()


def refresh_cache():
    # contours
    for k in ALL_KEYS:
        print("\rRefreshing cache for exp {:s}".format(k), end='')

        analysis.find_time_offset(k, from_cache=False)
        analysis.find_half_volume_time(k, from_cache=False)
        analysis.find_half_volume_index(k, from_cache=False)
        analysis.find_times(k, from_cache=False)

        V = [analysis.compute_volume(c, ccal=SETTINGS[k]["Ccal"]) for c in analysis.get_contours(k)]
        analysis.dump_to_cache(V, 'volume_' + k)
        analysis.compute_roughness_parameters(k, from_cache=False)
    print("\n\n")

    for k in ["a" + k for k in "3456789"] + ['b2', 'd1', 'd2', 'e2', 'e3', 'f2']:
        analysis.compute_scallop_speed(k, from_cache=False)

    # velocity field
    n = {'piv1': 990, 'piv2': 8990, 'piv3': 8990, 'piv4': 8990}
    rows = {'piv1': [30], 'piv2': [12, 30], 'piv3': [35, 53], 'piv4': [24, 34]}
    dwall = [1, 5, 9, 18]  # distance from wall
    for k in n:
        print("\rRefreshing cache for exp {:s}".format(k), end='')
        analysis.piv_advection(k, from_cache=False)
        analysis.piv_melt_rate(k, from_cache=False)
        analysis.compute_average_mask(k, from_cache=False)
        analysis.compute_mean_velocity_field(k, n=n[k], from_cache=False)
        analysis.compute_rms_velocity_field(k, n=n[k], from_cache=False)
        analysis.histogram_at_point(k, rows[k], dwall, n_frames=n[k], from_cache=False)
        analysis.time_series_at_point(k, rows[k][0], dwall[1], n_frames=n[k], from_cache=False)


if __name__ == "__main__":
    # refresh_cache()
    show_figures()

