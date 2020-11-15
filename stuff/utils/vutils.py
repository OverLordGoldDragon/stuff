import numpy as np
import matplotlib.pyplot as plt

#%%# WAVES ####################################################################
def complex_to_polar(x):
    return np.angle(x), np.abs(x)

def sines(freqs, plot=False, N=120):
    freqs = freqs if isinstance(freqs, (list, np.ndarray, tuple)) else [freqs]
    theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
    S = [np.cos(f * theta) for f in freqs]
    if plot:
        _=[plt.plot(s) for s in S]
        # plt.plot(np.sum(S))
    return S


def sinusoids(coeffs, odd=False):
    thetas, rs = complex_to_polar(coeffs)
    N = 2 * (len(coeffs) - 1) + int(odd)
    x = np.linspace(0, 2 * np.pi, N, endpoint=False)

    sines = []
    for f, (theta, r) in enumerate(zip(thetas, rs)):
        sines.append(r * np.cos(f * x - theta))
    return sines


def cos(f, dc=0, xmin=0, xmax=2 * np.pi, N=128, phi=0, plot=True):
    y = np.cos(f * np.linspace(xmin, xmax, N, endpoint=False) + phi) + dc
    x = np.linspace(0, 2 * np.pi, N, endpoint=False)
    if plot:
        wplot(x, y)
    else:
        return y, x


def xcos(f, dc=0, xmin=0, xmax=1, N=3, phi=0, ghost=0.2, red=True):
    y, x = cos(f=f, dc=dc, xmin=xmin, xmax=xmax, N=N, phi=phi, plot=False)
    wplot(x, y)
    if red:
        _ = [plt.axvline(t, color='r', linewidth=.6) for t in x]

    if ghost:
        x = np.linspace(xmin, xmax, f * 40, endpoint=False)
        y = np.cos(f * 2 * np.pi * x + phi) + dc
        plt.plot(x, y, color=(0, 0, 0, ghost))
    plt.show()

def ximp(N, i, real=False, polar=False, show_x=False, v=1, unwrap=False,
         get=False):
    """Impulse(s); form e.g. square wave"""
    if not isinstance(i, (list, tuple, np.ndarray)):
        i = [i]
    i = [(idx if idx >= 0 else N + idx) for idx in i]
    x = np.array([(v if n in i else 0) for n in range(N)])
    coef = np.fft.rfft(x) if real else np.fft.fft(x)

    if show_x:
        plt.plot(x)
        plt.show()
    if polar:
        theta, r = complex_to_polar(coef)
        wplot(r, show=0)
        theta = np.unwrap(theta) if unwrap else theta
        wplot(theta, show=1)
        if get:
            return theta, r, x
    else:
        wplot(coef.real, size=0, show=0)
        wplot(coef.imag, size=0, show=1)
        if get:
            return coef, x

def normal(mu=0, sigma=1, N=128, xlims=(-6, 6)):
    x = np.linspace(*xlims, N)
    return np.exp(-.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


def chirp(N, f=2, max=2 * np.pi):
    t = np.linspace(0, max, N, endpoint=False)
    return np.cos(t ** f)


def cchirp(fs=int(5e7), tau=5e-5, bw=int(1e7)):
    alpha = bw / tau
    t = np.arange(0, tau, 1 / fs)
    # return np.cos(np.pi * alpha * t ** 2)
    return np.exp(1j * np.pi * alpha * t ** 2)


def window(name, N, scipy_c=False):
    t = np.linspace(0, 1, N, False)
    if name == 'hamming':
        return .54 - .46 * np.cos(2 * np.pi * t)
    elif name == 'blackman':
        if scipy_c:
            a0, a1, a2 = 0.42, 0.5, 0.08
        else:
            a0, a1, a2 = 7938 / 18608, 9240 / 18608, 1430 / 18608
        return a0 - a1 * np.cos(2 * np.pi * t) + a2 * np.cos(4 * np.pi * t)
    else:
        return np.ones(N)
#%%# PLOTS ####################################################################
def _plot(coef=None, x=None, i=0, get=False):
    if coef is None:
        if i == 0:
            coef = dft(x)
        elif i == 1:
            coef = np.fft.fft(x)
        else:
            coef = np.fft.rfft(x)
    theta, r = complex_to_polar(coef)

    if get:
        return coef, theta, r
    plt.plot(r); plt.show()
    plt.plot(theta)


def polar(theta, r, neg_r=False, **kwargs):
    if not neg_r:
        theta += np.pi * (np.sign(r) < 0)
        r = np.abs(r)
    plt.polar(theta, r, **kwargs)
    plt.gcf().set_size_inches(10, 10)
    # plt.gca().set_rlim(0, max(np.abs(plt.gca().get_ylim())))
    if not neg_r:
        plt.gca().set_rlim(0, 1)
    else:
        plt.gca().set_rlim(-1, 1)


def plot(*args, **kwargs):
    plt.plot(*args, **kwargs)
    plt.gcf().set_size_inches(8, 8)


def cos_cartesian(f_signal, f_winding, N=120, plott=True):
    x = np.linspace(0, 2 * np.pi * f_signal, N) * 1j
    _cos = (np.exp(x) + np.exp(-x)) / 2

    exp = np.exp(np.linspace(0, 2 * np.pi * f_winding, N) * 1j)
    out = _cos * exp  # DFT integrand

    if plott:
        plt.plot(_cos); plt.show()
        plot(np.real(out), np.imag(out)); plt.show()
    return out, _cos, exp


def plot_imre_rfft(x):
    coef = np.fft.rfft(x)
    plt.plot(np.real(coef))
    plt.plot(np.imag(coef))
    plt.show()


def wplot(x, y=None, show=False, ax_equal=False, complex=0, w=1, h=1, **kw):
    if y is None:
        if complex:
            plt.plot(x.real, **kw)
            plt.plot(x.imag, **kw)
        else:
            plt.plot(x, **kw)
    else:
        if complex:
            plt.plot(x, y.real, **kw)
            plt.plot(x, y.imag, **kw)
        else:
            plt.plot(x, y.real, **kw)
    _scale_plot(plt.gcf(), plt.gca(), show=show, ax_equal=ax_equal, w=w, h=h)


def wscat(x, y=None, show=False, ax_equal=False, s=18, w=None, h=None, **kw):
    if y is None:
        plt.scatter(np.arange(len(x)), x, s=s, **kw)
    else:
        plt.scatter(x, y, s=s, **kw)
    _scale_plot(plt.gcf(), plt.gca(), show=show, ax_equal=ax_equal, w=w, h=h)


def mag_zoom(r, h=10, x0=0, show=True):
    N = len(r)
    N2 = N // 2
    a, b = N2 + x0 - h, N2 + x0 + h + 1
    k = np.linspace(-N2, N2 - 1, N)[a:b]
    plt.plot(k, r[a:b])
    _scale_plot(plt.gcf(), plt.gca(), show=show)


def imshow(data, norm=None, complex=None, abs=0, show=1, w=1, h=1,
           ridge=0, yticks=None, ticks=1, **kw):
    kw['interpolation'] = kw.get('interpolation', 'none')
    if norm is None:
        mx = np.max(np.abs(data))
        vmin, vmax = -mx, mx
    else:
        vmin, vmax = norm

    if abs:
        kw['cmap'] = kw.get('cmap', 'bone')
        plt.imshow(np.abs(data), vmin=0, vmax=vmax, **kw)
    else:
        kw['cmap'] = kw.get('cmap', 'bwr')
        if (complex is None and np.sum(np.abs(np.imag(data))) < 1e-8) or (
                complex is False):
            plt.imshow(np.real(data), vmin=vmin, vmax=vmax, **kw)
        else:
            fig, axes = plt.subplots(1, 2)
            axes[0].imshow(data.real, vmin=vmin, vmax=vmax, **kw)
            axes[1].imshow(data.imag, vmin=vmin, vmax=vmax, **kw)
            plt.subplots_adjust(left=0, right=1, bottom=0, top=1,
                                wspace=0, hspace=0)
    plt.gcf().set_size_inches(14 * w, 8 * h)

    if ridge:
        data_mx = np.where(np.abs(data) == np.abs(data).max(axis=0))
        plt.scatter(data_mx[1], data_mx[0], color='r', s=4)
    if yticks is not None:
        plt.gca().set_yticklabels(yticks)
    if not ticks:
        plt.xticks([])
        plt.yticks([])
    if show:
        plt.show()


def whist(x, bins=500, show=0, stats=0):
    x = np.asarray(x)
    _ = plt.hist(x.ravel(), bins=bins)
    if show:
        plt.show()
    if stats:
        mu, std, mn, mx = (x.mean(), x.std(), x.min(), x.max())
        print("(mean, std, min, max) = ({}, {}, {}, {})".format(
            *_fmt(mu, std, mn, mx)))
        return mu, std, mn, mx

def vhline(n, kind='v', color='r', linewidth=1, show=0):
    if kind == 'v':
        plt.axvline(n, color=color, linewidth=linewidth)
    else:
        plt.axhline(n, color=color, linewidth=linewidth)
    if show:
        plt.show()

def _fmt(*nums):
    return [(("%.3e" % n) if (abs(n) > 1e3 or abs(n) < 1e-3) else
             ("%.3f" % n)) for n in nums]


def _scale_plot(fig, ax, show=False, ax_equal=False, w=None, h=None):
    xmin, xmax = ax.get_xlim()
    rng = xmax - xmin
    ax.set_xlim(xmin + .018 * rng, xmax - .018 * rng)
    if w or h:
        fig.set_size_inches(14*(w or 1), 8*(h or 1))
    if ax_equal:
        yabsmax = max(np.abs([*ax.get_ylim()]))
        mx = max(yabsmax, max(np.abs([xmin, xmax])))
        ax.set_xlim(-mx, mx)
        ax.set_ylim(-mx, mx)
        fig.set_size_inches(8*(w or 1), 8*(h or 1))
    if show:
        plt.show()


def annot(i, scale):
    plt.annotate("scale=%.2f\nindex=%d" % (scale, i), weight='bold',
                 fontsize=14, xycoords='axes fraction', xy=(.85, .9))

def plotenergy(x, axis=1, **kw):
    wplot(np.sum(np.abs(x) ** 2, axis=axis), **kw)

def plot_opt_vline(x, lims=None, **kw):
    lims = lims or [0, None]
    idx = np.argmin(np.abs(np.diff(x))[lims[0]:lims[1]]) + lims[0]
    plt.axvline(idx, **kw)
    return idx

#%%# TRANSFORMS ###############################################################
def dft(x):
    N = len(x)
    reX = np.zeros(N // 2 + 1)
    imX = np.zeros(N // 2 + 1)

    for k in range(N // 2 + 1):
        for n in range(N):
            reX[k] += x[n] * np.cos(2 * np.pi * k * n / N)
            imX[k] += x[n] * np.sin(2 * np.pi * k * n / N)
    return reX - imX * 1j


def idft(coef, N=None):
    N = N or len(coef) * 2 - 1
    reX, imX = np.real(coef), np.imag(coef)

    reX = reX / (N / 2)
    reX[0] /= 2
    reX[N // 2] /= 2
    imX = - imX / (N / 2)

    x = np.zeros(N)
    for i in range(N):
        for k, (re, im) in enumerate(zip(reX, imX)):
            x[i] += (re * np.cos(2 * np.pi * k * i / N) +
                     im * np.sin(2 * np.pi * k * i / N))
    return x


def allfft(x):
    coef = np.fft.fft(x)
    return coef, np.angle(coef), np.abs(coef)
#%%# CONVERSION ###############################################################
def c2r_coef(ccoef):
    dc = ccoef[0]
    N = len(ccoef)
    if N % 2 == 0:
        f_pos = ccoef[1:N // 2]
        f_neg = ccoef[N // 2 + 1:][::-1]
        f_nyq = ccoef[N // 2]
    else:
        f_pos = ccoef[1:N // 2 + 1]
        f_neg = ccoef[N // 2 + 1:][::-1]
    f_pn = np.array(f_pos + np.conj(f_neg)) / 2

    rcoef = ([dc, *list(f_pn), f_nyq] if N % 2 == 0 else
             [dc, *list(f_pn)])
    return np.array(rcoef)


def rad(deg):
    if isinstance(deg, (list, tuple, np.ndarray)):
        return np.array([x / 180 * np.pi for x in deg])
    return deg / 180 * np.pi


# def complex_to_polar(data):
#     theta, r = [], []
#     for x in data:
#         R, I = np.real(x), np.imag(x)
#         r.append(np.sqrt(R ** 2 + I ** 2))
#         if R == 0:
#             if I >= 0:
#                 theta.append(np.pi / 2)
#             else:
#                 theta.append(3 * np.pi / 2)
#         else:
#             th = np.arctan(abs(I / R))
#             if I >= 0 and R > 0:
#                 theta.append(th)
#             elif I >= 0 and R < 0:
#                 theta.append(np.pi - th)
#             elif I <= 0 and R > 0:
#                 theta.append(2 * np.pi - th)
#             elif I <= 0 and R < 0:
#                 theta.append(np.pi + th)
#     assert all(th >= 0 for th in theta)
#     assert all(_r >= 0 for _r in r)

#     return np.array(theta), np.array(r)


def polar_to_complex(theta, r):
    RE, IM = [], []
    for th, R in zip(theta, r):
        re = R * abs(np.cos(th))
        im = R * abs(np.sin(th))
        if np.pi / 2 <= th < np.pi:
            re *= -1
        elif np.pi <= th < 3 * np.pi / 2:
            im *= -1
            re *= -1
        elif 3 * np.pi / 2 <= th < 2 * np.pi:
            im *= -1
        RE.append(re)
        IM.append(im)
    return np.array(RE), np.array(IM)


def maxima(x, th=1e-3):
    return np.where(abs(x - x.max()) < th)[0]


def l1(x, axis=-1):
    return np.sum(np.abs(x), axis=axis)

def l2(x, axis=-1):
    return np.sqrt(np.sum(np.abs(x ** 2), axis=axis))
