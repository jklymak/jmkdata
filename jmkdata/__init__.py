import numpy as np
import scipy.signal as signal
# import xarray as xr


def gappy_interp(xint, x0, y0, *, maxgap=None, **kwargs):
    """
    Interpolate as nuumpy.interp, but fill np.NaN is gaps of x0 that are
    greater than *maxgap*.

    xint : array-like
        The x-coordinates at which to evaluate the interpolated values.
    x0 : 1-D sequence of floats
        The x-coordinates of the data points, must be increasing if argument
        period is not specified. Otherwise, xp is internally sorted after
        normalizing the periodic boundaries with x0 = x0 % period.
    y0 : 1-D sequence of float or complex
        The y-coordinates of the data points, same length as x0.
    maxgap : float
        maximum gap size in xint to interpolate over.  Data between gaps is
        filled with NaN.
    **kwargs :
        Passed to `numpy.interp`.

    """

    yint = np.interp(xint, x0, y0, **kwargs)

    # figure out which x0 each xint belongs to:
    x_index = np.searchsorted(x0, xint, side='right')
    x_index = np.clip(x_index, 0, len(x0)-1)

    # figure out the space between sample pairs
    dx = np.concatenate(([0], np.diff(x0)))
    # get the gap size for each xint data point:
    # get the indices of xint that are too large:
    index = (dx[x_index] > maxgap)

    # this is fine, except the degenerate case when a xint point falls
    # directly on a x0 value.  In that case we want to keep the data at
    # that point.  So we just choose the other inequality for the index:
    index = ~np.logical_or((dx[x_index] <= maxgap),(np.isin(xint, x0)))

    # set interpolated values where xint is inside a big gap to NaN:
    yint[index] = np.NaN

    return yint


def gappy_fill(y, *, maxgap=None, axis=1):
    """
    Fill across any gaps smaller than maxgap in number of samples
    """

    if len(y.shape) == 1:
        y = y[np.newaxis, :]
        axis = 1
        squeeze = True
    else:
        squeeze = False
    if axis == 0:
        return gappy_fill(y.T, maxgap=maxgap).T

    ind = np.arange(y.shape[1])
    out = np.zeros_like(y) * np.NaN
    for i in range(y.shape[0]):
        good = np.where(np.isfinite(y[i, :]))[0]
        if len(good) > 2:
            out[i, :] = gappy_interp(ind, ind[good],
                                 y[i, good], maxgap=maxgap)
    if squeeze:
        out = out.flatten()

    return out

def get_n_blocks(nfft, N):
    for m in range(2, 500):
        noverlap = np.ceil(- (N - m * nfft) / (m - 1))
        if noverlap > 0 and noverlap >= nfft / 2:
            total = m * nfft - (m-1) * noverlap
            if total == nfft:
                noverlap = 0
            return int(noverlap), int(total), m


def gappy_psd(x, *, minfft=32, gapfrac=1/6, fs=1, nfft0=None, axl=None, fftstride=4):
    if nfft0 is None:
        nfft = len(x) / 2
    else:
        nfft = nfft0
    dat = gappy_fill(x, maxgap=int(nfft*gapfrac))

    # for the first one, no gaps can be remaining, except maybe at ends...
    f, p = signal.welch(dat, fs=fs, nperseg=nfft, nfft=nfft, noverlap=nfft/2)
    f0 = f[1:]
    p0 = p[1:]
    if axl:
        axl.loglog(f0, p0)

    for itt in range(50):
        nfft = int(nfft/fftstride)
        dat = x.copy()
        dat = gappy_fill(dat, maxgap=int(nfft * gapfrac))
        # for the first one, no gaps can be remaining, except maybe at ends...
        p = None
        good = np.nonzero(np.isfinite(dat))[0]
        if (len(good) < 10) or (nfft < minfft) or (len(good) <= nfft):
            #print('breaking')
            break
        # find a good block:
        stop = good[0]
        num = 0
        ind = 1
        while ind < len(good):
            start = stop+1
            while (ind < len(good)) and (good[ind] - good[ind-1]) == 1:
                ind += 1
            stop = ind - 1
            if good[stop] - good[start] >= nfft:
                dd = dat[good[start:stop]]
                N = len(dd)
                noverlap, total, nblocks = get_n_blocks(nfft, N)
                dd = dd[:total]
                f, pp = signal.welch(dd, fs=fs, nperseg=nfft,
                                     nfft=nfft, noverlap=noverlap)
                if p is not None:
                    p = p + pp * total
                else:
                    p = pp * total
                num = num+total
            ind += 1
        if p is not None:
            p = p / num
            if axl:
                axl.loglog(f[1:], p[1:])
            df = f[2] - f[1]
            st = 4
            p0[f0 > f[st] - df/2] = 10**np.interp(
                np.log10(f0[f0 > f[st] - df / 2]), np.log10(f[st:]),
                np.log10(p[st:]))

    return f0, p0