import numpy as np
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

    # figure out which sample each time in time base belongs to:
    x_index = np.searchsorted(x0, xint, side='right')
    x_index = np.clip(x_index, 0, len(x0)-1)

    # figure out the space between sample pairs
    dx = np.concatenate(([0], np.diff(x0)))
    # get the gap size for each timebase data point:
    # get the indices of timebase that are too large:
    index = (dx[x_index] > maxgap)

    # this is fine, except the degenerate case when a timebase point falls
    # directly on a sample time.  In that case we want to keep the data at
    # that point.  So we just choose the other inequality for the index:
    # figure out which sample each time in time base belongs to:
    x_index = np.searchsorted(x0, xint, side='right')
    x_index = np.clip(x_index, 0, len(x0)-1)

    # figure out the space between sample pairs
    dx = np.concatenate(([0], np.diff(x0)))
    # get the gap size for each timebase data point:
    # get the indices of timebase that are too large:
    index = np.logical_and(index, (dx[x_index] > maxgap))
    yint[index] = np.NaN

    return yint
