import jmkdata

import numpy as np


def test_gappy_interp():
    x0 = np.array([0, 1.2, 2.2, 4.4, 5.8, 8.0, 9.1])
    y0 = np.arange(7)
    xint = np.arange(10)

    expectedGood = np.array([True, True, True, False, False, True, False, False, True, True])
    yint = jmkdata.gappy_interp(xint, x0, y0, maxgap=2.0)
    assert np.all(np.isnan(yint[~expectedGood]))
    assert np.all(~np.isnan(yint[expectedGood]))

    x0 = np.array([0, 2.2, 2.5, 4.4, 5.8, 8.0, 9.1])
    y0 = np.arange(7)
    xint = np.arange(10)

    expectedGood = np.array([True, False, False, True, True, True, False, False, True, True])
    yint = jmkdata.gappy_interp(xint, x0, y0, maxgap=2.0)
    print(yint)
    assert np.all(np.isnan(yint[~expectedGood]))
    assert np.all(~np.isnan(yint[expectedGood]))
