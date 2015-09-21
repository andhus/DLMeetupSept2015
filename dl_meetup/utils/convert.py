from __future__ import division, print_function

"""
The following functions are useful for converting data from one common format
to another when working with RNNs/time series data.

Notion:
   S - sample dimension / "n_samples"
   T - time dimension / "n_time_steps"
   F - feature dimension / "n_features"
"""


def st1_of_st(st):
    """
    Parameters
    ----------
    st: ndarray(shape=(S, T))

    Returns
    -------
    st1 : ndarray(shape=(S, T, 1))
    """
    return st.reshape(st.shape + (1,))


def ts1_of_st(st):
    """
    Parameters
    ----------
    st: ndarray(shape=(S, T))

    Returns
    -------
    ts1 : ndarray(shape=(T, S, 1))
    """
    return st1_of_st(st).swapaxes(0, 1)


def st_of_t(t):
    """
    Parameters
    ----------
    t: ndarray(shape=(T,))

    Returns
    -------
    st : ndarray(shape=(1, T))
    """
    return t.reshape((1, -1))


def tsf_of_stf(stn):
    """
    Parameters
    ----------
    stn: ndarray(shape=(S, T, F))

    Returns
    -------
    tsn : ndarray(shape=(T, S, F))
    """
    return stn.swapaxes(0, 1)