# -*- coding: utf-8 -*-
"""
Created on Fri Jul 11 00:29:13 2014

# sparse stft and istft
@author: yingyang
"""

from math import ceil
import numpy as np
from scipy.fftpack import fft, ifft, fftfreq

from mne.utils import logger, verbose


@verbose
def sparse_stft(x, wsize, tstep, active_t_ind, verbose=None):
    """
    temporal sparse version of 
    STFT Short-Term Fourier Transform using a sine window,
    modified from mne-python 0.8 git
    Note that the temporal active set is a union of 
            all active time points for each signal

    The transformation is designed to be a tight frame that can be
    perfectly inverted. It only returns the positive frequencies.

    Parameters
    ----------
    x : 2d array of size n_signals x T
        containing multi-channels signal
    wsize : int
        length of the STFT window in samples (must be a multiple of 4)
    tstep : int
        step between successive windows in samples (must be a multiple of 2,
        a divider of wsize and smaller than wsize/2) 
    active_t_ind: 1d numpy.array
        [n_tstep,] boolean variable, active set, the union of all rows of X
        n_tstep = ceil(T/tstep)
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    X : 3d array of shape [n_signals, wsize / 2 + 1, n_step_active]
        STFT coefficients for positive frequencies with
    Usage
    -----
    X = stft(x, wsize, tstep, active_t_ind)
    """
    if not np.isrealobj(x):
        raise ValueError("x is not a real valued array")
    if x.ndim == 1:
        x = x[None, :]
    n_signals, T = x.shape
    wsize = int(wsize)
    ### Errors and warnings ###
    if wsize % 4:
        raise ValueError('The window length must be a multiple of 4.')
    tstep = int(tstep)
    if (wsize % tstep) or (tstep % 2):
        raise ValueError('The step size must be a multiple of 2 and a '
                         'divider of the window length.')
    if tstep > wsize / 2:
        raise ValueError('The step size must be smaller than half the '
                         'window length.')
    n_step = int(ceil(T / float(tstep)))
    n_freq = wsize // 2 + 1
    logger.info("Number of frequencies: %d" % n_freq)
    logger.info("Number of time steps: %d" % n_step)
    # added by Ying 
    if active_t_ind.size != n_step:
        raise ValueError('the size of the active set must match n_step')
    n_step_active = active_t_ind.sum()
    X = np.zeros((n_signals, n_freq, n_step_active), dtype=np.complex)
    if n_signals == 0:
        return X
    # Defining sine window
    win = np.sin(np.arange(.5, wsize + .5) / wsize * np.pi)
    win2 = win ** 2
    # by Ying, this seems to be a normalization
    swin = np.zeros((n_step - 1) * tstep + wsize)
    for t in range(n_step):
        swin[t * tstep:t * tstep + wsize] += win2
    swin = np.sqrt(wsize * swin)
    # Zero-padding and Pre-processing for edges
    xp = np.zeros((n_signals, wsize + (n_step - 1) * tstep),
                  dtype=x.dtype)
    xp[:, (wsize - tstep) // 2: (wsize - tstep) // 2 + T] = x
    x = xp
    active_t_ind_list = np.nonzero(active_t_ind)[0]
    for t in range(n_step_active):
        t0 = active_t_ind_list[t]
        # Framing
        wwin = win / swin[t0 * tstep: t0 * tstep + wsize]
        frame = x[:, t0 * tstep: t0 * tstep + wsize] * wwin[None, :]
        # FFT
        fframe = fft(frame)
        X[:, :, t] = fframe[:, :n_freq]
    return X
#==============================================================================
def sparse_istft(X, tstep, active_t_ind, Tx = None):
    """
    temporal sparse version
    ISTFT Inverse Short-Term Fourier Transform using a sine window

    Parameters
    ----------
    X : 3d array of shape [n_signals, wsize / 2 + 1, n_step_active ]
        The STFT coefficients for positive frequencies
    tstep : int
        step between successive windows in samples (must be a multiple of 2,
        a divider of wsize and smaller than wsize/2) 
    active_t_ind: 1d numpy.array
        [n_tstep,] boolean variable, active set, the union of all rows of X
        n_tstep = ceil(T/tstep)

    Returns
    -------
    x : [n_signals, T]
        vector containing the inverse STFT signal

    Usage
    -----
    sparse_istft(x, tstep, active_t_ind)
  
    """
 ### Errors and warnings ###
    n_signals, n_win, n_step_active = X.shape
    n_step = active_t_ind.size
    if Tx is None:
        Tx = n_step * tstep
    if (n_win % 2 == 0):
        ValueError('The number of rows of the STFT matrix must be odd.')
    wsize = 2 * (n_win - 1)
    if wsize % tstep:
        raise ValueError('The step size must be a divider of two times the '
                         'number of rows of the STFT matrix minus two.')
    if wsize % 2:
        raise ValueError('The step size must be a multiple of 2.')
    if tstep > wsize / 2:
        raise ValueError('The step size must be smaller than the number of '
                         'rows of the STFT matrix minus one.')
    T = n_step * tstep
    x = np.zeros((n_signals, T + wsize - tstep), dtype=np.float)
    if n_signals == 0:
        return x[:, :Tx]
    ### Computing inverse STFT signal ###
    # Defining sine window
    win = np.sin(np.arange(.5, wsize + .5) / wsize * np.pi)
    # win = win / norm(win);
    # Pre-processing for edges
    swin = np.zeros(T + wsize - tstep, dtype=np.float)
    for t in range(n_step):
        swin[t * tstep:t * tstep + wsize] += win ** 2
    swin = np.sqrt(swin / wsize)

    fframe = np.empty((n_signals, n_win + wsize // 2 - 1), dtype=X.dtype)
    active_t_ind_list = np.nonzero(active_t_ind)[0]
    for t in range(len(active_t_ind_list)):
        # IFFT
        fframe[:, :n_win] = X[:, :, t]
        fframe[:, n_win:] = np.conj(X[:, wsize // 2 - 1: 0: -1, t])
        frame = ifft(fframe)
        t0 = active_t_ind_list[t]
        wwin = win / swin[t0 * tstep:t0 * tstep + wsize]
        # Overlap-add
        x[:, t0 * tstep: t0 * tstep + wsize] += np.real(np.conj(frame) * wwin)
    # Truncation
    x = x[:, (wsize - tstep) // 2: (wsize - tstep) // 2 + T + 1][:, :Tx].copy()
    return x
# =============================================================================
# adiitional funciton sparse_phi, sparse_phiT t
# To Be Added

class sparse_Phi(object):
    ''' Util class for phi, stft
        active_t_ind must match n_steps
    '''
    def __init__(self, wsize, tstep):
        self.wsize = wsize
        self.tstep = tstep
        self.n_freq = wsize // 2 + 1
    def __call__(self,x,active_t_ind):
        z = sparse_stft(x,self.wsize, self.tstep, active_t_ind,
                           verbose = False)
        n_signal = x.shape[0]
        return z.reshape(n_signal, self.n_freq*active_t_ind.sum())
        
class sparse_PhiT(object):
    ''' Util class for phi.T, istft
        z must be first reshaped to [n_signals, n_freq, active_times]
    '''
    def __init__(self, tstep, n_freq, n_step, n_times):
        self.tstep = tstep
        self.n_freq = n_freq
        self.n_step = n_step
        self.n_times = n_times
    
    def __call__(self, z, active_t_ind):
        n_signal = z.shape[0]
        z = z.reshape(n_signal, self.n_freq, -1)
        return sparse_istft(z, self.tstep, active_t_ind, self.n_times)

#  Note: reshape has order = 'C' as default, i.e. the last dimension is filled first
# so as a [n_signals, n_freq* actiive_times], each row is time first, than frequency