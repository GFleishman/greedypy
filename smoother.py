#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GreedyPy - greedy weak diffeomorphic registration in python

Copyright: Greg M. Fleishman
Began: November 2019
"""

import pyfftw
import numpy as np


class smoother:

    def __init__(self, a, b, c, d, vox, sh, dtype):
        s = self
        s.ffter, s.iffter = s.initializeFFTW(sh, dtype)
        s.L, s.K = s.initialize_metric_kernel(a, b, c, d, vox, sh, dtype)


    def initializeFFTW(self, sh, dtype):
        """Initialize the forward and inverse transforms"""

        sh, ax = tuple(sh), list(range(len(sh)))
        inp = pyfftw.empty_aligned(sh, dtype=dtype)
        outp_sh = sh[:-1] + (sh[-1]//2+1,)
        if dtype is np.float32:
            cdtype = np.complex64
        elif dtype is np.float64:
            cdtype = np.complex128
        outp = pyfftw.empty_aligned(outp_sh, dtype=cdtype)
        ffter = pyfftw.FFTW(inp, outp, axes=ax, threads=1)
        iffter = pyfftw.FFTW(outp, inp, axes=ax, direction='FFTW_BACKWARD', threads=1)
        return ffter, iffter


    def initialize_metric_kernel(self, a, b, c, d, vox, sh, dtype):
        """Precompute the metric kernel and inverse"""

        # define some useful ingredients for later
        dim, oa = len(sh), np.ones(sh, dtype=dtype)
        sha = (np.diag(sh) - np.identity(dim) + 1).astype(int)

        # if grad of div term is 0, kernel is a scalar, else a Lin Txm
        if b == 0.0:
            L = oa * c
        else:
            L = np.zeros(sh + (dim, dim), dtype=dtype) + np.identity(dim) * c

        # compute the scalar (or diagonal) term(s) of kernel
        for i in range(dim):
            q = np.fft.fftfreq(sh[i], d=vox[i]).astype(dtype)
            X = a * (1 - np.cos(q*2.0*np.pi))
            X = np.reshape(X, sha[i])*oa
            if b == 0.0:
                L += X
            else:
                for j in range(dim):
                    L[..., j, j] += X
                L[..., i, i] += b*X/a

        # compute off diagonal terms of kernel
        # TODO: all b != 0 code is out of date and unlikely to work
        if b != 0.0:
            for i in range(dim):
                for j in range(i+1, dim):
                    q = np.fft.fftfreq(sh[i], d=vox[i])
                    X = np.sin(q*2.0*np.pi*vox[i])
                    X1 = np.reshape(X, sha[i])*oa
                    q = np.fft.fftfreq(sh[j], d=vox[j])
                    X = np.sin(q*2.0*np.pi*vox[j])
                    X2 = np.reshape(X, sha[j])*oa
                    X = X1*X2*b/(vox[i]*vox[j])
                    L[..., i, j] = X
                    L[..., j, i] = X

        # I only need half the coefficients (because we're using rfft)
        # compute and store the inverse kernel for regularization
        if b == 0.0:
            L = L[..., :sh[-1]//2+1]**d
            K = L**-1.0
            L = L[..., np.newaxis]
            K = K[..., np.newaxis]
        else:
            L = L[..., :sh[-1]//2+1, :, :]
            cp = np.copy(L)
            for i in range(int(d-1)):
                L = np.einsum('...ij,...jk->...ik', L, cp)
            K = self.gu_pinv(L)

        return L, K


    def gu_pinv(self, a, rcond=1e-15):
        """Return the pseudo-inverse of matrices at every voxel"""

        a = np.asarray(a)
        swap = np.arange(a.ndim)
        swap[[-2, -1]] = swap[[-1, -2]]
        u, s, v = np.linalg.svd(a)
        cutoff = np.maximum.reduce(s, axis=-1, keepdims=True) * rcond
        mask = s > cutoff
        s[mask] = 1. / s[mask]
        s[~mask] = 0
        return np.einsum('...uv,...vw->...uw',
                         np.transpose(v, swap) * s[..., None, :],
                         np.transpose(u, swap))


    def smooth(self, field):
        s = self
        if field.shape[-1] in [1, 2, 3]:
            return s.ifft( s.K * s.fft(field), field.shape )
        else:
            return s.ifft( s.K.squeeze() * s.fft(field), field.shape)


    def fft(self, f):
        """Return the DFT of the real valued vector field f"""

        if f.shape[-1] not in [1, 2, 3]:
            f = f[..., np.newaxis]
        sh, d = f.shape[:-1], f.shape[-1]
        if f.dtype == np.float32:
            cdtype = np.complex64
        elif f.dtype == np.float64:
            cdtype = np.complex128
        F = np.empty(sh[:-1] + (sh[-1]//2+1, d), dtype=cdtype)
        for i in range(d):
            F[..., i] = self.ffter(f[..., i])
        return F.squeeze()


    def ifft(self, F, sh):
        """Return the iDFT of the vector field F"""

        if sh[-1] not in [1, 2, 3]:
            sh += (1,)
            F = F[..., np.newaxis]
        if F.dtype == np.complex64:
            dtype = np.float32
        elif F.dtype == np.complex128:
            dtype = np.float64
        f = np.empty(sh, dtype=dtype)
        for i in range(sh[-1]):
                f[..., i] = self.iffter(F[..., i])
        return f.squeeze()


