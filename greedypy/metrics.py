#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GreedyPy - greedy weak diffeomorphic registration in python

Copyright: Greg M. Fleishman
Began: November 2019
"""

import numpy as np
from itertools import product

class local_correlation:


    # TODO: store rad and tolerance in class
    def __init__(self, fixed, moving, rad, tolerance=1e-6):
        """
        """

        s = self
        s.mov_mean = np.mean(moving)  # should recomp after interp, but accurate enough
        s.mov_winsor_min, s.mov_winsor_max = np.percentile(moving, [0.1, 99.9])
        s.fix_mean = np.mean(fixed)
        s.fix_shifted = fixed - s.fix_mean
        s.u_fix_shifted = s._local_means(s.fix_shifted, rad)
        s.u_fix = s.u_fix_shifted + s.fix_mean
        s.v_fix = s._local_means(s.fix_shifted**2, rad) - s.u_fix_shifted**2
        winsor_min, winsor_max = np.percentile(fixed, [0.1, 99.9])
        s.fix_mask = s.v_fix / (winsor_max - winsor_min) < tolerance


    def evaluate(self, fixed, moving, rad, tolerance=1e-6, mean=True):
        """
        """

        s = self
        mov_shifted = moving - s.mov_mean
        u_mov_shifted = s._local_means(mov_shifted, rad)
        u_mov = u_mov_shifted + s.mov_mean
        v_mov = s._local_means(mov_shifted**2, rad) - u_mov_shifted**2
        mov_mask = v_mov / (s.mov_winsor_max - s.mov_winsor_min) < tolerance
        v_fixmov = s._local_means(s.fix_shifted*mov_shifted, rad) - \
                                 s.u_fix_shifted*u_mov_shifted
        with np.errstate(divide='ignore', invalid='ignore'):
            cc = v_fixmov**2 / (s.v_fix * v_mov)
        if mean: cc = -np.mean(cc[~(s.fix_mask + mov_mask)])
        return cc


    def gradient(self, fixed, moving, rad, vox, tolerance=1e-6):
        """
        """

        s = self
        mov_shifted = moving - s.mov_mean
        u_mov_shifted = s._local_means(mov_shifted, rad)
        u_mov = u_mov_shifted + s.mov_mean
        v_mov = s._local_means(mov_shifted**2, rad) - u_mov_shifted**2
        mov_mask = v_mov / (s.mov_winsor_max - s.mov_winsor_min) < tolerance
        v_fixmov = s._local_means(s.fix_shifted*mov_shifted, rad) - \
                                 s.u_fix_shifted*u_mov_shifted

        with np.errstate(divide='ignore', invalid='ignore'):
            cc = v_fixmov**2 / (s.v_fix * v_mov)
            cc = -np.mean(cc[~(s.fix_mask + mov_mask)])
            ccgrad = v_fixmov * ( (moving-u_mov)  * v_fixmov - \
                                  (fixed-s.u_fix) * v_mov ) / (s.v_fix*v_mov**2)
        ccgrad[s.fix_mask + mov_mask] = 0
        grad = np.moveaxis(np.gradient(moving, *vox), 0, -1)
        ccgrad = ccgrad[..., np.newaxis] * np.ascontiguousarray(grad)
        return cc, ccgrad


    def _local_means(self, im, rad):
        """
        """

        # pad array, compute summed area table
        d = len(im.shape)
        im_ = im / (2*rad+1)**d  # more stable to apply denominator first (smaller numbers)
        sat = np.pad(im_, [(rad+1, rad),]*d, mode='reflect')
        # potential stability problems with np.cumsum (sequential adding)
        for i in range(d):
            sat.cumsum(axis=i, out=sat, dtype=np.longdouble)

        # take appropriate vectorized array differences
        # kept track using binary strings and slice objects
        binary_strings = ["".join(x) for x in product("01", repeat=d)]
        so = self._get_slice_object(binary_strings.pop(-1), rad)
        means = np.copy(sat[so])
        for bs in binary_strings:
            so = self._get_slice_object(bs, rad)
            s = (-1)**(d - np.sum( [int(x) for x in bs] ))
            means += s*sat[so]
        return means.astype(im.dtype)


    def _get_slice_object(self, bs, rad):
        """
        """

        so0 = slice(None, -2*rad-1, None)
        so1 = slice(2*rad+1, None, None)
        return tuple([so0 if b == "0" else so1 for b in bs])


class sum_squared_differences:


    def __init__(self):
        """
        """

        None


    def evaluate(self, fixed, moving):
        """
        """

        return np.sum( (fixed - moving)**2 )


    def gradient(self, fixed, moving, vox):
        """
        """

        diff = fixed - moving
        grad = np.moveaxis(np.gradient(moving, vox), 0, -1)
        return diff * np.ascontiguousarray(grad)

