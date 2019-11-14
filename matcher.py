#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GreedyPy - greedy weak diffeomorphic registration in python

Copyright: Greg M. Fleishman
Began: November 2019
"""

import numpy as np

class matcher:


    u_fixed_shifted, u_fixed, v_fixed = None, None, None

    def __init__(self, fixed, moving, rad, tolerance=1e-6):
        s = self
        s.mov_mean = np.mean(moving)
        s.fix_mean = np.mean(fixed)
        s.fix_shifted = fixed - s.fix_mean
        s.u_fix_shifted = s.local_means(s.fix_shifted, rad)
        s.u_fix = s.u_fix_shifted + s.fix_mean
        s.v_fix = s.local_means(s.fix_shifted**2, rad) - s.u_fix_shifted**2
        s.v_fix[s.v_fix < tolerance] = 1


    def lcc(self, fixed, moving, rad, tolerance=1e-6, mean=True):
        """evaluate the lcc image match function"""

        s = self
        mov_shifted = moving - s.mov_mean
        u_mov_shifted = s.local_means(mov_shifted, rad)
        u_mov = u_mov_shifted + s.mov_mean
        v_mov = s.local_means(mov_shifted**2, rad) - u_mov_shifted**2
        v_mov[v_mov < tolerance] = 1
        v_fixmov = s.local_means(s.fix_shifted*mov_shifted, rad) - \
                                 s.u_fix_shifted*u_mov_shifted
        v_fixmov[v_fixmov < tolerance] = 0
        cc = v_fixmov**2 / (s.v_fix * v_mov)
        if mean: cc = -np.mean(cc)
        return cc


    def lcc_grad(self, fixed, moving, rad, vox, tolerance=1e-6):
        """evaluate the lcc image match gradient,
           returns the lcc and the lcc gradient"""

        # TODO: check LCC accuracy against np.corrcoef function
        s = self
        mov_shifted = moving - s.mov_mean
        u_mov_shifted = s.local_means(mov_shifted, rad)
        u_mov = u_mov_shifted + s.mov_mean
        v_mov = s.local_means(mov_shifted**2, rad) - u_mov_shifted**2
        v_mov[v_mov < tolerance] = 1
        v_fixmov = s.local_means(s.fix_shifted*mov_shifted, rad) - \
                                 s.u_fix_shifted*u_mov_shifted
        v_fixmov[v_fixmov < tolerance] = 0

        cc = -np.mean( v_fixmov**2 / (s.v_fix * v_mov) )
        ccgrad = v_fixmov * ( (moving-u_mov)  * v_fixmov - \
                              (fixed-s.u_fix) * v_mov ) / (s.v_fix*v_mov**2)
        grad = np.moveaxis(np.gradient(moving, *vox), 0, -1)
        ccgrad = ccgrad[..., np.newaxis] * np.ascontiguousarray(grad)
        return cc, ccgrad


    def local_means(self, im, rad):
        """compute local mean image using summed area table"""

        # pad array, compute summed area table
        d = len(im.shape)
        im_ = im / (2*rad+1)**d  # more stable to apply denominator first (smaller numbers)
        sat = np.pad(im_, [(rad+1, rad),]*d, mode='constant')
        for i in range(d):
            sat.cumsum(axis=i, out=sat, dtype=np.longdouble)  # potential stability problems (sequential adding)

        # take appropriate vectorized array differences
        # kept track of using binary strings and slice objects
        so = self.get_slice_object([1,]*d, rad)
        means = np.copy(sat[so])
        bs = [0,]*d
        for i in range(2**d - 1):
            so = self.get_slice_object(bs, rad)
            s = (-1)**(d-np.sum(bs))
            means += s*sat[so]
            bs = self.increment_binary_string(bs)
        return means.astype(im.dtype)


    def get_slice_object(self, bs, rad):
        """Helper function for local_means
           construct slice object corresponding to a binary string"""

        so0 = slice(None, -2*rad-1, None)
        so1 = slice(2*rad+1, None, None)
        return [so0 if b == 0 else so1 for b in bs]


    def increment_binary_string(self, bs):
        """Helper function for local_means
           increment a binary string"""

        done = False
        b = len(bs)
        while not done and b > 0:
            if bs[b-1] == 0:
                bs[b-1] = 1
                done = True
            else:
                bs[b-1] = 0
            b -= 1
        return bs


    def ssd(self, fixed, moving):
        """evaluate ssd image match function"""

        return np.sum( (fixed - moving)**2 )


    def ssd_grad(self, fixed, moving, vox):
        """evaluate ssd image match gradient"""

        diff = fixed - moving
        grad = np.moveaxis(np.gradient(moving, vox), 0, -1)
        return diff * np.ascontiguousarray(grad)

