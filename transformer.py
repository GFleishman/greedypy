#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GreedyPy - greedy weak diffeomorphic registration in python

Copyright: Greg M. Fleishman
Began: November 2019
"""

import numpy as np
from scipy.ndimage import map_coordinates


class transformer:


    def __init__(self, sh, vox, dtype):
        s = self
        s.X = s.set_position_array(sh, vox, dtype)


    def set_position_array(self, sh, vox, dtype):
        """Return a position array in physical coordinates with shape sh"""
        
        sh, vox = tuple(sh), np.array(vox, dtype=dtype)
        coords = np.array(np.meshgrid(*[range(x) for x in sh], indexing='ij'), dtype=dtype)
        return vox * np.ascontiguousarray(np.moveaxis(coords, 0, -1))


    def set_initial_transform(self, matrix):
        s = self
        mm = matrix[:, :-1]
        tt = matrix[:, -1]
        s.Xit = np.einsum('...ij,...j->...i', mm, s.X) + tt


    def apply_transform(self, img, vox, dX, initial_transform=False, order=1):
        """Return img warped by transform X"""

        # TODO: learn about behavior of map_coordinates w.r.t. memory order)
        if len(img.shape) == len(vox):
            img = img[..., np.newaxis]
        X = self.Xit+dX if initial_transform else self.X+dX
        X *= 1./vox
        ret = np.empty(X.shape[:-1] + (img.shape[-1],), dtype=img.dtype)
        X = np.moveaxis(X, -1, 0)
        for i in range(img.shape[-1]):
            ret[..., i] = map_coordinates(img[..., i], X,
                                          order=order, mode='nearest')
        return ret.squeeze()

