#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GreedyPy - greedy weak diffeomorphic registration in python

Copyright: Greg M. Fleishman
Began: November 2019
"""

import numpy as np
from scipy.ndimage import map_coordinates

import time


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
        # TODO: storing X and Xit as contiguous arrays with vector dims
        #       first will probably speed things up; should really be done
        #       throughout entire package

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


    def invert(self, vox, dX, exp=2):
        """Compute numerical inverse of given transform"""

        root = self.nth_square_root(vox, dX, exp)
        inv = np.zeros(root.shape, dtype=dX.dtype)
        for i in range(20):
            inv = - self.apply_transform(root, vox, inv)
        for i in range(exp):
            inv = inv + self.apply_transform(inv, vox, inv)
        return inv


    def nth_square_root(self, vox, dX, exp):
        """Compute nth square root of a displacement field"""

        root = np.copy(dX)
        for i in range(int(exp)):
            root = self.square_root(vox, root)
        return root


    def square_root(self, vox, dX):
        """Numerically find the square root of a displacement field"""

        dXroot = np.zeros(dX.shape, dtype=dX.dtype)
        for i in range(5):
            error = dX - dXroot - self.apply_transform(dXroot, vox, dXroot)
            dXroot += 0.5 * error
        return dXroot


    def square_root_grad(self, vox, dX):

        dXroot = np.zeros(dX.shape)
        for i in range(20):
            error = dX - dXroot - self.apply_transform(dXroot, vox, dXroot)
            jac = self.jacobian(dXroot, vox)
            jac = self.apply_transform_jacobian(jac, vox, dXroot)
            jac_error = np.einsum('...ij,...j->...i', jac, error)
            dXroot += 0.5 * (error + jac_error)

            error_norm = np.linalg.norm(error, axis=-1)
            print(error_norm.max(), '\t', np.sum(error_norm), '\t', np.mean(error_norm))
        return dXroot


    def jacobian(self, v, vox):
        """Return Jacobian field of vector field v"""

        sh, d = v.shape[:-1], v.shape[-1]
        jac = np.empty(sh + (d, d))
        for i in range(d):
            grad = np.moveaxis(np.array(np.gradient(v[..., i], *vox)), 0, -1)
            jac[..., i, :] = np.ascontiguousarray(grad)
        return jac


    def apply_transform_jacobian(self, jac, vox, dX):

        ret = np.empty_like(jac)
        for i in range(3):
            ret[..., i] = self.apply_transform(jac[..., i].squeeze(), vox, dX)
        return ret




