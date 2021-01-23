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


    def __init__(self, sh, vox, initial_transform=None, dtype=np.float32):
        """
        """

        self.sh = tuple(sh)
        self.vox = np.array(vox)
        X = self._get_position_array(dtype)
        if initial_transform is not None:
            if initial_transform.shape == X.shape:
                self.X = X + initial_transform
            else:
                mm = initial_transform[:, :-1]
                tt = initial_transform[:, -1]
                self.X = np.einsum('...ij,...j->...i', mm, X) + tt
        else:
            self.X = X


    def _get_position_array(self, dtype):
        """
        """
        
        coords = np.meshgrid(*[range(x) for x in self.sh], indexing='ij')
        coords = np.array(coords, dtype=dtype)
        return self.vox * np.ascontiguousarray(np.moveaxis(coords, 0, -1))


    def apply_transform(self, img, dX, order=1, mode='nearest'):
        """
        """

        # TODO: storing X as contiguous array with vector dims
        #       first will probably speed things up; should really be done
        #       throughout entire package

        if len(img.shape) == len(self.vox):
            img = img[..., np.newaxis]
        X = (self.X + dX) / self.vox
        X = np.moveaxis(X, -1, 0)
        ret = np.empty(X.shape[1:] + (img.shape[-1],), dtype=img.dtype)
        for i in range(img.shape[-1]):
            ret[..., i] = map_coordinates(img[..., i], X, order=order, mode=mode)
        return ret.squeeze()


    def invert(self, dX, exp=2):
        """
        """

        root = self._nth_square_root(dX, exp)
        inv = np.zeros(root.shape, dtype=dX.dtype)
        for i in range(20):
            inv = - self.apply_transform(root, inv)
        for i in range(exp):
            inv = inv + self.apply_transform(inv, inv)
        return inv


    def _nth_square_root(self, dX, exp):
        """
        """

        root = np.copy(dX)
        for i in range(int(exp)):
            root = self._square_root(root)
        return root


    def _square_root(self, dX):
        """
        """

        dXroot = np.zeros(dX.shape, dtype=dX.dtype)
        for i in range(5):
            error = dX - dXroot - self.apply_transform(dXroot, dXroot)
            dXroot += 0.5 * error
        return dXroot



    # TODO: figure out where these are used... refactor or remove them
    def square_root_grad(self, vox, dX):
        """
        """

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
        """
        """

        sh, d = v.shape[:-1], v.shape[-1]
        jac = np.empty(sh + (d, d))
        for i in range(d):
            grad = np.moveaxis(np.array(np.gradient(v[..., i], *vox)), 0, -1)
            jac[..., i, :] = np.ascontiguousarray(grad)
        return jac


    def apply_transform_jacobian(self, jac, vox, dX):
        """
        """

        ret = np.empty_like(jac)
        for i in range(3):
            ret[..., i] = self.apply_transform(jac[..., i].squeeze(), vox, dX)
        return ret




