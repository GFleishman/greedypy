#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GreedyPy - greedy weak diffeomorphic registration in python

Copyright: Greg M. Fleishman
Began: November 2019
"""

import numpy as np
from os.path import splitext, abspath, isdir


def ensureArray(reference, dataset_path):
    """
    """

    if not isinstance(reference, np.ndarray):
        if not isinstance(reference, str):
            raise ValueError("image references must be ndarrays or filepaths")
        reference, vox, meta = read_image(reference, dataset_path)[...]  # hdf5 arrays are lazy
    else:
        vox, meta = None, None
    return reference, vox, meta


def parse_n5_slice(slice_string):

    sss = slice_string.split('x')
    sss = [x.split(':') for x in sss]
    sss = [[None if x == '' else int(x) for x in xx] for xx in sss]
    return [slice(s[0], s[1], s[2]) for s in sss]




def read_image(path, dtype, n5_path=None):
    """Read an image and its voxel spacing"""

    x = splitext(path)[1]
    ext = x if x != '.gz' else splitext(splitext(path)[0])[1]
    if ext == '.nii':
        import nibabel
        img = nibabel.load(abspath(path))
        img_meta = img.header
        img_data = img.get_data().squeeze()
        img_data = dtype(img_data)
        img_vox = np.array(img.header.get_zooms()[0:3])
        return img_data, img_vox, img_meta
    elif ext == '.nrrd':
        import nrrd
        img_data, img_meta = nrrd.read(abspath(path))
        img_data = dtype(img_data)
        img_vox = np.diag(img_meta['space directions'].astype(dtype))
        return img_data, img_vox, img_meta
    elif isdir(path) and n5_path is not None:
        import zarr, json
        img = zarr.open(store=zarr.N5Store(path), mode='r')
        slices = parse_n5_slice(n5_path[1])
        if len(slices) == 3:
            img_data = img[n5_path[0]][slices[0], slices[1], slices[2]]
            img_data = np.ascontiguousarray(np.moveaxis(img_data, (0, 2), (2, 0)))
        elif len(slices) == 2:
            img_data = img[n5_path[0]][slices[0], slices[1]]
            img_data = np.ascontiguousarray(np.moveaxis(img_data, 0, -1))
        img_data = img_data.astype(dtype)
        with open(path + n5_path[0] + '/attributes.json') as atts:
            atts = json.load(atts)
        img_vox = np.absolute(np.array(atts['pixelResolution']) *
                              np.array(atts['downsamplingFactors']))
        img_vox = img_vox.astype(dtype)
        return img_data, img_vox, atts




def write_image(field, path):
    """Write estimated field"""

    x = splitext(path)[1]
    ext = x if x != '.gz' else splitext(splitext(path)[0])[1]
    if ext == '.nii':
        import nibabel
        img = nibabel.Nifti1Image(field, np.eye(4))
        nibabel.save(img, abspath(path))
    elif ext == '.nrrd':
        import nrrd
        # TODO: need to create better metadata
        nrrd.write(abspath(path), field)

