#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GreedyPy - greedy weak diffeomorphic registration in python

Copyright: Greg M. Fleishman
Began: November 2019
"""

import numpy as np
from os.path import splitext, abspath, isdir


def parse_n5_slice(slice_string):

    sss = slice_string.split('x')
    sss = [x.split(':') for x in sss]
    sss = [[None if x == '' else int(x) for x in xx] for xx in sss]
    return [slice(s[0], s[1], s[2]) for s in sss]




def read_image(path, dtype, n5_path=None, n5_slice=None):
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
    elif isdir(path):
        import z5py, json
        img = z5py.File(path, use_zarr_format=False)
        slices = parse_n5_slice(n5_slice)
        if len(slices) == 3:
            img_data = img[n5_path][slices[0], slices[1], slices[2]]
        elif len(slices) == 2:
            img_data = img[n5_path][slices[0], slices[1]]
        with open(path + n5_path + '/attributes.json') as atts:
            atts = json.load(atts)
        img_vox = np.absolute(np.array(atts['pixelResolution']) *
                              np.array(atts['downsamplingFactors']))
        return img_data, img_vox, atts




def write_field(field, path, fixed):
    """Write estimated field"""

    x = splitext(path)[1]
    ext = x if x != '.gz' else splitext(splitext(path)[0])[1]
    if ext == '.nii':
        import nibabel
        img = nibabel.load(abspath(fixed))
        aff = img.affine
        img = nibabel.Nifti1Image(field, aff)
        nibabel.save(img, path)
    elif ext == '.nrrd':
        import nrrd
        img, meta = nrrd.read(fixed)
        # TODO: need to create better metadata
        nrrd.write(path, field)