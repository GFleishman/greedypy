#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GreedyPy - greedy weak diffeomorphic registration in python

Copyright: Greg M. Fleishman
Began: November 2019
"""

import numpy as np
import time
from scipy.ndimage import zoom
from os import makedirs
from os.path import splitext, abspath, dirname
import gc

import greedypy.regularizers as regularizers
import greedypy.metrics as metrics
import greedypy.transformer as transformer
import greedypy.fileio as fileio

# DEBUG
import sys


def record(message, log):
    """
    """

    print(message)
    print(message, file=log)


def initialize_constants(args):
    """Process command line args, read input images/mask
    all constants stored in a constants_container"""

    # constant throughout alignment
    CONS = {}

    # datatype precision
    CONS['dtype'] = np.float32 if args.precision == 'single' else np.float64
    dtype = CONS['dtype']

    # metric parameters
    CONS['lcc_radius'] = int(args.lcc_radius)

    # gradient descent/optimization parameters
    CONS['iterations'] = [int(x) for x in args.iterations.split('x')]
    CONS['gradient_step'] = dtype(args.gradient_step)
    CONS['tolerance'] = dtype(args.optimization_tolerance)

    # regularizer coefficients
    CONS['field_abcd'] = [dtype(x) for x in args.field_regularizer.split('x')]
    CONS['grad_abcd'] = [dtype(x) for x in args.grad_regularizer.split('x')]

    # output directory and log
    CONS['outdir'] = abspath(dirname(args.output))
    CONS['log'] = open(dirname(args.output)+'/greedypy.log', 'w')
    makedirs(CONS['outdir'], exist_ok=True)

    # TODO: generalize to allow file paths or ndarrays
    fixed, fspacing, fmeta = fileio.read_image(args.fixed, CONS['dtype'],
        args.n5_fixed_path)
    moving, mspacing, mmeta = fileio.read_image(args.moving, CONS['dtype'],
        args.n5_moving_path)
    CONS['fixed'] = fixed
    CONS['fixed_meta'] = fmeta
    CONS['moving'] = moving
    CONS['moving_meta'] = mmeta
    CONS['grid'] = fixed.shape
    CONS['spacing'] = fspacing

    if args.initial_transform:
        if splitext(args.initial_transform)[1] == '.mat':
            matrix = np.loadtxt(abspath(args.initial_transform))
            matrix = dtype(matrix)
            CONS['initial_transform'] = matrix

    if args.mask:
        if isinstance(args.mask, str):
            mask, _not_used_, _not_used__ = fileio.read_image(args.mask)
        # TODO: implement mask support

    if args.auto_mask:
        CONS['auto_mask'] = [int(x) for x in args.auto_mask]

    return CONS


def initialize_variables(CONS, phi, level):
    """Resample target transform and initial velocity
    initialize other objects for scale level"""

    # container to hold all variables
    VARS = {}

    # smooth and downsample the images
    aaSmoother = regularizers.differential(1+level, 0, 1, 2,
        CONS['spacing'], CONS['grid'], CONS['dtype'],
    )

    fix_smooth = np.copy(aaSmoother.smooth(CONS['fixed']))
    mov_smooth = np.copy(aaSmoother.smooth(CONS['moving']))

    VARS['fixed'] = zoom(fix_smooth, 1./2**level, mode='wrap')
    VARS['moving'] = zoom(mov_smooth, 1./2**level, mode='wrap')

    # initialize a few odds and ends
    shape = VARS['fixed'].shape
    VARS['spacing'] = CONS['spacing'] * 2**level
    VARS['warped_transform'] = np.empty(shape + (len(shape),), dtype=CONS['dtype'])

    # initialize or resample the deformation
    if phi is None:
        VARS['phi'] = np.zeros(shape + (len(shape),), dtype=CONS['dtype'])
    else:
        zoom_factor = np.array(shape) / np.array(phi.shape[:-1])
        phi_ = [zoom(phi[..., i], zoom_factor, mode='nearest') for i in range(3)]
        VARS['phi'] = np.ascontiguousarray(np.moveaxis(np.array(phi_), 0, -1))

    # initialize the transformer
    if 'initial_transform' in CONS.keys():
        VARS['transformer'] = transformer.transformer(
            shape, VARS['spacing'], dtype=CONS['dtype'],
            initial_transform=CONS['initial_transform'],
        )
    else:
        VARS['transformer'] = transformer.transformer(
            shape, VARS['spacing'], dtype=CONS['dtype'],
        )

    # initialize the smoothers
    VARS['field_smoother'] = regularizers.differential(
        CONS['field_abcd'][0] * 2**level,
        *CONS['field_abcd'][1:], VARS['spacing'], shape, CONS['dtype'])

    VARS['grad_smoother'] = regularizers.differential(
        CONS['grad_abcd'][0] * 2**level,
        *CONS['grad_abcd'][1:], VARS['spacing'], shape, CONS['dtype'])

    # initialize the matcher
    VARS['matcher'] = metrics.local_correlation(
        VARS['fixed'], VARS['moving'], CONS['lcc_radius'],
    )


    # initialzie the mask if necessary
    if 'auto_mask' in CONS.keys():
        mask = np.ones(VARS['moving'].shape, dtype=np.uint8)
        transformed = VARS['transformer'].apply_transform(
            VARS['moving'], np.zeros_like(VARS['phi']), mode='constant',
        )

        for xxx in CONS['auto_mask']:
            mask[transformed == xxx] = 0

        mask = mask[..., None]
        VARS['auto_mask'] = mask

    return VARS



def register(args):

    # initialize constants dictionary, record params and initial energy
    CONS = initialize_constants(args)
    # TODO: include initial transform in this energy calculation
    metric = metrics.local_correlation(
        CONS['fixed'], CONS['moving'], CONS['lcc_radius'],
    )
    energy = metric.evaluate(
        CONS['fixed'], CONS['moving'], CONS['lcc_radius'],
    )
    record(args, CONS['log'])
    record('initial energy: ' + str(energy), CONS['log'])

    # multiscale loop
    level = len(CONS['iterations']) - 1
    start_time = time.perf_counter()
    lowest_phi = 0

    for local_iterations in CONS['iterations']:

        # initialize level
        phi_ = None if level == len(CONS['iterations'])-1 else lowest_phi
        VARS = initialize_variables(CONS, phi_, level)
        init_trans = True if args.initial_transform is not None else False
        iteration, backstep_count, converged = 0, 0, False
        local_step = CONS['gradient_step']
        lowest_energy = 0

        # loop for current level
        while iteration < local_iterations and not converged:
            t0 = time.perf_counter()

            # compute the residual
            warped = VARS['transformer'].apply_transform(
                VARS['moving'], VARS['phi'], mode='constant',
            )
            energy, residual = VARS['matcher'].gradient(
                VARS['fixed'], warped,
                CONS['lcc_radius'], VARS['spacing'],
            )
            residual = VARS['grad_smoother'].smooth(residual)
            max_residual = np.linalg.norm(residual, axis=-1).max()
            residual *= VARS['spacing'].min()/max_residual

            # apply moving image mask to residual
            if 'auto_mask' in VARS.keys():
                residual *= VARS['auto_mask']

            # monitor the optimization
            if energy > (1 - CONS['tolerance']) * lowest_energy:
                VARS['phi'] = np.copy(lowest_phi)
                local_step *= 0.5
                backstep_count += 1
                iteration -= 1
                VARS['field_smoother'] = regularizers.differential(
                    CONS['field_abcd'][0] * 2**level / 4**backstep_count,
                    *CONS['field_abcd'][1:], VARS['spacing'], VARS['fixed'].shape, CONS['dtype'])
                if backstep_count >= max(local_iterations//10, 5): converged = True
            else:
                if energy < lowest_energy:
                    lowest_energy, lowest_phi = energy, np.copy(VARS['phi'])
                    backstep_count = max(0, backstep_count-1)

                # the gradient descent update
                residual *= -local_step
                for i in range(3):
                    VARS['warped_transform'][..., i] = VARS['transformer'].apply_transform(
                        VARS['phi'][..., i], residual, mode='nearest',
                    )
                VARS['phi'] = VARS['warped_transform'] + residual
                VARS['phi'] = VARS['field_smoother'].smooth(VARS['phi'])

                iteration += 1
            # record progress
            message = 'it: ' + str(iteration) + \
                      ', en: ' + str(energy) + \
                      ', time: ' + str(time.perf_counter() - t0) + \
                      ', bsc: ' + str(backstep_count)
            print(message)
            print(message, file=CONS['log'])
        level -= 1



    message = 'total optimization time: ' + str(time.perf_counter() - start_time)
    print(message)
    print(message, file=CONS['log'])



    # explicitly free some memory
    del VARS['warped_transform'], VARS['phi']
    del VARS['grad_smoother'], VARS['field_smoother']


    init_trans = True if args.initial_transform is not None else False
    if args.final_lcc is not None or \
       args.warped_image is not None:
        warped = VARS['transformer'].apply_transform(
            CONS['moving'], lowest_phi, mode='constant',
        )


    # write the warped image
    if args.warped_image is not None:
        fileio.write_image(warped, args.warped_image)


    # write the final lcc
    if args.final_lcc is not None:
        final_lcc = VARS['matcher'].evaluate(
            CONS['fixed'], warped, CONS['lcc_radius'], mean=False,
        )
        fileio.write_image(final_lcc, args.final_lcc)
        del warped, final_lcc


    # write the deformation field
    output = lowest_phi
    if args.compose_output_with_it:
        output = lowest_phi + VARS['transformer'].X
        output = output - VARS['transformer']._get_position_array(output.dtype)
    fileio.write_image(output, args.output)
    del VARS['fixed'], VARS['moving'], CONS['fixed'], CONS['moving']
    gc.collect()


    # write the inverse
    if args.inverse is not None:
        inverse = VARS['transformer'].invert(lowest_phi)
        if args.compose_output_with_it:
            matrix = np.array([CONS['initial_transform'][0],
                               CONS['initial_transform'][1],
                               CONS['initial_transform'][2],
                               [0, 0, 0, 1]])
            inv_matrix = np.linalg.inv(matrix)[:-1]
            inv_trans = transformer.transformer(
                inverse.shape[:-1], CONS['spacing'], dtype=np.float32,
                initial_transform=inv_matrix,
            )
            inverse = inverse + inv_trans.X
            inverse = inverse - inv_trans._get_position_array(inverse.dtype)
        fileio.write_image(inverse, args.inverse)



