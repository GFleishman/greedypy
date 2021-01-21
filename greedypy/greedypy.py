#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GreedyPy - greedy weak diffeomorphic registration in python

Copyright: Greg M. Fleishman
Began: November 2019
"""

from greedypy import interface
from greedypy import optimizer
from argparse import Namespace


# using command line interface
if __name__ == "__main__":

    args = interface.parser.parse_args()
    optimizer.register(args)


# importing into python code, provide default args
else:

    args = {}
    for k in interface.OPTIONS.keys():
        if 'default' in interface.OPTIONS[k].keys():
            v = interface.OPTIONS[k]['default']
        else:
            v = ''
        if k[:2] == '--':
            k = k[2:]
        args[k] = v


    def check_args():
        for k in args.keys():
            print(k, ': ', args[k])


    def set_fixed(fixed):
        args['fixed'] = fixed

    def set_moving(moving):
        args['moving'] = moving

    def set_output(output):
        args['output'] = output

    def set_iterations(iterations):
        args['iterations'] = iterations

    def set_lcc_radius(lcc_radius):
        args['lcc_radius'] = lcc_radius

    def set_mask(mask):
        args['mask'] = mask

    def set_auto_mask(auto_mask):
        args['auto_mask'] = auto_mask

    def set_field_regularizer(abcd):
        args['field_regularizer'] = abcd

    def set_grad_regularizer(abcd):
        args['grad_regularizer'] = abcd

    def set_gradient_step(step):
        args['gradient_step'] = step

    def set_optimization_tolerance(tolerance):
        args['optimization_tolerance'] = tolerance

    def set_initial_transform(initial_transform):
        args['initial_transform'] = initial_transform

    def set_precision(precision):
        args['precision'] = precision

    def set_n5_fixed_path(n5_subpath, n5_slice):
        args['n5_fixed_path'] = [n5_subpath, n5_slice]

    def set_n5_moving_path(n5_subpath, n5_slice):
        args['n5_moving_path'] = [n5_subpath, n5_slice]

    def set_warped_image(warped_image):
        args['warped_image'] = warped_image

    def set_final_lcc(final_lcc):
        args['final_lcc'] = final_lcc

    def set_compose_output_with_it():
        args['compose_output_with_it'] = True

    def set_inverse(inverse):
        args['inverse'] = inverse


    def register():
        global args
        args = Namespace(**args)
        optimizer.register(args)


