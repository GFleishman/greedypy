#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GreedyPy - greedy weak diffeomorphic registration in python

Copyright: Greg M. Fleishman
Began: November 2019
"""

import interface
from argparse import Namespace
import optimizer


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

    def set_lcc_radius(radius):
        args['lcc_radius'] = lcc_radius

    def set_mask(mask):
        args['mask'] = mask

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


    def register():
        global args
        args = Namespace(**args)
        optimizer.register(args)


