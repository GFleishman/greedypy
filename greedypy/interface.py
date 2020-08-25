#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GreedyPy - greedy weak diffeomorphic registration in python

Copyright: Greg M. Fleishman
Began: November 2019
"""

import argparse
from argparse import RawTextHelpFormatter

# VERSION INFORMATION
VERSION = 'GreedyPy - Version: 0.1'

# DESCRIPTION
DESCRIPTION = """
~~~***~~**~*         GreedyPy         *~**~~***~~~
Greedy weak diffeomorphic registration in python

Quick deformable registration algorithm

----    ---    ---    ----    ---    ---    ----
"""

# TODO: add flag to automatically compute metric mask based on initial
#       transform, i.e. for regions in the moving image that were
#       interpolated from off-grid, set metric gradient to 0

# ARGUMENTS
ARGUMENTS = {
'fixed':'the fixed image',
'moving':'the moving image',
'output':'output path for the transform',
'iterations':'iterations per subsampling level, example: 100x50x25',
'--lcc_radius':'voxel radius of window used by LCC image match; default 8',
'--mask':'region of fixed image that should be matched; not implemented yet',
'--auto_mask':'list of intensity values to ignore in the moving image, separate with spaces',
'--field_regularizer':'AxBxCxD for field metric (A*divgrad + B*graddiv + C)^D; default 1x0x1x2',
'--grad_regularizer':'AxBxCxD for gradient metric (A*divgrad + B*graddiv + C)^D; default 3x0x1x2',
'--gradient_step':'initial gradient descent step size; default 1.0',
'--optimization_tolerance':'factor by which energy may *increase* between iterations; default .0005',
'--initial_transform':'transform applied to moving image before optimization',
'--precision':'single or double precision; default single',
'--n5_fixed_path':'if using n5 format, path within fixed dataset',
'--n5_moving_path':'if using n5 format, path within moving dataset',
'--warped_image':'write the warped moving image',
'--final_lcc':'write image of final lcc metric to given path',
'--compose_output_with_it':'compose the output with the initial transform',
'--inverse':'writes inverse transform to given path'
}

# OPTIONS
OPTIONS = {a:{'help':ARGUMENTS[a]} for a in ARGUMENTS.keys()}
OPTIONS['--lcc_radius'] = {**OPTIONS['--lcc_radius'], 'default':'8'}
OPTIONS['--auto_mask'] = {**OPTIONS['--auto_mask'], 'nargs':'+'}
OPTIONS['--field_regularizer'] = {**OPTIONS['--field_regularizer'], 'default':'1x0x1x2'}
OPTIONS['--grad_regularizer'] = {**OPTIONS['--grad_regularizer'], 'default':'3x0x1x2'}
OPTIONS['--gradient_step'] = {**OPTIONS['--gradient_step'], 'default':'1.0'}
OPTIONS['--optimization_tolerance'] = {**OPTIONS['--optimization_tolerance'], 'default':'.0005'}
OPTIONS['--precision'] = {**OPTIONS['--precision'], 'default':'single'}
OPTIONS['--n5_fixed_path'] = {**OPTIONS['--n5_fixed_path'], 'nargs':2}
OPTIONS['--n5_moving_path'] = {**OPTIONS['--n5_moving_path'], 'nargs':2}
OPTIONS['--compose_output_with_it'] = {**OPTIONS['--compose_output_with_it'], 'action':'store_true'}
OPTIONS['--compose_output_with_it'] = {**OPTIONS['--compose_output_with_it'], 'default':False}


# BUILD PARSER
parser = argparse.ArgumentParser(description=DESCRIPTION,
	                             formatter_class=RawTextHelpFormatter)
for arg in ARGUMENTS.keys():
	parser.add_argument(arg, **OPTIONS[arg])
