#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Unit Test for Maxwell equation solver
"""

import os
import sys
import pytest
import json
import glob
import itertools
import subprocess

import numpy as np


JSONSTR_TEMPLATE = \
"""
{{
    "application": {{
        "rebuild": {{
            "interval": 1,
            "loglevel": 0
        }}
    }},
    "diagnostic": {{
        "interval" : 10,
        "prefix" : "{filename}_"
    }},
    "parameter": {{
        "Nx" : {Nx},
        "Ny" : {Ny},
        "Nz" : {Nz},
        "Cx" : {Cx},
        "Cy" : {Cy},
        "Cz" : {Cz},
        "delt" : 0.5,
        "delh" : 1.0,
        "cc"   : 1.0,
        "kdir" : {kdir}
    }}
}}
"""

FILENAME_TEMPLATE = "{prefix}-{Cx:03d}-{Cy:03}-{Cz:03d}-kdir{kdir:01d}"


def run_cmdline(cmdline, cleanup=True, **kwargs):
    # convert everything to string for passing to subprocess
    cmd = [str(c) for c in cmdline]

    # run
    status = True
    try:
        print('Running command : ', ' '.join(cmd))
        subprocess.run(cmd, capture_output=True, check=True)
    except subprocess.CalledProcessError as e:
        status = False
        print(e.cmd)
        print(e.returncode)
        print(e.output)
        print(e.stdout)
        print(e.stderr)

    # cleanup if succeeded
    if status and cleanup:
        # remove files
        files = list()
        files.extend(glob.glob('main.out_PE*.std*'))
        files.extend(glob.glob('{filename}_*.data'.format(**kwargs)))
        files.extend(glob.glob('{filename}_*.json'.format(**kwargs)))
        files.append('config-{filename}.json'.format(**kwargs))
        for f in files:
            os.remove(f)
    return status


def prepare_for_run(**kwargs):
    parameters = kwargs.copy()
    # generate JSON configuration file
    jsonstr = JSONSTR_TEMPLATE.format(**parameters)
    config = 'config-' + parameters['filename'] + '.json'
    with open(config, 'w') as fp:
        json.dump(json.loads(jsonstr), fp, indent=4)
    # generate command-line for subprocess
    cmdline = ('mpiexec', '-n', parameters['nproc'], './main.out',
               '--tmax', parameters['tmax'],
               '--config', config)
    return config, cmdline


def generate_chunk_patterns(numchunk):
    if numchunk == 1:
        return [(1, 1, 1)]
    if numchunk == 2:
        chunk = list()
        for p in [(2, 1, 1)]:
            chunk.extend(list(set(itertools.permutations(p))))
        return chunk
    if numchunk == 4:
        chunk = list()
        for p in [(4, 1, 1), (2, 2, 1)]:
            chunk.extend(list(set(itertools.permutations(p))))
        return chunk
    if numchunk == 8:
        chunk = list()
        for p in [(8, 1, 1), (4, 2, 1), (2, 2, 2)]:
            chunk.extend(list(set(itertools.permutations(p))))
        return chunk

    return None


def do_test_returncode(nproc, kdir, Nx, Ny, Nz, Cx, Cy, Cz):
    tmax = 5.0
    kwargs = dict()
    kwargs['prefix'] = 'testing'
    kwargs['nproc'] = nproc
    kwargs['Nx'] = Nx
    kwargs['Ny'] = Ny
    kwargs['Nz'] = Nz
    kwargs['Cx'] = Cx
    kwargs['Cy'] = Cy
    kwargs['Cz'] = Cz
    kwargs['kdir'] = kdir
    kwargs['tmax'] = tmax
    kwargs['filename'] = FILENAME_TEMPLATE.format(**kwargs)
    config, cmdline = prepare_for_run(**kwargs)
    assert run_cmdline(cmdline, **kwargs) == True


@pytest.mark.parametrize("nproc", [1])
@pytest.mark.parametrize("Cx, Cy, Cz", generate_chunk_patterns(1))
def test_returncode_numchunk1(nproc, Cx, Cy, Cz):
    kdir = 0
    Nz, Ny, Nx = 32, 32, 32
    do_test_returncode(nproc, kdir, Nx, Ny, Nz, Cx, Cy, Cz)


@pytest.mark.parametrize("nproc", [1, 2])
@pytest.mark.parametrize("Cx, Cy, Cz", generate_chunk_patterns(2))
def test_returncode_numchunk2(nproc, Cx, Cy, Cz):
    kdir = 0
    Nz, Ny, Nx = 32, 32, 32
    do_test_returncode(nproc, kdir, Nx, Ny, Nz, Cx, Cy, Cz)


@pytest.mark.parametrize("nproc", [1, 2, 4])
@pytest.mark.parametrize("Cx, Cy, Cz", generate_chunk_patterns(4))
def test_returncode_numchunk4(nproc, Cx, Cy, Cz):
    kdir = 0
    Nz, Ny, Nx = 32, 32, 32
    do_test_returncode(nproc, kdir, Nx, Ny, Nz, Cx, Cy, Cz)


@pytest.mark.parametrize("nproc", [1, 2, 4, 8])
@pytest.mark.parametrize("Cx, Cy, Cz", generate_chunk_patterns(8))
def test_returncode_numchunk8(nproc, Cx, Cy, Cz):
    kdir = 0
    Nz, Ny, Nx = 32, 32, 32
    do_test_returncode(nproc, kdir, Nx, Ny, Nz, Cx, Cy, Cz)


@pytest.mark.parametrize("nproc", [1, 2, 4, 8])
@pytest.mark.parametrize(
    "Nx, Ny, Nz",
    [
        (32, 8, 8), (8, 32, 8), (8, 8, 32),
        (32, 16, 8), (8, 32, 16), (16, 32, 8),
        (32, 32, 8), (8, 32, 32), (32, 8, 32),
    ])
def test_returncode_numgrid(nproc, Nx, Ny, Nz):
    kdir = 0
    Cx, Cy, Cz = 2, 2, 2
    do_test_returncode(nproc, kdir, Nx, Ny, Nz, Cx, Cy, Cz)


if __name__ == '__main__':
    pass
