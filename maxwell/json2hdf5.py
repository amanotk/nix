#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" JSON to HDF5 converter
"""

import os
import sys
import numpy as np
import json
import h5py

_DEBUG = True
FMT_CHUNKID = '%.8d'

def json2hdf5(jsonfile, hdffile=None, verbose=True):
    if hdffile is None:
        hdffile = os.path.splitext(jsonfile)[0] + '.h5'

    # data directory
    datadir = os.path.dirname(jsonfile)
    if datadir == '':
        datadir = './'

    if verbose:
        print('Processing {} to produce {}'.format(jsonfile, hdffile))

    # read json and store in memory
    with open(jsonfile, 'r') as fp:
        obj = json.load(fp)

    # get meta data
    try:
        meta = obj.get('meta')

        # check endian
        endian = meta.get('endian')
        if endian == 1:          # little endian
            byteorder = '<'
        elif endian == 16777216: # big endian
            byteorder = '>'
        else:
            errmsg = 'unrecognized endian flag: {}'.format(endian)
            report_error(errmsg)

        # check raw data file
        rawfile = os.sep.join([datadir, meta.get('rawfile')])
        if not os.path.exists(rawfile):
            errmsg = 'rawfile {} does not exist'.format(rawfile)
            report_error(errmsg)

        # check array order
        order = meta.get('order', 0)
    except Exception:
        print('ignoring {}'.format(jsonfile))
        return

    #
    # create hdf5
    #
    with h5py.File(hdffile, 'w') as h5fp:
        pass

    # create dataset
    group = ''
    add_dataset(obj, hdffile, rawfile, byteorder, order, group, verbose)

    # create virtual dataset for chunked dataset
    chunkmap = 'chunkmap'
    chunked = 'chunked'
    chunk = add_chunkmap(obj, hdffile, rawfile, byteorder, order, chunkmap, verbose)
    if chunk:
        dataset = create_chunked_dataset(hdffile, chunkmap, chunked, verbose)
        create_vds(hdffile, chunkmap, chunked, dataset, verbose)

    if verbose:
        print('done !')


def add_dataset(obj, hdffile, rawfile, byteorder, order, group='', verbose=True):
    with h5py.File(hdffile, 'r+') as h5fp, open(rawfile, 'r') as rawfp:
        # attribute
        attribute = obj.get('attribute', [])
        for attr_name in attribute:
            attr = attribute.get(attr_name)
            data = read_data(rawfp, attr, byteorder)
            name = '/'.join([group, attr_name])
            h5fp.attrs.create(name, data)
            if verbose:
                print('  - attribute "{}" has been created '.format(name), end='')
                print('with data : {}'.format(data))

        # dataset
        dataset = obj.get('dataset', [])
        for ds_name in dataset:
            data = dataset.get(ds_name)
            offset, dsize, dtype, shape = read_info(data, byteorder)
            if order == 0:
                shape = shape[::-1]
            ext = ((rawfile, offset, dsize), )
            name = '/'.join([group, ds_name])
            h5fp.create_dataset(name, shape=shape, dtype=dtype, external=ext)
            if verbose:
                print('  - dataset "{}" has been created '.format(name), end='')
                print('with dtype = "{}" and shape = "{}"'.format(dtype, shape))


def add_chunkmap(obj, hdffile, rawfile, byteorder, order, group='', verbose=True):
    chunkmap = obj.get('chunkmap', None)
    if chunkmap is None:
        return False

    with h5py.File(hdffile, 'r+') as h5fp, open(rawfile, 'r') as rawfp:
        for ds_name in ('rank', 'coord', 'chunkid'):
            if not ds_name in chunkmap:
                continue
            offset, dsize, dtype, shape = read_info(chunkmap[ds_name], byteorder)
            if order == 0:
                shape = shape[::-1]
            ext = ((rawfile, offset, dsize), )
            name = '/'.join([group, ds_name])
            h5fp.create_dataset(name, shape=shape, dtype=dtype, external=ext)
            if verbose:
                print('  - dataset "{}" has been created '.format(name), end='')
                print('with dtype = "{}" and shape = "{}"'.format(dtype, shape))

    return True


def is_chunked_dataset(dataset, cshape):
    dshape = dataset.shape
    status1 = (dshape[0] == np.prod(cshape))
    status2 = np.alltrue([d%c == 0 for d, c in zip(dshape[1:4], cshape[0:3])])
    return (status1 and status2)


def is_external_contiguous(external):
    status = True
    filename, offset, size = external[0]
    for ext in external[1:]:
        status = status and (filename == ext[0])
        status = status and (offset + size == ext[1])
        offset = ext[1]
        size = ext[2]
    return status


def create_chunked_dataset(hdffile, chunkmap, chunked, verbose=True):
    chunked_dataset = []
    with h5py.File(hdffile, 'r+') as h5fp:
        cmap = h5fp.get(chunkmap, None)
        csh = cmap.get('chunkid')[()].shape
        for name, data in h5fp.items():
            #
            # check dataset
            #
            if name == chunkmap:
                # ignore chunkmap
                continue
            if not is_chunked_dataset(data, csh):
                # check if the dataset may be divided into chunks
                continue
            if not is_external_contiguous(data.external):
                # check if the external storage is contiguous in disk
                continue
            #
            # now create chunked dataset
            #
            dshape = data.shape
            rawfile = data.external[0][0]
            chunked_offset = data.external[0][1]
            for i in range(dshape[0]):
                chunked_name = '/'.join([chunked, ('%s' + FMT_CHUNKID) % (name, i)])
                chunked_shape = dshape[1:]
                chunked_dtype = data.dtype
                chunked_dsize = np.prod(chunked_shape) * chunked_dtype.itemsize
                chunked_ext = ((rawfile, chunked_offset, chunked_dsize), )
                kwargs = { 'name' : chunked_name,
                           'shape' : chunked_shape,
                           'dtype' : chunked_dtype,
                           'external' : chunked_ext,
                }
                h5fp.create_dataset(**kwargs)
                # move offset
                chunked_offset = chunked_offset + chunked_dsize
                if verbose:
                    print('  - chunked dataset "{}" has been created '.format(chunked_name), end='')
                    print('with dtype = "{}" and shape = "{}"'.format(chunked_dtype, chunked_shape))
        chunked_dataset.append(name)
    return chunked_dataset


def create_vds(hdffile, chunkmap, chunked, dataset, verbose=True):
    group = 'vds'
    with h5py.File(hdffile, 'r+') as h5fp:
        # get chunk ID
        chunkid = h5fp.get('/'.join([chunkmap, 'chunkid']))[()]
        # try to create VDS for each dataset
        for ds in dataset:
            prefix = '/'.join([chunked, ds]) + FMT_CHUNKID
            srcfile = [prefix % i for i in range(chunkid.size)]
            srcdata = [h5fp.get(src) for src in srcfile]
            #
            # determine data shape assuming 3D chunk
            #
            csh = list(srcdata[0].shape)
            gsh = csh.copy()
            gsh[0] = csh[0]*chunkid.shape[0]
            gsh[1] = csh[1]*chunkid.shape[1]
            gsh[2] = csh[2]*chunkid.shape[2]
            csh = tuple(csh)          # shape of each chunk
            gsh = tuple(gsh)          # shape of global array
            dtype = srcdata[0].dtype  # datatype
            # create VirtualLayout object and assign chunks
            layout = h5py.VirtualLayout(shape=gsh, dtype=dtype)
            for cz in range(chunkid.shape[0]):
                zslice = slice(cz*csh[0], (cz+1)*csh[0])
                for cy in range(chunkid.shape[1]):
                    yslice = slice(cy*csh[1], (cy+1)*csh[1])
                    for cx in range(chunkid.shape[2]):
                        xslice = slice(cx*csh[2], (cx+1)*csh[2])
                        src = h5py.VirtualSource(srcdata[chunkid[cz,cy,cx]])
                        layout[zslice,yslice,xslice,:] = src
            # create VirtualDataset using the layout
            vds_name = '/'.join([group, ds])
            h5fp.create_virtual_dataset(vds_name, layout)
            if verbose:
                print('  - virtual dataset "{}" has been created '.format(vds_name), end='')
                print('with dtype = "{}" and shape = "{}"'.format(dtype, gsh))


def read_data(fp, obj, byteorder):
    offset   = obj['offset']
    datatype = byteorder + obj['datatype']
    shape    = obj['shape']
    size     = np.product(shape)
    fp.seek(offset)
    x = np.fromfile(fp, datatype, size).reshape(shape)
    if len(shape) == 1 and shape[0] == 1:
        x = x[0]
    return x


def read_info(obj, byteorder):
    offset   = obj['offset']
    datatype = byteorder + obj['datatype']
    shape    = obj['shape']
    datasize = np.product(shape) * np.dtype(datatype).itemsize
    return offset, datasize, datatype, shape


def report_error(msg):
    print('Error: {}'.format(msg))
    if _DEBUG:
        raise ValueError(msg)


if __name__ == '__main__':
    import optparse
    parser = optparse.OptionParser()
    parser.add_option('-v', '--verbose', dest='verbose',
                      action='store_true', default=False,
                      help='print verbose messages')
    options, args = parser.parse_args()
    for f in args:
        if os.path.exists(f) and os.path.splitext(f)[1] == '.json':
            json2hdf5(f, verbose=options.verbose)
