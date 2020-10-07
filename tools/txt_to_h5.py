import glob
import argparse
import os
import h5py
import pandas as pd
import numpy as np
from event_packagers import *


def get_sensor_size(txt_path):
    try:
        header = pd.read_csv(txt_path, delim_whitespace=True, header=None, names=['width', 'height'],
                         dtype={'width': np.int, 'height': np.int},
                         nrows=1)
        width, height = header.values[0]
        sensor_size = [height, width]
    except:
        sensor_size = None
        print('Warning: could not read sensor size from first line of {}'.format(txt_path))
    return sensor_size


def extract_txt(txt_path, output_path, zero_timestamps=False,
                packager=hdf5_packager):
    ep = packager(output_path)
    first_ts = -1
    t0 = -1
    if not os.path.exists(txt_path):
        print("{} does not exist!".format(txt_path))
        return

    # compute sensor size
    sensor_size = get_sensor_size(txt_path)
    # Extract events to h5
    ep.set_data_available(num_images=0, num_flow=0)
    total_num_pos, total_num_neg, last_ts = 0, 0, 0

    chunksize = 100000
    iterator = pd.read_csv(txt_path, delim_whitespace=True, header=None,
                           names=['t', 'x', 'y', 'pol'],
                           dtype={'t': np.float64, 'x': np.int16, 'y': np.int16, 'pol': np.int16},
                           engine='c',
                           skiprows=1, chunksize=chunksize, nrows=None, memory_map=True)

    for i, event_window in enumerate(iterator):
        events = event_window.values
        ts = events[:, 0].astype(np.float64)
        xs = events[:, 1].astype(np.int16)
        ys = events[:, 2].astype(np.int16)
        ps = events[:, 3]
        ps[ps < 0] = 0 # should be [0 or 1]
        ps = ps.astype(bool)

        if first_ts == -1:
            first_ts = ts[0]

        if zero_timestamps:
            ts -= first_ts
        last_ts = ts[-1]
        if sensor_size is None or sensor_size[0] < max(ys) or sensor_size[1] < max(xs):
            sensor_size = [max(xs), max(ys)]
            print("Sensor size inferred from events as {}".format(sensor_size))

        sum_ps = sum(ps)
        total_num_pos += sum_ps
        total_num_neg += len(ps) - sum_ps
        ep.package_events(xs, ys, ts, ps)
        if i % 10 == 9:
            print('Events written: {} M'.format((total_num_pos + total_num_neg) / 1e6)) 
    print('Events written: {} M'.format((total_num_pos + total_num_neg) / 1e6)) 
    print("Detect sensor size {}".format(sensor_size))
    t0 = 0 if zero_timestamps else first_ts
    ep.add_metadata(total_num_pos, total_num_neg, last_ts-t0, t0, last_ts, num_imgs=0, num_flow=0, sensor_size=sensor_size)


def extract_txts(txt_paths, output_dir, zero_timestamps=False):
    for path in txt_paths:
        filename = os.path.splitext(os.path.basename(path))[0]
        out_path = os.path.join(output_dir, "{}.h5".format(filename))
        print("Extracting {} to {}".format(path, out_path))
        extract_txt(path, out_path, zero_timestamps=zero_timestamps)


if __name__ == "__main__":
    """
    Tool for converting txt events to an efficient HDF5 format that can be speedily
    accessed by python code.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="txt file to extract or directory containing txt files")
    parser.add_argument("--output_dir", default="/tmp/extracted_data", help="Folder where to extract the data")
    parser.add_argument('--zero_timestamps', action='store_true', help='If true, timestamps will be offset to start at 0')
    args = parser.parse_args()

    print('Data will be extracted in folder: {}'.format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if os.path.isdir(args.path):
        txt_paths = sorted(list(glob.glob(os.path.join(args.path, "*.txt")))
                         + list(glob.glob(os.path.join(args.path, "*.zip"))))
    else:
        txt_paths = [args.path]
    extract_txts(txt_paths, args.output_dir, zero_timestamps=args.zero_timestamps)
