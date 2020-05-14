# Event Contrast Maximization Library
A python library for contrast maximization and voxel creation using events.

## Usage
To use this library, you need to first convert the events to hdf5 file format. This is necessary because reading the events from rosbag is _painfully_ slow. So use the script in tools/rosbag_to_h5.py like so:
```python rosbag_to_h5.py --output_dir /tmp/my_rosbag.bag --event_topic /dvs/events```
Obviously for this to work, you will need to have ros installed.

If you just want to get going, you can also download a couple of h5 event sequences from [here](https://drive.google.com/open?id=1z3Gjn4HLkHhgFeoa2viC-fuldUCZQGUL) (slider_depth from Mueggler, The Event-Camera Dataset and Simulator, IJRR17 and a super simple sequence I recorded myself).

To run a quick demo (show the cost landscape and run through a bunch of objective functions) go to utils and run 
```python events_cmax.py /path/to/my/events.h5```
To implement your own objective functions, check out `utils/objectives.py` to implement your own warps (at the moment there is only linear velocity warp, I'll put more in soon) check out `utils/warps.py`.

## Dependencies
You need to be running at least Python 3 and have PyTorch installed (GPU not necessary). 

## Overview
This library contains functions for generally useful event-based vision tasks. Here is q quick overview:
### Tools
#### Event conversions
-`rosbag_to_h5.py` converts rosbag events to HDF5 files, together with lots of useful metadata (for example, images contain the index of the time-synchronized event). Works for color event cameras.

-`h5_to_memmap.py` converts HDF5 events to MemMap events, as sometimes used at [RPG](http://rpg.ifi.uzh.ch/).

Implementing your own data format converter is easy, by implementing an event packager in `event_packagers.py`.

### Utils
This contains the contrast maximization code as well as a load of other utility functions.
#### `event_utils.py`
- Binary search over timestamps for HDF5 files (this means you don't need to load the entire event sequence into RAM to search) and binary search of events as tensors.
- Loading HDF5 events (`read_h5_events`)
- Turn a voxel grid into an image (each 'slice' placed side by side) (`get_voxel_grid_as_image`, `plot_voxel_grid`)
- Get a mask for all events out of a certain spatial bounds (`events_bounds_mask`, `clip_events_to_bounds`)
-  Turn a set of events to an image (`events_to_image`, `events_to_image_torch`) turns a set of events into an image. If the event coordinates are floating point, due to camera calibration or other transforming, this can be done with ilinear interpolation. If some warp has been applied to the events and the jacobian of that warp is available, you may compute the derivative images using `events_to_image_drv`. You may also get an average timestamp image as described in [`Unsupervised Event-based Learning of Optical Flow, Depth, and Egomotion`](https://arxiv.org/abs/1812.08156) with `events_to_timestamp_image`, `events_to_timestamp_image_torch` (currently the only public implementation of this).
- Generate voxel grids. This is frequently required for [deep learning](https://timostoff.github.io/20ecnn) with events. You can generate voxel grids with fixed k events (`voxel_grids_fixed_n_torch`), fixed t-seconds (`voxel_grids_fixed_t_torch`) between two times (`events_to_voxel_timesync_torch`) or simply turn the entire collection of events into a single voxels (`events_to_voxel`, `events_to_voxel_torch`). You can also get the voxel grid positive and negative channels separately if you wish (`events_to_neg_pos_voxel`, `events_to_neg_pos_voxel_torch`). So you can see, there are many options.
- Warp events over a flow field (`warp_events_flow_torch`).
When functions are appended with `_torch`, it indicates that the method expects data inputs to be `torch.tensor`, otherwise `np.array` is expected.
#### `events_cmax.py`
Contains the code for running contrast maximization with various objectives/warp functions on the events. To implement your own warp function, see `warps.py`, for your own objective functions see `objectives.py`. For example code, see the main code in `events_cmax`.

## Citation
If you use any of this code, please cite: T. Stoffregen and L. Kleeman, Event Cameras, Contrast Maximization and Reward Functions: An Analysis, The IEEE Conference on Computer Vision and Pattern Recognition (CVPR), June 2019.
```
@InProceedings{Stoffregen19cvpr,
author = {Stoffregen, Timo and Kleeman, Lindsay},
title = {Event Cameras, Contrast Maximization and Reward Functions: An Analysis},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2019}
} 
```
