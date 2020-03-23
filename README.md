# Event Contrast Maximization Library
A python library for contrast maximization and voxel creation using events.

## Usage
To use this library, you need to first convert the events to hdf5 file format. This is necessary because reading the events from rosbag is _painfully_ slow. So use the script in tools/rosbag_to_h5.py like so:
```python rosbag_to_h5.py --output_dir /tmp/my_rosbag.bag --event_topic /dvs/events```
Obviously for this to work, you will need to have ros installed.

If you just want to get going, you can also download a couple of h5 event sequences from here (slider_depth from Mueggler, The Event-Camera Dataset and Simulator, IJRR17 and a super simple sequence I recorded myself).

To run a quick demo (show the cost landscape and run through a bunch of objective functions) go to utils and run 
```python events_cmax.py /path/to/my/events.h5```
To implement your own objective functions, check out `utils/objectives.py` to implement your own warps (at the moment there is only linear velocity warp, I'll put more in soon) check out `utils/warps.py`.

## Dependencies
You need to be running at least Python 3 and have PyTorch installed (GPU not necessary). 

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
