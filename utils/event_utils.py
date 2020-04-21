import argparse
import time
import h5py
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import torch

def binary_search_h5_timestamp(hdf_path, l, r, x, side='left'):
    f = h5py.File(hdf_path, 'r')
    if r is None:
        r = f.attrs['num_events']-1
    while l <= r:
        mid = l + (r - l)//2;
        midval = f['events/ts'][mid]
        if midval == x:
            return mid
        elif midval < x:
            l = mid + 1
        else:
            r = mid - 1
    if side == 'left':
        return l
    return r

def read_h5_events(hdf_path):
    f = h5py.File(hdf_path, 'r')
    if 'events/x' in f:
        #legacy
        events = np.stack((f['events/x'][:], f['events/y'][:], f['events/ts'][:], np.where(f['events/p'][:], 1, -1)), axis=1)
    else:
        events = np.stack((f['events/xs'][:], f['events/ys'][:], f['events/ts'][:], np.where(f['events/ps'][:], 1, -1)), axis=1)
    return events

def read_h5_event_components(hdf_path):
    f = h5py.File(hdf_path, 'r')
    if 'events/x' in f:
        #legacy
        return (f['events/x'][:], f['events/y'][:], f['events/ts'][:], np.where(f['events/p'][:], 1, -1))
    else:
        return (f['events/xs'][:], f['events/ys'][:], f['events/ts'][:], np.where(f['events/ps'][:], 1, -1))

def plot_image(image, lognorm=False, cmap='gray'):
    if lognorm:
        image = np.log10(image)
        cmap='viridis'
    image = cv.normalize(image, None, 0, 1.0, cv.NORM_MINMAX)
    plt.imshow(image, cmap=cmap)
    plt.show()

def plot_voxel_grid(voxelgrid, cmap='gray'):
    images = []
    splitter = np.ones((voxelgrid.shape[1], 2))*np.max(voxelgrid)
    for image in voxelgrid:
        images.append(image)
        images.append(splitter)
    images.pop()
    sidebyside = np.hstack(images)
    sidebyside = cv.normalize(sidebyside, None, 0, 1.0, cv.NORM_MINMAX)
    plt.imshow(sidebyside, cmap=cmap)
    plt.show()

def events_bounds_mask(xs, ys, x_min, x_max, y_min, y_max):
    """
    Get a mask of the events that are within the given bounds
    """
    mask = np.where(np.logical_or(xs<=x_min, xs>x_max), 0.0, 1.0)
    mask *= np.where(np.logical_or(ys<=y_min, ys>y_max), 0.0, 1.0)
    return mask

def clip_events_to_bounds(xs, ys, ps, bounds):
    """
    Clip events to the given bounds
    """
    mask = events_bounds_mask(xs, ys, 0, bounds[1], 0, bounds[0])
    return xs*mask, ys*mask, ps*mask

def events_to_image(xs, ys, ps, sensor_size=(180, 240), interpolation=None, padding=False):
    """
    Place events into an image using numpy
    """
    img_size = sensor_size
    if interpolation == 'bilinear' and xs.dtype is not torch.long and xs.dtype is not torch.long:
        xt, yt, pt = torch.from_numpy(xs), torch.from_numpy(ys), torch.from_numpy(ps)
        xt, yt, pt = xt.float(), yt.float(), pt.float()
        img = events_to_image_torch(xt, yt, pt, clip_out_of_range=True, interpolation='bilinear', padding=padding)
        img = img.numpy()
    else:
        coords = np.stack((ys, xs))
        abs_coords = np.ravel_multi_index(coords, sensor_size)
        img = np.bincount(abs_coords, weights=ps, minlength=sensor_size[0]*sensor_size[1])
    img = img.reshape(sensor_size)
    return img

def interpolate_to_image(pxs, pys, dxs, dys, weights, img):
    """
    Accumulate x and y coords to an image using bilinear interpolation
    """
    img.index_put_((pys,   pxs  ), weights*(1.0-dxs)*(1.0-dys), accumulate=True)
    img.index_put_((pys,   pxs+1), weights*dxs*(1.0-dys), accumulate=True)
    img.index_put_((pys+1, pxs  ), weights*(1.0-dxs)*dys, accumulate=True)
    img.index_put_((pys+1, pxs+1), weights*dxs*dys, accumulate=True)
    return img

def interpolate_to_derivative_img(pxs, pys, dxs, dys, d_img, w1, w2):
    """
    Accumulate x and y coords to an image using double weighted bilinear interpolation
    """
    for i in range(d_img.shape[0]):
        d_img[i].index_put_((pys,   pxs  ), w1[i] * (-(1.0-dys)) + w2[i] * (-(1.0-dxs)), accumulate=True)
        d_img[i].index_put_((pys,   pxs+1), w1[i] * (1.0-dys)    + w2[i] * (-dxs), accumulate=True)
        d_img[i].index_put_((pys+1, pxs  ), w1[i] * (-dys)       + w2[i] * (1.0-dxs), accumulate=True)
        d_img[i].index_put_((pys+1, pxs+1), w1[i] * dys          + w2[i] *  dxs, accumulate=True)

def events_to_image_drv(xn, yn, pn, jacobian_xn, jacobian_yn,
        device=None, sensor_size=(180, 240), clip_out_of_range=True,
        interpolation=None, padding=True, compute_gradient=False):
    """
    Method to turn event tensor to image. Allows for bilinear interpolation.
        :param xs: tensor of x coords of events
        :param ys: tensor of y coords of events
        :param ps: tensor of event polarities/weights
        :param device: the device on which the image is. If none, set to events device
        :param sensor_size: the size of the image sensor/output image
        :param clip_out_of_range: if the events go beyond the desired image size,
            clip the events to fit into the image
        :param interpolation: which interpolation to use. Options=None,'bilinear'
        :param if bilinear interpolation, allow padding the image by 1 to allow events to fit:
    """
    xt, yt, pt = torch.from_numpy(xn), torch.from_numpy(yn), torch.from_numpy(pn)
    xs, ys, ps, = xt.float(), yt.float(), pt.float()
    if compute_gradient:
        jacobian_x, jacobian_y = torch.from_numpy(jacobian_xn), torch.from_numpy(jacobian_yn)
        jacobian_x, jacobian_y = jacobian_x.float(), jacobian_y.float()
    if device is None:
        device = xs.device
    if padding:
        img_size = (sensor_size[0]+1, sensor_size[1]+1)
    else:
        img_size = sensor_size

    mask = torch.ones(xs.size())
    if clip_out_of_range:
        zero_v = torch.tensor([0.])
        ones_v = torch.tensor([1.])
        clipx = img_size[1] if interpolation is None and padding==False else img_size[1]-1
        clipy = img_size[0] if interpolation is None and padding==False else img_size[0]-1
        mask = torch.where(xs>=clipx, zero_v, ones_v)*torch.where(ys>=clipy, zero_v, ones_v)

    pxs = xs.floor()
    pys = ys.floor()
    dxs = xs-pxs
    dys = ys-pys
    pxs = (pxs*mask).long()
    pys = (pys*mask).long()
    masked_ps = ps*mask
    img = torch.zeros(img_size).to(device)
    interpolate_to_image(pxs, pys, dxs, dys, masked_ps, img)

    if compute_gradient:
        d_img = torch.zeros((2, *img_size)).to(device)
        w1 = jacobian_x*masked_ps
        w2 = jacobian_y*masked_ps
        interpolate_to_derivative_img(pxs, pys, dxs, dys, d_img, w1, w2)
        d_img = d_img.numpy()
    else:
        d_img = None
    return img.numpy(), d_img

def events_to_zhu_timestamp_image(xn, yn, ts, pn,
        device=None, sensor_size=(180, 240), clip_out_of_range=True,
        interpolation='bilinear', padding=True, compute_gradient=False, showimg=False):
    """
    Method to generate the average timestamp images from 'Zhu19, Unsupervised Event-based Learning 
    of Optical Flow, Depth, and Egomotion'. This method does not have known derivative.
    """
    xt, yt, ts, pt = torch.from_numpy(xn), torch.from_numpy(yn), torch.from_numpy(ts), torch.from_numpy(pn)
    xs, ys, ts, ps = xt.float(), yt.float(), ts.float(), pt.float()
    zero_v = torch.tensor([0.])
    ones_v = torch.tensor([1.])
    if device is None:
        device = xs.device
    if padding:
        img_size = (sensor_size[0]+1, sensor_size[1]+1)
    else:
        img_size = sensor_size

    mask = torch.ones(xs.size())
    if clip_out_of_range:
        clipx = img_size[1] if interpolation is None and padding==False else img_size[1]-1
        clipy = img_size[0] if interpolation is None and padding==False else img_size[0]-1
        mask = torch.where(xs>=clipx, zero_v, ones_v)*torch.where(ys>=clipy, zero_v, ones_v)

    pos_events_mask = torch.where(ps>0, ones_v, zero_v)
    neg_events_mask = torch.where(ps<=0, ones_v, zero_v)
    normalized_ts = (ts-ts[0])/(ts[-1]+1e-6)
    pxs = xs.floor()
    pys = ys.floor()
    dxs = xs-pxs
    dys = ys-pys
    pxs = (pxs*mask).long()
    pys = (pys*mask).long()
    masked_ps = ps*mask

    pos_weights = normalized_ts*pos_events_mask
    neg_weights = normalized_ts*neg_events_mask
    img_pos = torch.zeros(img_size).to(device)
    img_pos_cnt = torch.ones(img_size).to(device)
    img_neg = torch.zeros(img_size).to(device)
    img_neg_cnt = torch.ones(img_size).to(device)

    interpolate_to_image(pxs, pys, dxs, dys, pos_weights, img_pos)
    interpolate_to_image(pxs, pys, dxs, dys, pos_events_mask, img_pos_cnt)
    interpolate_to_image(pxs, pys, dxs, dys, neg_weights, img_neg)
    interpolate_to_image(pxs, pys, dxs, dys, neg_events_mask, img_neg_cnt)

    img_pos, img_pos_cnt = img_pos.numpy(), img_pos_cnt.numpy()
    img_pos_cnt[img_pos_cnt==0] = 1
    img_neg, img_neg_cnt = img_neg.numpy(), img_neg_cnt.numpy()
    img_neg_cnt[img_neg_cnt==0] = 1
    return img_pos, img_neg #/img_pos_cnt, img_neg/img_neg_cnt

def events_to_image_torch(xs, ys, ps,
        device=None, sensor_size=(180, 240), clip_out_of_range=True,
        interpolation=None, padding=True):
    """
    Method to turn event tensor to image. Allows for bilinear interpolation.
        :param xs: tensor of x coords of events
        :param ys: tensor of y coords of events
        :param ps: tensor of event polarities/weights
        :param device: the device on which the image is. If none, set to events device
        :param sensor_size: the size of the image sensor/output image
        :param clip_out_of_range: if the events go beyond the desired image size,
            clip the events to fit into the image
        :param interpolation: which interpolation to use. Options=None,'bilinear'
        :param if bilinear interpolation, allow padding the image by 1 to allow events to fit:
    """
    if device is None:
        device = xs.device
    if interpolation == 'bilinear' and padding:
        img_size = (sensor_size[0]+1, sensor_size[1]+1)
    else:
        img_size = sensor_size

    mask = torch.ones(xs.size())
    if clip_out_of_range:
        zero_v = torch.tensor([0.])
        ones_v = torch.tensor([1.])
        clipx = img_size[1] if interpolation is None and padding==False else img_size[1]-1
        clipy = img_size[0] if interpolation is None and padding==False else img_size[0]-1
        mask = torch.where(xs>=clipx, zero_v, ones_v)*torch.where(ys>=clipy, zero_v, ones_v)

    img = torch.zeros(img_size).to(device)
    if interpolation == 'bilinear' and xs.dtype is not torch.long and xs.dtype is not torch.long:
        pxs = xs.floor()
        pys = ys.floor()
        dxs = xs-pxs
        dys = ys-pys
        pxs = (pxs*mask).long()
        pys = (pys*mask).long()
        masked_ps = ps*mask
        interpolate_to_image(pxs, pys, dxs, dys, masked_ps, img)
    else:
        if xs.dtype is not torch.long:
            xs = xs.long().to(device)
        if ys.dtype is not torch.long:
            ys = ys.long().to(device)
        img.index_put_((ys, xs), ps, accumulate=True)
    return img

def voxel_grids_fixed_n_torch(xs, ys, ts, ps, B, n, sensor_size=(180, 240), temporal_bilinear=True):
    """
    Given a set of events, return a list of voxel grids with a fixed number of events.
    Parameters
    ----------
    xs : list of event x coordinates (torch tensor)
    ys : list of event y coordinates (torch tensor)
    ts : list of event timestamps (torch tensor)
    ps : list of event polarities (torch tensor)
    B : number of bins in output voxel grids (int)
    n : the number of events per voxel
    sensor_size : the size of the event sensor/output voxels
    temporal_bilinear : whether the events should be naively
        accumulated to the voxels (faster), or properly
        temporally distributed
    Returns
    -------
    voxels: list of output voxel grids
    """
    voxels = []
    for idx in range(0, len(xs)-n, n):
        voxels.append(events_to_voxel_torch(xs[idx:idx+n], ys[idx:idx+n],
            ts[idx:idx+n], ps[idx:idx+n], B, sensor_size=sensor_size,
            temporal_bilinear=temporal_bilinear))
    return voxels

def voxel_grids_fixed_t_torch(xs, ys, ts, ps, B, t, sensor_size=(180, 240), temporal_bilinear=True):
    """
    Given a set of events, return a list of voxel grids with a fixed temporal width.
    Parameters
    ----------
    xs : list of event x coordinates (torch tensor)
    ys : list of event y coordinates (torch tensor)
    ts : list of event timestamps (torch tensor)
    ps : list of event polarities (torch tensor)
    B : number of bins in output voxel grids (int)
    t : the time width of the voxel grids
    sensor_size : the size of the event sensor/output voxels
    temporal_bilinear : whether the events should be naively
        accumulated to the voxels (faster), or properly
        temporally distributed
    Returns
    -------
    voxels: list of output voxel grids
    """
    device = xs.device
    voxels = []
    np_ts = ts.cpu().numpy()
    for t_start in np.arange(ts[0].item(), ts[-1].item()-t, t):
        voxels.append(events_to_voxel_timesync_torch(xs, ys, ts, ps, B, t_start, t_start+t, np_ts=np_ts,
            sensor_size=sensor_size, temporal_bilinear=temporal_bilinear))
    return voxels

def events_to_voxel_timesync_torch(xs, ys, ts, ps, B, t0, t1, device=None, np_ts=None, sensor_size=(180, 240), temporal_bilinear=True):
    """
    Given a set of events, return a voxel grid of the events between t0 and t1
    Parameters
    ----------
    xs : list of event x coordinates (torch tensor)
    ys : list of event y coordinates (torch tensor)
    ts : list of event timestamps (torch tensor)
    ps : list of event polarities (torch tensor)
    B : number of bins in output voxel grids (int)
    t0 : the start time of the voxel grid
    t1 : the end time of the voxel grid
    device : device to put voxel grid. If left empty, same device as events
    np_ts : a numpy copy of ts (optional). If not given, will be created in situ
    sensor_size : the size of the event sensor/output voxels
    temporal_bilinear : whether the events should be naively
        accumulated to the voxels (faster), or properly
        temporally distributed
    Returns
    -------
    voxel: voxel of the events between t0 and t1
    """
    assert(t1>t0)
    if np_ts is None:
        np_ts = ts.cpu().numpy()
    if device is None:
        device = xs.device
    start_idx = np.searchsorted(np_ts, t0)
    end_idx = np.searchsorted(np_ts, t1)
    assert(start_idx < end_idx)
    voxel = events_to_voxel_torch(xs[start_idx:end_idx], ys[start_idx:end_idx],
        ts[start_idx:end_idx], ps[start_idx:end_idx], B, device, sensor_size=sensor_size,
        temporal_bilinear=temporal_bilinear)
    return voxel

def events_to_voxel_torch(xs, ys, ts, ps, B, device=None, sensor_size=(180, 240), temporal_bilinear=True):
    """
    Turn set of events to a voxel grid tensor, using temporal bilinear interpolation
    Parameters
    ----------
    xs : list of event x coordinates (torch tensor)
    ys : list of event y coordinates (torch tensor)
    ts : list of event timestamps (torch tensor)
    ps : list of event polarities (torch tensor)
    B : number of bins in output voxel grids (int)
    device : device to put voxel grid. If left empty, same device as events
    sensor_size : the size of the event sensor/output voxels
    temporal_bilinear : whether the events should be naively
        accumulated to the voxels (faster), or properly
        temporally distributed
    Returns
    -------
    voxel: voxel of the events between t0 and t1
    """
    if device is None:
        device = xs.device
    assert(len(xs)==len(ys) and len(ys)==len(ts) and len(ts)==len(ps))
    num_events_per_bin = len(xs)//B
    bins = []
    dt = ts[-1]-ts[0]
    t_norm = (ts-ts[0])/dt*(B-1)
    zeros = torch.zeros(t_norm.size())
    for bi in range(B):
        if temporal_bilinear:
            bilinear_weights = torch.max(zeros, 1.0-torch.abs(t_norm-bi))
            weights = ps*bilinear_weights
        else:
            beg = bi*num_events_per_bin
            end = beg + num_events_per_bin
            vb = events_to_image_torch(xs[beg:end], ys[beg:end],
                    weights[beg:end], device, sensor_size=sensor_size,
                    clip_out_of_range=False)
        vb = events_to_image_torch(xs, ys,
                weights, device, sensor_size=sensor_size,
                clip_out_of_range=False)
        bins.append(vb)
    bins = torch.stack(bins)
    return bins

def events_to_neg_pos_voxel_torch(xs, ys, ts, ps, B, device=None,
        sensor_size=(180, 240), temporal_bilinear=True):
    zero_v = torch.tensor([0.])
    ones_v = torch.tensor([1.])
    pos_weights = torch.where(ps>0, ones_v, zero_v)
    neg_weights = torch.where(ps<=0, ones_v, zero_v)

    voxel_pos = events_to_voxel_torch(xs, ys, ts, pos_weights, B, device=device,
            sensor_size=sensor_size, temporal_bilinear=temporal_bilinear)
    voxel_neg = events_to_voxel_torch(xs, ys, ts, neg_weights, B, device=device,
            sensor_size=sensor_size, temporal_bilinear=temporal_bilinear)

    return voxel_pos, voxel_neg

if __name__ == "__main__":
    """
    Quick demo of some of the voxel/event image generating functions
    in the utils library
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="h5 events path")
    args = parser.parse_args()
    events = read_h5_events(args.path)
    xs, ys, ts, ps = read_h5_event_components(args.path)


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = 'cpu'

    test_loop = 100
    s=80000
    e=s+150000
    bins = 5

    start = time.time()
    for i in range(test_loop):
        xt, yt, tt, pt = torch.from_numpy(xs), torch.from_numpy(ys), torch.from_numpy(ts), torch.from_numpy(ps)
        xt = xt.float().to(device)
        yt = yt.float().to(device)
        tt = (tt[:]-tt[0]).float().to(device)
        pt = pt.float().to(device)
    end = time.time()
    print("conversion to torch: time elapsed  = {:0.5f}".format((end-start)/test_loop))

    start = time.time()
    t0 = ts[len(ts)-1] #worst case
    for i in range(test_loop):
       idx = binary_search_h5_timestamp(args.path, 0, None, t0)
    end = time.time()
    print("binary search of hdf5 (idx={}): time elapsed  = {:0.5f}".format(idx, (end-start)/test_loop))

    start = time.time()
    t0 = ts[len(ts)-1] #worst case
    for i in range(test_loop):
        idx = np.searchsorted(ts, t0)
    end = time.time()
    print("binary search of np timestamps (idx={}): time elapsed  = {:0.5f}".format(idx, (end-start)/test_loop))

    start = time.time()
    for i in range(test_loop):
        events_to_image(xs, ys, ps)
    end = time.time()
    print("event-to-image, numpy: time elapsed  = {:0.5f}".format((end-start)/test_loop))

    start = time.time()
    for i in range(test_loop):
        events_to_image_torch(xt, yt, pt, device, clip_out_of_range=False, padding=False)
    end = time.time()
    print("event-to-image, vanilla: time elapsed = {:0.5f}".format((end-start)/test_loop))

    start = time.time()
    for i in range(test_loop):
        events_to_image_torch(xt, yt, pt, device, interpolation='bilinear')
    end = time.time()
    print("event-to-image, bilinear: time elapsed = {:0.5f}".format((end-start)/test_loop))

    start = time.time()
    for i in range(test_loop):
        events_to_voxel_torch(xt[s:e], yt[s:e], tt[s:e], pt[s:e], bins, device)
    end = time.time()
    print("voxel grid: time elapsed = {:0.5f}".format((end-start)/test_loop))

    start = time.time()
    for i in range(test_loop):
        events_to_voxel_timesync_torch(xt, yt, tt, pt, bins, tt[s], tt[e])
    end = time.time()
    print("voxel grid timesynced: time elapsed = {:0.5f}".format((end-start)/test_loop))

    start = time.time()
    for i in range(test_loop):
        vgs = voxel_grids_fixed_t_torch(xt, yt, tt, pt, bins, 0.1)
    end = time.time()
    tottime = (end-start)/test_loop
    print("voxel grids fixed t: time elapsed = {:0.5f}/{}={:0.5f}".format(tottime, len(vgs), tottime/len(vgs)))

    start = time.time()
    for i in range(test_loop):
        vgs = voxel_grids_fixed_n_torch(xt, yt, tt, pt, bins, len(xt)//34)
    end = time.time()
    tottime = (end-start)/test_loop
    print("voxel grids fixed n: time elapsed = {:0.5f}/{}={:0.5f}".format(tottime, len(vgs), tottime/len(vgs)))

    start = time.time()
    for i in range(test_loop):
        vgs = warp_events_flow_torch(xt, yt, tt, pt, None) 
    end = time.time()
    tottime = (end-start)/test_loop
    print("voxel grids fixed n: time elapsed = {:0.5f}/{}={:0.5f}".format(tottime, len(vgs), tottime/len(vgs)))
