import argparse
import time
import numpy as np
import scipy
import scipy.optimize as opt
from scipy.ndimage.filters import gaussian_filter
import torch
from event_utils import *
from objectives import *
from warps import *

def draw_objective_function(xs, ys, ts, ps, objective, warpfunc, x_range=(-200, 200), y_range=(-200, 200),
        gt=(0,0), show_gt=True, resolution=20, img_size=(180, 240)):
    """
    Draw the objective function given by sampling over a range. Depending on the value of resolution, this
    can involve many samples and take some time.
    Parameters:
        xs,ys,ts,ps (numpy array) The event components
        objective (object) The objective function
        warpfunc (object) The warp function
        x_range, y_range (tuple) the range over which to plot the parameters
        gt (tuple) The ground truth
        show_gt (bool) Whether to draw the ground truth in
        resolution (float) The resolution of the sampling
        img_size (tuple) The image sensor size
    """
    width = x_range[1]-x_range[0]
    height = y_range[1]-y_range[0]
    print("Drawing objective function. Taking {} samples".format((width*height)/resolution))
    imshape = (int(height/resolution+0.5), int(width/resolution+0.5))
    img = np.zeros(imshape)
    for x in range(img.shape[1]):
       for y in range(img.shape[0]):
           params = np.array([x*resolution+x_range[0], y*resolution+y_range[0]])
           img[y,x] = -objective.evaluate_function(params, xs, ys, ts, ps, warpfunc, img_size, blur_sigma=0)
    img = cv.normalize(img, None, 0, 1.0, cv.NORM_MINMAX)
    plt.imshow(img, interpolation='bilinear', cmap='viridis')
    plt.xticks([])
    plt.yticks([])
    if show_gt:
        xloc = ((gt[0]-x_range[0])/(width))*imshape[1]
        yloc = ((gt[1]-y_range[0])/(height))*imshape[0]
        plt.axhline(y=yloc, color='r', linestyle='--')
        plt.axvline(x=xloc, color='r', linestyle='--')
    plt.show()

def optimize_contrast(xs, ys, ts, ps, warp_function, objective, optimizer=opt.fmin_bfgs, x0=None,
        numeric_grads=False, blur_sigma=None, img_size=(180, 240)):
    """
    Optimize contrast for a set of events
    Parameters:
    xs (numpy float array) The x components of the events
    ys (numpy float array) The y components of the events
    ts (numpy float array) The timestamps of the events. Timestamps should be ts-t[0] to avoid precision issues.
    ps (numpy float array) The polarities of the events
    warp_function (function) The function with which to warp the events
    objective (objective class object) The objective to optimize
    optimizer (function) The optimizer to use
    x0 (np array) The initial guess for optimization
    numeric_grads (bool) If true, use numeric derivatives, otherwise use analytic drivatives if available.
        Numeric grads tend to be more stable as they are a little less prone to noise and don't require as much
        tuning on the blurring parameter. However, they do make optimization slower.
    img_size (tuple) The size of the event camera sensor
    blur_sigma (float) Size of the blurring kernel. Blurring the images of warped events can
        have a large impact on the convergence of the optimization.

    Returns:
        The max arguments for the warp parameters wrt the objective
    """
    args = (xs, ys, ts, ps, warp_function, img_size, blur_sigma)
    x0 = np.array([0,0])
    if x0 is None:
        x0 = np.zeros(warp_function.dims)
    if numeric_grads:
        argmax = optimizer(objective.evaluate_function, x0, args=args, epsilon=1, disp=False)
    else:
        argmax = optimizer(objective.evaluate_function, x0, fprime=objective.evaluate_gradient, args=args, disp=False)
    return argmax

def optimize(xs, ys, ts, ps, warp, obj, numeric_grads=True, img_size=(180, 240)):
    """
    Optimize contrast for a set of events. Uses optimize_contrast() for the optimiziation, but allows
    blurring schedules for successive optimization iterations.
    Parameters:
    xs (numpy float array) The x components of the events
    ys (numpy float array) The y components of the events
    ts (numpy float array) The timestamps of the events. Timestamps should be ts-t[0] to avoid precision issues.
    ps (numpy float array) The polarities of the events
    warp (function) The function with which to warp the events
    obj (objective class object) The objective to optimize
    numeric_grads (bool) If true, use numeric derivatives, otherwise use analytic drivatives if available.
        Numeric grads tend to be more stable as they are a little less prone to noise and don't require as much
        tuning on the blurring parameter. However, they do make optimization slower.
    img_size (tuple) The size of the event camera sensor

    Returns:
        The max arguments for the warp parameters wrt the objective
    """
    numeric_grads = numeric_grads if obj.has_derivative else True
    argmax_an = optimize_contrast(xs, ys, ts, ps, warp, obj, numeric_grads=numeric_grads, blur_sigma=blur, img_size=img_size)
    return argmax_an

def optimize_r2(xs, ys, ts, ps, warp, obj, numeric_grads=True, img_size=(180, 240)):
    """
    Optimize contrast for a set of events, finishing with SoE loss.
    Parameters:
    xs (numpy float array) The x components of the events
    ys (numpy float array) The y components of the events
    ts (numpy float array) The timestamps of the events. Timestamps should be ts-t[0] to avoid precision issues.
    ps (numpy float array) The polarities of the events
    warp (function) The function with which to warp the events
    obj (objective class object) The objective to optimize
    numeric_grads (bool) If true, use numeric derivatives, otherwise use analytic drivatives if available.
        Numeric grads tend to be more stable as they are a little less prone to noise and don't require as much
        tuning on the blurring parameter. However, they do make optimization slower.
    img_size (tuple) The size of the event camera sensor

    Returns:
        The max arguments for the warp parameters wrt the objective
    """
    soe_obj = soe_objective()
    numeric_grads = numeric_grads if obj.has_derivative else True
    argmax_an = optimize_contrast(xs, ys, ts, ps, warp, obj, numeric_grads=numeric_grads, blur_sigma=None)
    argmax_an = optimize_contrast(xs, ys, ts, ps, warp, soe_obj, x0=argmax_an, numeric_grads=numeric_grads, blur_sigma=1.0)
    return argmax_an

if __name__ == "__main__":
    """
    Quick demo of various objectives.
    Args:
        path Path to h5 file with event data
        gt Ground truth optic flow for event slice
        img_size The size of the event camera sensor
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="h5 events path")
    parser.add_argument("--gt", nargs='+', type=float, default=(0,0))
    parser.add_argument("--img_size", nargs='+', type=float, default=(180,240))
    args = parser.parse_args()

    xs, ys, ts, ps = read_h5_event_components(args.path)
    ts = ts-ts[0]
    gt_params = tuple(args.gt)
    img_size=tuple(args.img_size)

    start_idx = 20000
    end_idx=start_idx+15000
    blur = None

    draw_objective_function(xs[start_idx:end_idx], ys[start_idx:end_idx], ts[start_idx:end_idx], ps[start_idx:end_idx], variance_objective(), linvel_warp())

    objectives = [r1_objective(), zhu_timestamp_objective(), variance_objective(), sos_objective(), soe_objective(), moa_objective(),
            isoa_objective(), sosa_objective(), rms_objective()]
    warp = linvel_warp()
    for obj in objectives:
        argmax = optimize(xs[start_idx:end_idx], ys[start_idx:end_idx], ts[start_idx:end_idx], ps[start_idx:end_idx], warp, obj, numeric_grads=True)
        loss = obj.evaluate_function(argmax, xs[start_idx:end_idx], ys[start_idx:end_idx], ts[start_idx:end_idx],
                ps[start_idx:end_idx], warp, img_size=img_size)
        gtloss = obj.evaluate_function(gt_params, xs[start_idx:end_idx], ys[start_idx:end_idx],
                ts[start_idx:end_idx], ps[start_idx:end_idx], warp, img_size=img_size)
        print("{}:({})={}, gt={}".format(obj.name, argmax, loss, gtloss))
        if obj.has_derivative:
            argmax = optimize(xs[start_idx:end_idx], ys[start_idx:end_idx], ts[start_idx:end_idx],
                    ps[start_idx:end_idx], warp, obj, numeric_grads=False)
            loss_an = obj.evaluate_function(argmax, xs[start_idx:end_idx], ys[start_idx:end_idx],
                    ts[start_idx:end_idx], ps[start_idx:end_idx], warp, img_size=img_size)
            print("   analytical:{}={}".format(argmax, loss_an))
