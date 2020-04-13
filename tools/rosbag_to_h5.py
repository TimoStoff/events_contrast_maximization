import argparse
import rosbag
import rospy
from cv_bridge import CvBridge, CvBridgeError
import os
import h5py
import numpy as np

def append_to_dataset(dataset, data):
    dataset.resize(dataset.shape[0] + len(data), axis=0)
    if len(data) == 0:
        return
    dataset[-len(data):] = data[:]

def timestamp_float(ts):
    return ts.secs + ts.nsecs / float(1e9)

def get_rosbag_stats(bag, event_topic, image_topic=None, flow_topic=None):
    num_event_msgs = 0
    num_img_msgs = 0
    num_flow_msgs = 0
    topics = bag.get_type_and_topic_info().topics
    for topic_name, topic_info in topics.iteritems():
        if topic_name == args.event_topic:
            num_event_msgs = topic_info.message_count
            print('Found events topic: {} with {} messages'.format(topic_name, topic_info.message_count))
        if topic_name == image_topic:
            num_img_msgs = topic_info.message_count
            print('Found image topic: {} with {} messages'.format(topic_name, num_img_msgs))
        if topic_name == flow_topic:
            num_flow_msgs = topic_info.message_count
            print('Found flow topic: {} with {} messages'.format(topic_name, num_flow_msgs))
    return num_event_msgs, num_img_msgs, num_flow_msgs

#Inspired by https://github.com/uzh-rpg/rpg_e2vid
def extract_rosbag(rosbag_path, output_path, event_topic, image_topic=None,
        flow_topic=None, start_time=None, end_time=None, zero_timestamps=False):
    topics = (event_topic, image_topic, flow_topic)
    event_msg_sum = 0
    num_msgs_between_logs = 25
    first_ts = -1
    if not os.path.exists(rosbag_path):
        print("{} does not exist!".format(rosbag_path))
        return
    with rosbag.Bag(rosbag_path, 'r') as bag:
        # Look for the topics that are available and save the total number of messages for each topic (useful for the progress bar)
        num_event_msgs, num_img_msgs, num_flow_msgs = get_rosbag_stats(bag, event_topic, image_topic, flow_topic)
        # Extract events to h5
        xs, ys, ts, ps = [], [], [], []
        max_buffer_size = 1000000
        events_file = h5py.File(output_path, 'w')
        event_xs = events_file.create_dataset("events/xs", (0, ), dtype=np.dtype(np.int16), maxshape=(None, ), chunks=True)
        event_ys = events_file.create_dataset("events/ys", (0, ), dtype=np.dtype(np.int16), maxshape=(None, ), chunks=True)
        event_ts = events_file.create_dataset("events/ts", (0, ), dtype=np.dtype(np.float64), maxshape=(None, ), chunks=True)
        event_ps = events_file.create_dataset("events/ps", (0, ), dtype=np.dtype(np.bool_), maxshape=(None, ), chunks=True)
        if num_img_msgs > 0:
            image_dset = events_file.create_group("images")
            image_dset.attrs['num_images'] = num_img_msgs
        if num_flow_msgs > 0:
            flow_dset = events_file.create_group("flow")
            flow_dset.attrs['num_images'] = num_flow_msgs
        num_pos, num_neg, last_ts, img_cnt, flow_cnt = 0, 0, 0, 0, 0

        for topic, msg, t in bag.read_messages():
            if first_ts == -1 and topic in topics:
                timestamp = timestamp_float(msg.header.stamp)
                first_ts = timestamp
                if zero_timestamps:
                    timestamp = timestamp-first_ts
                if start_time is None:
                    start_time = first_ts
                start_time = start_time + first_ts
                if end_time is not None:
                    end_time = end_time+start_time
                if zero_timestamps:
                    timestamp = timestamp-first_ts

            if topic == image_topic:
                timestamp = timestamp_float(msg.header.stamp)-(first_ts if zero_timestamps else 0)
                image = CvBridge().imgmsg_to_cv2(msg, "mono8")
                image_dset = events_file.create_dataset("images/image{:09d}".format(img_cnt), data=image,
                                                        dtype=np.dtype(np.uint8))
                image_dset.attrs['size'] = image.shape
                image_dset.attrs['timestamp'] = timestamp
                last_img_ts = timestamp
                img_cnt += 1

            elif topic == flow_topic:
                timestamp = timestamp_float(msg.header.stamp)-(first_ts if zero_timestamps else 0)

                flow_x = np.array(msg.flow_x)
                flow_y = np.array(msg.flow_y)
                flow_x.shape = (msg.height, msg.width)
                flow_y.shape = (msg.height, msg.width)
                flow_image = np.stack((flow_x, flow_y), axis=0)

                flow_dset = events_file.create_dataset("flow/flow{:09d}".format(flow_cnt), data=flow_image,
                                                       dtype=np.dtype(np.float32))
                flow_dset.attrs['size'] = flow_image.shape
                flow_dset.attrs['timestamp'] = flow_img_ts
                flow_cnt += 1

            elif topic == event_topic:
                event_msg_sum += 1
                if event_msg_sum % num_msgs_between_logs == 0 or event_msg_sum >= num_event_msgs - 1:
                    print('Event messages: {} / {}'.format(event_msg_sum + 1, num_event_msgs))
                for e in msg.events:
                    timestamp = timestamp_float(e.ts)-(first_ts if zero_timestamps else 0)
                    xs.append(e.x)
                    ys.append(e.y)
                    ts.append(timestamp)
                    ps.append(1 if e.polarity else 0)
                    if e.polarity:
                        num_pos += 1
                    else:
                        num_neg += 1
                    last_ts = timestamp
                if (len(xs) > max_buffer_size and timestamp >= start_time) or (end_time is not None and timestamp >= start_time):
                    # print("Writing events")
                    append_to_dataset(event_xs, xs)
                    append_to_dataset(event_ys, ys)
                    append_to_dataset(event_ts, ts)
                    append_to_dataset(event_ps, ps)
                    del xs[:]
                    del ys[:]
                    del ts[:]
                    del ps[:]
                if end_time is not None and timestamp >= start_time:
                    return
                append_to_dataset(event_xs, xs)
                append_to_dataset(event_ys, ys)
                append_to_dataset(event_ts, ts)
                append_to_dataset(event_ps, ps)
                del xs[:]
                del ys[:]
                del ts[:]
                del ps[:]
        events_file.attrs['num_events'] = num_pos+num_neg
        events_file.attrs['num_pos'] = num_pos
        events_file.attrs['num_neg'] = num_neg
        events_file.attrs['duration'] = last_ts-first_ts
        events_file.attrs['t0'] = first_ts
        events_file.attrs['tk'] = last_ts
        events_file.attrs['num_imgs'] = img_cnt
        events_file.attrs['num_flow'] = flow_cnt

def extract_rosbags(rosbag_paths, output_dir, event_topic, image_topic, flow_topic, zero_timestamps=False):
    for path in rosbag_paths:
        bagname = os.path.splitext(os.path.basename(path))[0]
        out_path = os.path.join(output_dir, "{}.h5".format(bagname))
        print("Extracting {} to {}".format(path, out_path))
        extract_rosbag(path, out_path, event_topic, image_topic=image_topic, flow_topic=flow_topic, zero_timestamps=zero_timestamps)

if __name__ == "__main__":
    """
    Tool for converting rosbag events to an efficient HDF5 format that can be speedily
    accessed by python code.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="ROS bag file to extract or directory containing bags")
    parser.add_argument("--output_dir", default="/tmp/extracted_data", help="Folder where to extract the data")
    parser.add_argument("--event_topic", default="/dvs/events", help="Event topic")
    parser.add_argument("--image_topic", default=None, help="Image topic (if left empty, no images will be collected)")
    parser.add_argument("--flow_topic", default=None, help="Flow topic (if left empty, no flow will be collected)")
    parser.add_argument('--zero_timestamps', action='store_true', help='If true, timestamps will be offset to start at 0')
    args = parser.parse_args()

    print('Data will be extracted in folder: {}'.format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if os.path.isdir(args.path):
        rosbag_paths = sorted(glob.glob(os.path.join(args.path, ".bag")))
    else:
        rosbag_paths = [args.path]
    extract_rosbags(rosbag_paths, args.output_dir, args.event_topic, args.image_topic, args.flow_topic, args.zero_timestamps)
