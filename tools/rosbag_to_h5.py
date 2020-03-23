import argparse
import rosbag
import rospy
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

#Inspired by https://github.com/uzh-rpg/rpg_e2vid
def extract_rosbag(rosbag_path, output_path, event_topic, start_time=None, end_time=None):
    event_sum = 0
    event_msg_sum = 0
    num_msgs_between_logs = 25
    first_ts = -1
    if not os.path.exists(rosbag_path):
        print("{} does not exist!".format(rosbag_path))
        return
    with rosbag.Bag(rosbag_path, 'r') as bag:
        # Look for the topics that are available and save the total number of messages for each topic (useful for the progress bar)
        total_num_event_msgs = 0
        topics = bag.get_type_and_topic_info().topics
        for topic_name, topic_info in topics.iteritems():
            if topic_name == args.event_topic:
                total_num_event_msgs = topic_info.message_count
                print('Found events topic: {} with {} messages'.format(topic_name, topic_info.message_count))

        # Extract events to h5
        xs, ys, ts, ps = [], [], [], []
        max_buffer_size = 1000000
        events_file = h5py.File(output_path, 'w')
        event_xs = events_file.create_dataset("events/x", (0, ), dtype=np.dtype(np.int16), maxshape=(None, ), chunks=True)
        event_ys = events_file.create_dataset("events/y", (0, ), dtype=np.dtype(np.int16), maxshape=(None, ), chunks=True)
        event_ts = events_file.create_dataset("events/ts", (0, ), dtype=np.dtype(np.float64), maxshape=(None, ), chunks=True)
        event_ps = events_file.create_dataset("events/p", (0, ), dtype=np.dtype(np.bool_), maxshape=(None, ), chunks=True)

        for topic, msg, t in bag.read_messages():
            if topic == event_topic:
                event_msg_sum += 1
                if event_msg_sum % num_msgs_between_logs == 0 or event_msg_sum >= total_num_event_msgs - 1:
                    print('Event messages: {} / {}'.format(event_msg_sum + 1, total_num_event_msgs))
                for e in msg.events:
                    timestamp = timestamp_float(e.ts)
                    if first_ts == -1:
                        first_ts = timestamp
                        if start_time is None:
                            start_time = first_ts
                        start_time = start_time + first_ts
                        if end_time is not None:
                            end_time = end_time+start_time
                    xs.append(e.x)
                    ys.append(e.y)
                    ts.append(timestamp)
                    ps.append(1 if e.polarity else 0)
                    event_sum += 1
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


def extract_rosbags(rosbag_paths, output_dir, event_topic):
    for path in rosbag_paths:
        bagname = os.path.splitext(os.path.basename(path))[0]
        out_path = os.path.join(output_dir, "{}.h5".format(bagname))
        print("Extracting {} to {}".format(path, out_path))
        extract_rosbag(path, out_path, event_topic)

if __name__ == "__main__":
    """
    Tool for converting rosbag events to an efficient HDF5 format that can be speedily
    accessed by python code.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="ROS bag file to extract or directory containing bags")
    parser.add_argument("--output_dir", default="/tmp/extracted_data", help="Folder where to extract the data")
    parser.add_argument("--event_topic", default="/dvs/events", help="Event topic")
    args = parser.parse_args()

    print('Data will be extracted in folder: {}'.format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if os.path.isdir(args.path):
        rosbag_paths = sorted(glob.glob(os.path.join(args.path, ".bag")))
    else:
        rosbag_paths = [args.path]
    extract_rosbags(rosbag_paths, args.output_dir, args.event_topic)
