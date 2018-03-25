import os
import tensorflow as tf

#TODO: add logger and printout for dataset read ops

FLAGS = tf.app.flags.FLAGS

def get_filelist(data_root):
    """
        Returns a list of tuples (img_path,class)
        :param data_root: root data repository
    """
    filelist = []
    dict = {}
    for subdir, dirs, files in os.walk(data_root):
        for file in files:
            if os.path.splitext(file)[-1] == ".jpg": #filtering .DS_name and non image files
                filelist.append(os.path.join(subdir, file))
                dict[os.path.join(subdir, file)] = subdir.split("/")[-1]
    return filelist, dict

def read_image(img_path):
    ...
    return img #numpy array


def construct_dataset(filenames,num_workers,worker_index):
    """Instantiate a dataset object from filenames.
        :param filenames: a list of tuples (filename, class)
        :param num_workers: total number of workers
        :param worker_index: index of the current worker
    """
    dataset = tf.data.TextLineDataset(filenames)
    dataset = tf.data.Dataset.from_tensor_slices(filenames)
    dataset = dataset.shard(FLAGS.num_workers, FLAGS.worker_index)
    dataset = dataset.shuffle(buffer_size=10000)  # Equivalent to min_after_dequeue=10000.
    dataset = dataset.batch(100)
    return dataset

def test():
    filelist = get_filelist(os.path.expanduser("~/data/101_ObjectCategories"))
    print(filelist[0])
    print(len(filelist))
    dataset = construct_dataset(filelist,1,0)
    dataset.make_initializable_iterator()
