import os
import tensorflow as tf

#TODO: add logger and printout for dataset read ops


def get_filelist(data_root):
    """
        Returns a list of img_path and a list of classes
        :param data_root: root data repository
    """
    filelist = []
    labels = []
    # dict = {}
    for subdir, dirs, files in os.walk(data_root):
        for file in files:
            if os.path.splitext(file)[-1] == ".jpg": #filtering .DS_name and non image files
                filelist.append(os.path.join(subdir, file))
                labels.append(subdir.split("/")[-1])
                # dict[os.path.join(subdir, file)] = subdir.split("/")[-1]
    return filelist, labels


def get_class_encoding(classes):
    """Sort classes in alphabetical order so that the encoding is always the same,
    then returns a dictionnary {class : id (int)}
    :param classes: list of all the classes
    #1 unique()
    #2 sort classes in alphabetical
    #3 construct dict
    """
    classes = sorted(classes) #phew! do not sort in place!
    classes = list(set(classes)) #extract unique elements
    encoding = {}
    decoding = {}
    i = 0
    for c in classes:
        encoding[c] = i
        decoding[i] = c
        i += 1
    return encoding, decoding

def encode(labels, encoding):
    return(map(lambda x: encoding[x], labels))

def _parse_function(filepath, label):
    """
    Read an image path into a tensor
    :param filepath: path to an image
    :param label: label of the image
    """
    image_string = tf.read_file(filepath)
    image_decoded = tf.image.decode_image(image_string, channels = 3)
    image_resized = tf.image.resize_image_with_crop_or_pad(image_decoded, 300, 300)
    #TODO Force image to be (300,300,3). May be done with decode_image. -> done throuch channels = 3
    return image_resized, filepath, label

def construct_dataset(filenames, labels, batch_size, num_workers,worker_index):
    """Instantiate a dataset object from filenames. Data will not be sharded so
    workers will see the same batch multiple times. This will avergae out across epochs.
        Returns a tensor of (image, filepath, label)
        :param filenames: a list of image file path
        :param labels: the associated labels
        :param num_workers: total number of workers
        :param worker_index: index of the current worker
    """
    dataset = tf.data.TextLineDataset(filenames)
    dataset = tf.data.Dataset.from_tensor_slices((filenames,labels))
    #dataset = dataset.shard(num_workers, worker_index)
    dataset = dataset.shuffle(buffer_size=10000)  # Equivalent to min_after_dequeue=10000.
    dataset = dataset.map(_parse_function)
    #TODO - careful here, we are reading the same data at each epoch
    dataset = dataset.batch(batch_size)
    return dataset


def test():
    filelist, labels = get_filelist(os.path.expanduser("~/data/101_ObjectCategories"))
    encoding, decoding = get_class_encoding(labels)
    #this is not optimised as the below ops will be ran only on master. Find a way to have it
    #done on each worker
    print(labels[:1000])
    labels = encode(labels, encoding)
    print(filelist[0])
    print(labels[:1000])

    dataset = construct_dataset(filelist,labels,5,1,0)
    a = dataset.make_initializable_iterator()
    with tf.Session() as sess:
        print(sess.run(a.get_next()))


if __name__ == "__main__":
    test()
