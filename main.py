import os
import tensorflow as tf
from input import *
from model import *

try:
     job_name = os.environ['JOB_NAME']
     task_index = os.environ['TASK_INDEX']
     ps_hosts = os.environ['PS_HOSTS']
     worker_hosts = os.environ['WORKER_HOSTS']
     num_workers = len(worker_hosts)
     print(num_workers)
except:
    job_name = None
    task_index = 0
    ps_hosts = None
    worker_hosts = None
    num_workers = 1

flags = tf.app.flags
FLAGS = flags.FLAGS

PATH_TO_LOCAL_LOGS = os.path.expanduser("~/logs/toy-distributed-tf")

environment = os.environ.get('CLUSTERONE_CLOUD') or os.environ.get('TENSORPORT_CLOUD')
environment = 'clusterone-cloud' if environment else "local"

if environment == 'local': #if running locally
    logs = PATH_TO_LOCAL_LOGS
    data_dir = os.path.expanduser("~/data/101_ObjectCategories")
else:
    logs = "/logs"
    data_dir = "/data/malo/malo-caltech-101"
    assert os.path.isdir(data_dir)

flags.DEFINE_string("job_name", job_name,
                    "job name: worker or ps")
flags.DEFINE_integer("task_index", task_index,
                    "Worker task index, should be >= 0. task_index=0 is "
                    "the chief worker task that performs the variable "
                    "initialization")
flags.DEFINE_integer("worker_index", task_index,
                    "Worker index. TODO remove if not necessary. Clarify the difference with task index")
flags.DEFINE_integer("num_workers", num_workers,
                    "Number of workers")
flags.DEFINE_string("ps_hosts", ps_hosts,
                    "Comma-separated list of hostname:port pairs")
flags.DEFINE_string("worker_hosts", worker_hosts,
                    "Comma-separated list of hostname:port pairs")
flags.DEFINE_string("data_dir", data_dir,
                    "Comma-separated list of hostname:port pairs")
flags.DEFINE_integer("batch_size",64,"Size of the batch of data")
flags.DEFINE_float("reg_weight",1e-3,"Regularization weight")
flags.DEFINE_string("run_id",None,"Id of the run, a new folder will be created locally for checkpoints if supplied")
flags.DEFINE_float("lr",1e-6,"Learning rate")

if not FLAGS.run_id == None and job_name == None:
    logs += "/run-" + FLAGS.run_id
    print("Saving checkpoints and outputs at %s" % logs)

def device_and_target():
    # If FLAGS.job_name is not set, we're running single-machine TensorFlow.
    # Don't set a device.
    if FLAGS.job_name is None:
        print("Running single-machine training")
        return (None, "")
    # Otherwise we're running distributed TensorFlow.
    print("Running distributed training")
    if FLAGS.task_index is None or FLAGS.task_index == "":
        raise ValueError("Must specify an explicit `task_index`")
    if FLAGS.ps_hosts is None or FLAGS.ps_hosts == "":
        raise ValueError("Must specify an explicit `ps_hosts`")
    if FLAGS.worker_hosts is None or FLAGS.worker_hosts == "":
        raise ValueError("Must specify an explicit `worker_hosts`")

    cluster_spec = tf.train.ClusterSpec({
            "ps": FLAGS.ps_hosts.split(","),
            "worker": FLAGS.worker_hosts.split(","),
    })
    server = tf.train.Server(
            cluster_spec, job_name=FLAGS.job_name, task_index=FLAGS.task_index)
    if FLAGS.job_name == "ps":
        server.join()

    worker_device = "/job:worker/task:{}".format(FLAGS.task_index)
    # The device setter will automatically place Variables ops on separate
    # parameter servers (ps). The non-Variable ops will be placed on the workers.
    return (
            tf.train.replica_device_setter(
                    worker_device=worker_device,
                    cluster=cluster_spec),
            server.target,
            )

device, target = device_and_target()


def main(_):
    with tf.device(device):
        filelist, labels = get_filelist(FLAGS.data_dir)
        print("Reading from %s image files" % len(filelist))
        encoding, decoding = get_class_encoding(labels)
        print("Loading dataset")
        dataset = construct_dataset(filelist, encode(labels,encoding), FLAGS.batch_size, FLAGS.num_workers,FLAGS.worker_index)
        print("Dataset loaded")
        iterator = dataset.make_one_shot_iterator()

        batch = iterator.get_next()
        img_batch, filepath_batch, label_batch = batch

        num_classes = len(encoding.keys())
        regularizer = tf.contrib.layers.l2_regularizer(scale=FLAGS.reg_weight)

        logits, probs, preds, batch_img = cnn(img_batch, num_classes,regularizer)

        #Apply regularizer
        reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        reg_term = tf.contrib.layers.apply_regularization(regularizer, reg_variables)

        loss = tf.losses.sparse_softmax_cross_entropy(labels=label_batch, logits=logits)
        loss += reg_term

        optimizer = tf.train.GradientDescentOptimizer(learning_rate=FLAGS.lr)
        training_summary = tf.summary.scalar('Training_Loss', loss)#add to tboard

        global_step = tf.train.get_or_create_global_step()

        train_op = optimizer.minimize(
            loss=loss,
            global_step=global_step
            )
        hooks=[tf.train.StopAtStepHook(last_step=1000000)]
        # reshuffle etc

        #TODO : reinitializable dataset.make_initializable_iterator()

    with tf.train.MonitoredTrainingSession(master=target,
        is_chief=(FLAGS.task_index == 0),checkpoint_dir=logs,hooks = hooks,
        save_summaries_steps = 5) as sess:
        #TODO: restore save_summaries_steps default, here it is saving frequently and slowing down training
        try:
            while not sess.should_stop():
                sess.run(train_op)
                loss_val, _ , batch_img_val= sess.run([loss,training_summary,batch_img])
                print(loss_val)
                print(batch_img_val.shape)
                print(sess.should_stop())
        except Exception as e:
            print(e)

    #TODO: add this in the Dataset
    #https://www.tensorflow.org/programmers_guide/datasets
#
# def main_debug(_):
#     filelist, labels = get_filelist(FLAGS.data_dir)
#     encoding, decoding = get_class_encoding(labels)
#     dataset = construct_dataset(filelist, encode(labels,encoding), FLAGS.batch_size, FLAGS.num_workers,FLAGS.worker_index)
#     iterator = dataset.make_one_shot_iterator()
#
#     batch = iterator.get_next()
#     img_batch, filepath_batch, label_batch = batch
#
#     with tf.Session() as sess:
#         for i in range(1):
#             one_img = sess.run(label_batch)
#             print(one_img)
#             print(sum(one_img))
#
#
#     # dataset = tf.data.Dataset.range(6)
#
#     # sum = tf.summary.image(
#     #     name,
#     #     iterator[3],
#     #     max_outputs=3,
#     #     collections=None,
#     #     family=None)
#     #



if __name__ == "__main__":
    tf.app.run(main=main)
