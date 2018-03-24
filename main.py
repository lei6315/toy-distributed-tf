import os
import tensorflow as tf
from input import *

try:
     job_name = os.environ['JOB_NAME']
     task_index = os.environ['TASK_INDEX']
     ps_hosts = os.environ['PS_HOSTS']
     worker_hosts = os.environ['WORKER_HOSTS']
     num_workers = len(worker_hosts)
except:
    job_name = None
    task_index = 0
    ps_hosts = None
    worker_hosts = None
    num_workers = 1

flags = tf.app.flags
FLAGS = flags.FLAGS

PATH_TO_LOCAL_LOGS = os.path.expanduser("~/logs/toy-distributed-tf")

if job_name == None: #if running locally
    logs = PATH_TO_LOCAL_LOGS
    data_dir = os.path.expanduser("~/data/101_ObjectCategories")
else:
    logs = "/logs"
    data_dir = "/data/malo/malo-caltech-101"

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
    #Model
    a = tf.placeholder(tf.int32)

    # hooks=[tf.train.StopAtStepHook(last_step=100)]
    hooks = []
    with tf.train.MonitoredTrainingSession(master=target,
        is_chief=(FLAGS.task_index == 0),checkpoint_dir=logs,hooks = hooks) as sess:

        for i in range(100):
            print(sess.run(a, feed_dict={a:3}))

def main_2(_):
    filelist = get_filelist(datadir)
    print(filelist)
    dataset = construct_dataset(filelist,FLAGS.num_workers,FLAGS.worker_index)
    # dataset = tf.data.Dataset.range(6)
    iterator = dataset.make_one_shot_iterator()
    res = iterator.get_next()
    with tf.Session() as sess:
            print(sess.run(res))
            print("--------------")
            print("--------------")
            print("--------------")
            print(sess.run(res))
            print("--------------")
            print("--------------")
            print("--------------")
            print(sess.run(res))



if __name__ == "__main__":
    tf.app.run(main=main_2)
