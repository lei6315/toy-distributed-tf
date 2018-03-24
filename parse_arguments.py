import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer("batch",256,"Batch size")

def main(_):
    print(FLAGS.flag_values_dict())

if __name__ == "__main__":
    tf.app.run()
