import tensorflow as tf


def cnn(batch_img, num_classes,regularizer):
    """
    :param batch_img: a batch of images
    :param num_classes:number of classes
    """

    #TODO - this should be handled upstream when reading images
    batch_img = tf.reshape(batch_img, [-1, 300, 300, 3])
    batch_img = tf.cast(batch_img, tf.float32)
    layer1 = tf.layers.conv2d(
        batch_img,
        32,
        (3,3),
        strides=(1,1),
        padding='valid',
        data_format='channels_last',
        dilation_rate=(1, 1),
        activation=None,
        use_bias=True,
        kernel_initializer=tf.contrib.layers.xavier_initializer(),
        kernel_regularizer = regularizer,
        bias_initializer=tf.zeros_initializer()
        )
    pool1 = tf.layers.max_pooling2d(inputs=layer1, pool_size=[2, 2], strides=2)
    layer2 = tf.layers.conv2d(
        pool1,
        32,
        (3,3),
        strides=(1,1),
        padding='valid',
        data_format='channels_last',
        dilation_rate=(1, 1),
        activation=None,
        use_bias=True,
        kernel_initializer=tf.contrib.layers.xavier_initializer(),
        kernel_regularizer = regularizer,
        bias_initializer=tf.zeros_initializer()
        )
    pool2 = tf.layers.max_pooling2d(inputs=layer2, pool_size=[2, 2], strides=2)
    flat = tf.reshape(pool2, [-1, 73 * 73 * 32])
    dense1 = tf.layers.dense(inputs=flat, units=1000, activation=tf.nn.relu,kernel_regularizer=regularizer)
    dense2 = tf.layers.dense(inputs=dense1, units=num_classes, activation=tf.nn.relu,kernel_regularizer=regularizer)
    logits = dense2
    probabilities = tf.nn.softmax(logits, name="softmax_tensor")
    pred_classes = tf.argmax(input=logits, axis=1)

    return(logits, probabilities, pred_classes, batch_img)
