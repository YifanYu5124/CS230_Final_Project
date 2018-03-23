"""Define the model."""

import tensorflow as tf
import numpy as np
import collections
# https://github.com/kratzert/finetune_alexnet_with_tensorflow/blob/master/alexnet.py
def convLayer(x, kHeight, kWidth, strideX, strideY, featureNum, name, padding, weights, biases):
    with tf.variable_scope(name) as scope:
        conv = tf.nn.conv2d(x, weights, [1, strideX, strideY, 1], padding = padding)
        bias = tf.nn.bias_add(conv, biases)
        relu = tf.nn.relu(bias)
        return relu

def build_model(is_training, inputs, params):
    """Compute logits of the model (output distribution)

    Args:
        is_training: (bool) whether we are training or not
        inputs: (dict) contains the inputs of the graph (features, labels...)
                this can be `tf.placeholder` or outputs of `tf.data`
        params: (Params) hyperparameters

    Returns:
        output: (tf.Tensor) output of the model
    """
    images = inputs['images']

    assert images.get_shape().as_list() == [None, params.image_size, params.image_size, 3]


    # TO DO : vgg input for images are 224 * 224 * 3

    # VGG RGB -> BGR
    VGG_MEAN = [103.939, 116.779, 123.68]
    red, green, blue = tf.split(axis=3, num_or_size_splits=3, value= images)
    out = tf.concat(axis=3, values=[blue - VGG_MEAN[0], green - VGG_MEAN[1],red - VGG_MEAN[2]])


    ##### modification - Add VGG  ########

    # TO DO : get the vgg pretrained weight and name it vgg_weight.npy, name of weight in the file 
    weights = np.load("vgg_weights.npy", encoding = 'latin1')
    pretrainedWeights = weights.item()
    trainedWeights = collections.defaultdict()

    num_conv_layers = 5 # pls change later
    num_repeat_times_1 = 2
    num_repeat_times_2 = 3
    repear_layers1 = [1,2]
    repear_layers2 = [3,4,5]

    for i in range(1, num_conv_layers + 1):
        if i in repear_layers1:
            for j in range(1,num_repeat_times_1 + 1):
                trainedWeights['conv' + str(i) + '_' + str(j) + '_weights'] = tf.get_variable('conv' + str(i) + '_' + str(j) +  '_weights', initializer=pretrainedWeights['conv' + str(i) + "_" + str(j) ][0], trainable = False)
                trainedWeights['conv' + str(i) + '_' + str(j) + '_biases'] = tf.get_variable('conv' + str(i) + '_' + str(j)  + '_biases', initializer=pretrainedWeights['conv' + str(i) + "_" + str(j) ][1], trainable = False)
        if i in repear_layers2:
            for j in range(1,num_repeat_times_2 + 1):
                trainedWeights['conv' + str(i) + '_' + str(j) + '_weights'] = tf.get_variable('conv' + str(i) + '_' + str(j) +  '_weights', initializer=pretrainedWeights['conv' + str(i) + "_" + str(j) ][0], trainable = False)
                trainedWeights['conv' + str(i) + '_' + str(j) + '_biases'] = tf.get_variable('conv' + str(i) + '_' + str(j)  + '_biases', initializer=pretrainedWeights['conv' + str(i) + "_" + str(j) ][1], trainable = False)

    num_fc_layers = 2 # pls change later
    for i in range(1, num_fc_layers + 1):
        trainedWeights['fc' + str(i) + '_weights'] = tf.get_variable('fc' + str(i) + '_weights', initializer=pretrainedWeights['fc' + str(i + num_conv_layers)][0], trainable = False) # i + num_conv_layers might not hold for other weights file
        trainedWeights['fc' + str(i) + '_biases'] = tf.get_variable('fc' + str(i) + '_biases', initializer=pretrainedWeights['fc' + str(i + num_conv_layers)][1], trainable = False) # i + num_conv_layers might not hold for other weights file


    # block 1 -- outputs 112x112x64
    # conv1_1
    k = 3; c = 64; s_h = 1; s_w = 1
    conv1_1_out = convLayer(out, k, k, s_h, s_w, c, "conv1_1", "SAME", trainedWeights["conv1_1_weights"], trainedWeights["conv1_1_biases"])

    # conv1_2
    k = 3; c = 64; s_h = 1; s_w = 1
    conv1_2_out = convLayer(conv1_1_out, k, k, s_h, s_w, c, "conv1_2", "SAME", trainedWeights["conv1_2_weights"], trainedWeights["conv1_2_biases"])
    
    # maxpool1
    k_h = 2; k_w = 2; s_h = 2; s_w = 2
    padding = 'SAME'
    maxpool1 = tf.nn.max_pool(conv1_2_out, ksize=[1,  k_h, k_w , 1],strides=[1, s_h, s_w, 1], padding=padding)

    #block 2 -- outputs 56x56x128
    # conv2_1
    k = 3; c = 128; s_h = 1; s_w = 1
    conv2_1_out = convLayer(maxpool1, k, k, s_h, s_w, c, "conv2_1", "SAME", trainedWeights["conv2_1_weights"], trainedWeights["conv2_1_biases"])

    # conv2_2
    k = 3; c = 64; s_h = 1; s_w = 1
    conv2_2_out = convLayer(conv2_1_out, k, k, s_h, s_w, c, "conv2_2", "SAME", trainedWeights["conv2_2_weights"], trainedWeights["conv2_2_biases"])
    
    # maxpool2
    k_h = 2; k_w = 2; s_h = 2; s_w = 2
    padding = 'SAME'
    maxpool2 = tf.nn.max_pool(conv2_2_out, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)
    
    # block 3 -- outputs 28x28x256
    # conv3_1
    k = 3; c = 256; s_h = 1; s_w = 1
    conv3_1_out = convLayer(maxpool2, k, k, s_h, s_w, c, "conv3_1", "SAME", trainedWeights["conv3_1_weights"], trainedWeights["conv3_1_biases"])

    # conv3_2
    k = 3; c = 256; s_h = 1; s_w = 1
    conv3_2_out = convLayer(conv3_1_out, k, k, s_h, s_w, c, "conv3_2", "SAME", trainedWeights["conv3_2_weights"], trainedWeights["conv3_2_biases"])
    
    # conv3_3
    k = 3; c = 256; s_h = 1; s_w = 1
    conv3_3_out = convLayer(conv3_2_out, k, k, s_h, s_w, c, "conv3_3", "SAME", trainedWeights["conv3_3_weights"], trainedWeights["conv3_3_biases"])

    # maxpool3
    k_h = 2; k_w = 2; s_h = 2; s_w = 2
    padding = 'SAME'
    maxpool3 = tf.nn.max_pool(conv3_3_out, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

    # block 4 -- outputs 14x14x512
    # COV 4 
    # conv4_1
    k = 3; c = 512; s_h = 1; s_w = 1
    conv4_1_out = convLayer(maxpool3, k, k, s_h, s_w, c, "conv4_1", "SAME", trainedWeights["conv4_1_weights"], trainedWeights["conv4_1_biases"])

    # conv4_2
    k = 3; c = 512; s_h = 1; s_w = 1
    conv4_2_out = convLayer(conv4_1_out, k, k, s_h, s_w, c, "conv4_2", "SAME", trainedWeights["conv4_2_weights"], trainedWeights["conv4_2_biases"])
    
    # conv4_3
    k = 3; c = 512; s_h = 1; s_w = 1
    conv4_3_out = convLayer(conv4_2_out, k, k, s_h, s_w, c, "conv4_3", "SAME", trainedWeights["conv4_3_weights"], trainedWeights["conv4_3_biases"])

    # maxpool4
    k_h = 2; k_w = 2; s_h = 2; s_w = 2
    padding = 'SAME'
    maxpool4 = tf.nn.max_pool(conv4_3_out, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

    # block 5 -- outputs 7x7x512
    # COV 5
    # conv5_1
    k = 3; c = 512; s_h = 1; s_w = 1
    conv5_1_out = convLayer(maxpool4, k, k, s_h, s_w, c, "conv5_1", "SAME", trainedWeights["conv5_1_weights"], trainedWeights["conv5_1_biases"])

    # conv5_2
    k = 3; c = 512; s_h = 1; s_w = 1
    conv5_2_out = convLayer(conv5_1_out, k, k, s_h, s_w, c, "conv5_2", "SAME", trainedWeights["conv5_2_weights"], trainedWeights["conv5_2_biases"])
    
    # conv5_3
    k = 3; c = 512; s_h = 1; s_w = 1
#     conv5_3_out = convLayer(conv5_2_out, k, k, s_h, s_w, c, "conv5_3", "SAME", trainedWeights["conv5_3_weights"], trainedWeights["conv5_3_biases"])
    conv5_3_out = tf.layers.conv2d(conv5_2_out, filters=c, kernel_size=k, strides=(s_h, s_w), padding='same')

    # maxpool5
    k_h = 2; k_w = 2; s_h = 2; s_w = 2
    padding = 'SAME'
    maxpool5 = tf.nn.max_pool(conv5_3_out, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)
    
    # fullly connected layer
    fc1_in =  tf.reshape(maxpool5, [-1, 7*7*512])

    with tf.variable_scope('fc_1'):
        # w = fc1_weights, shape = [4096, 4096], dtype = "float")
        # b = tf.get_variable(fc1_biases, shape = [4096], dtype = "float")
#         out = tf.nn.xw_plus_b(fc1_in,trainedWeights["fc1_weights"], trainedWeights["fc1_biases"])
        out = tf.layers.dense(fc1_in,4096)
        fc2_in = tf.nn.relu(out)

    with tf.variable_scope('fc_2'):
        #w = tf.get_variable(fc2_weights, shape = [4096, 4096], dtype = "float")
        #b = tf.get_variable(fc2_biases, shape = [4096], dtype = "float")
#         out = tf.nn.xw_plus_b(fc2_in, trainedWeights["fc2_weights"], trainedWeights["fc2_biases"])
        out = tf.layers.dense(fc2_in,4096)
        fc3_in = tf.nn.relu(out)

    ### trainable layer #####
    with tf.variable_scope('fc_3'):
        logits = tf.layers.dense(fc3_in, params.num_labels)
    return logits


    ########### END ##########

    # assert out.get_shape().as_list() == [None, 4, 4, num_channels * 8]
    #
    # out = tf.reshape(out, [-1, 4 * 4 * num_channels * 8])
    # with tf.variable_scope('fc_1'):
    #     out = tf.layers.dense(out, num_channels * 8)
    #     if params.use_batch_norm:
    #         out = tf.layers.batch_normalization(out, momentum=bn_momentum, training=is_training)
    #     out = tf.nn.relu(out)
    # with tf.variable_scope('fc_2'):
    #     logits = tf.layers.dense(out, params.num_labels)
    #
    # return logits
    #

def model_fn(mode, inputs, params, reuse=False):
    """Model function defining the graph operations.

    Args:
        mode: (string) can be 'train' or 'eval'
        inputs: (dict) contains the inputs of the graph (features, labels...)
                this can be `tf.placeholder` or outputs of `tf.data`
        params: (Params) contains hyperparameters of the model (ex: `params.learning_rate`)
        reuse: (bool) whether to reuse the weights

    Returns:
        model_spec: (dict) contains the graph operations or nodes needed for training / evaluation
    """
    is_training = (mode == 'train')
    labels = inputs['labels']
    labels = tf.cast(labels, tf.int64)

    # -----------------------------------------------------------
    # MODEL: define the layers of the model
    with tf.variable_scope('model', reuse=reuse):
        # Compute the output distribution of the model and the predictions
        logits = build_model(is_training, inputs, params)
        predictions = tf.argmax(logits, 1)

    # Define loss and accuracy
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(labels, predictions), tf.float32))

    # Define training step that minimizes the loss with the Adam optimizer
    if is_training:
        optimizer = tf.train.AdamOptimizer(params.learning_rate)
        global_step = tf.train.get_or_create_global_step()
        if params.use_batch_norm:
            # Add a dependency to update the moving mean and variance for batch normalization
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                train_op = optimizer.minimize(loss, global_step=global_step)
        else:
            train_op = optimizer.minimize(loss, global_step=global_step)


    # -----------------------------------------------------------
    # METRICS AND SUMMARIES
    # Metrics for evaluation using tf.metrics (average over whole dataset)
    with tf.variable_scope("metrics"):
        metrics = {
            'accuracy': tf.metrics.accuracy(labels=labels, predictions=tf.argmax(logits, 1)),
            'loss': tf.metrics.mean(loss)
        }

    # Group the update ops for the tf.metrics
    update_metrics_op = tf.group(*[op for _, op in metrics.values()])

    # Get the op to reset the local variables used in tf.metrics
    metric_variables = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="metrics")
    metrics_init_op = tf.variables_initializer(metric_variables)

    # Summaries for training
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('accuracy', accuracy)
    tf.summary.image('train_image', inputs['images'])

    #TODO: if mode == 'eval': ?
    # Add incorrectly labeled images
    mask = tf.not_equal(labels, predictions)

    # Add a different summary to know how they were misclassified
    for label in range(0, params.num_labels):
        mask_label = tf.logical_and(mask, tf.equal(predictions, label))
        incorrect_image_label = tf.boolean_mask(inputs['images'], mask_label)
        tf.summary.image('incorrectly_labeled_{}'.format(label), incorrect_image_label)

    # -----------------------------------------------------------
    # MODEL SPECIFICATION
    # Create the model specification and return it
    # It contains nodes or operations in the graph that will be used for training and evaluation
    model_spec = inputs
    model_spec['variable_init_op'] = tf.global_variables_initializer()
    model_spec["predictions"] = predictions
    model_spec['loss'] = loss
    model_spec['accuracy'] = accuracy
    model_spec['metrics_init_op'] = metrics_init_op
    model_spec['metrics'] = metrics
    model_spec['update_metrics'] = update_metrics_op
    model_spec['summary_op'] = tf.summary.merge_all()

    if is_training:
        model_spec['train_op'] = train_op

    return model_spec
