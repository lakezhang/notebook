#!/usr/bin/env python
# -*- coding:utf-8 -*-

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def lenet_model() :
    inputs = keras.Input(shape=(28, 28, 1), name='input_1')
    c1 = layers.Conv2D(filters=6, kernel_size=(5, 5), padding='valid', activation='relu')(inputs)
    p1 = layers.MaxPool2D(pool_size=(2, 2), padding='valid', data_format='channels_last')(c1)

    c2 = layers.Conv2D(filters=16, kernel_size=(5, 5), padding='valid', activation='relu')(p1)
    p2 = layers.MaxPool2D(pool_size=(2, 2), padding='valid', data_format='channels_last')(c2)

    c3 = layers.Conv2D(filters=120, kernel_size=(1, 1), padding='valid', activation='relu')(p2)

    f1 = layers.Flatten()(c3)
    d1 = layers.Dense(84, activation='relu')(f1)
    outputs = layers.Dense(10, activation='softmax')(d1)

    return keras.Model(inputs = inputs, outputs = outputs, name='lenet')

def mlp_model() :
    inputs = keras.Input(shape=(784,), name='input_1')
    x = layers.Dense(64, activation='relu')(inputs)
    x = layers.Dense(32, activation='relu')(x)
    outputs = layers.Dense(10, activation='softmax')(x)

    return keras.Model(inputs = inputs, outputs = outputs, name='mnist_model')

def model_function(features, labels, mode):
    # get the model
    if FLAGS.model == 'mlp' :
        model = dense_model()
    elif FLAGS.model == 'lenet' :
        model = lenet_model()
    else :
        raise Exception("No this model")

    if mode == tf.estimator.ModeKeys.TRAIN:
        # pass the input through the model
        logits = model(features, training=True)

        # get the cross-entropy loss and name it
        loss = tf.losses.sparse_softmax_cross_entropy(
            labels=labels,
            logits=logits)
        tf.identity(loss, 'train_loss')

        # record the accuracy and name it
        accuracy = tf.metrics.accuracy(
            labels=labels,
            predictions=tf.argmax(logits, axis=1))
        tf.identity(accuracy[1], name='train_accuracy')

        # use Adam to optimize
        optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        # create an estimator spec to optimize the loss
        estimator_spec = tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.TRAIN,
            loss=loss,
            train_op=optimizer.minimize(loss, tf.train.get_or_create_global_step()))

    elif mode == tf.estimator.ModeKeys.EVAL:
        # pass the input through the model
        logits = model(features, training=False)

        # get the cross-entropy loss
        loss = tf.losses.sparse_softmax_cross_entropy(
            labels=labels,
            logits=logits)

        # use the accuracy as a metric
        accuracy = tf.metrics.accuracy(
            labels=labels,
            predictions=tf.argmax(logits, axis=1))

        # create an estimator spec with the loss and accuracy
        estimator_spec = tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.EVAL,
            loss=loss,
            eval_metric_ops={
                'accuracy': accuracy
            })
        
        tf.identity(accuracy[1], name='eval_accuracy')

    return estimator_spec

def get_input_fn() :
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    if FLAGS.model == 'mlp' :
        x_train = x_train.reshape(-1, 784).astype('float32')/255
        x_test= x_test.reshape(-1, 784).astype('float32')/255
    elif FLAGS.model == 'lenet' :
        x_train = x_train.reshape(-1, 28, 28, 1).astype('float32')/255
        x_test= x_test.reshape(-1, 28, 28, 1).astype('float32')/255
    
    y_train = y_train.astype('int32')
    y_test = y_test.astype('int32')

    def _input_fn(features, labels, batch_size):
        dataset = tf.data.Dataset.from_tensor_slices((features, labels))
        return dataset.batch(batch_size)

    return lambda:_input_fn(x_train, y_train,64), lambda:_input_fn(x_test, y_test,2000)

def main(_) :
    hooks_train = [
        tf.train.LoggingTensorHook(
            ['train_accuracy', 'train_loss'],
            every_n_iter=10)
    ]

    hooks_eval = [
        tf.train.LoggingTensorHook(
            ['eval_accuracy'],
            every_n_iter=1)
    ]
    
    mnist_classifier = tf.estimator.Estimator(
        model_fn=model_function, model_dir=FLAGS.model_dir)

    train_input_fn, test_input_fn = get_input_fn()

    for _ in range(FLAGS.epochs):
        mnist_classifier.train(
            input_fn= train_input_fn, 
            hooks=hooks_train)

        mnist_classifier.evaluate(
            input_fn= test_input_fn, 
            hooks=hooks_eval)

if __name__ == '__main__':
    
    flags = tf.app.flags
    flags.DEFINE_string('model', 'lenet',
                        'mlp or lenet')
    flags.DEFINE_string('model_dir', './model/mnist',
                        'Directory where all models are saved')
    flags.DEFINE_integer('batch_size', 100, 'Batch size.')
    flags.DEFINE_integer('epochs', 5,
                         'Num of batches to train (epochs).')
    flags.DEFINE_float('learning_rate', 0.001, 'Learning Rate')
    FLAGS = flags.FLAGS

    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
