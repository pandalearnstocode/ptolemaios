import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
import matplotlib.pyplot as plt
from loguru import logger


def debug_only(record):
    return record["level"].name == "CRITICAL"


logger.add("critical.log", rotation="12:00", filter=debug_only)


def my_cool_test_method1():
    return "It works!"


def my_cool_test_method2():
    return "It works too!"


def run_regression(x, y, n, learning_rate=0.01, training_epochs=1000):
    X = tf.placeholder("float")
    Y = tf.placeholder("float")
    W = tf.Variable(np.random.randn(), name="W")
    b = tf.Variable(np.random.randn(), name="b")
    y_pred = tf.add(tf.multiply(X, W), b)
    cost = tf.reduce_sum(tf.pow(y_pred - Y, 2)) / (2 * n)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(training_epochs):
            for (_x, _y) in zip(x, y):
                sess.run(optimizer, feed_dict={X: _x, Y: _y})
            if (epoch + 1) % 50 == 0:
                c = sess.run(cost, feed_dict={X: x, Y: y})
                msg = (
                    "Epoch",
                    (epoch + 1),
                    ": cost =",
                    c,
                    "W =",
                    sess.run(W),
                    "b =",
                    sess.run(b),
                )
                logger.info(f"{msg}")

        training_cost = sess.run(cost, feed_dict={X: x, Y: y})
        weight = sess.run(W)
        bias = sess.run(b)
    predictions = weight * x + bias
    result = (
        "Y = "
        + str(training_cost)
        + " + Weight * "
        + str(weight)
        + " + bias * "
        + str(bias)
    )
    response = {"output": result, "prediction": list(predictions.round(4))}
    return response
