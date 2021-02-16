from ptolemaios.core import my_cool_test_method1
from ptolemaios.core import my_cool_test_method2
from ptolemaios.core import run_regression
import numpy as np


def test_my_cool_test_method1():
    assert my_cool_test_method1() == "It works!"


def test_my_cool_test_method2():
    assert my_cool_test_method2() == "It works too!"


def test_regression_functionality():
    np.random.seed(101)
    x = np.linspace(0, 50, 50)
    y = np.linspace(0, 50, 50)
    x += np.random.uniform(-4, 4, 50)
    y += np.random.uniform(-4, 4, 50)
    n = len(x)  # Number of data points
    result = run_regression(x, y, n, learning_rate=0.01, training_epochs=1000)
    assert True