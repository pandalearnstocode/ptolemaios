from ptolemaios.core import my_cool_test_method1
from ptolemaios.core import my_cool_test_method2


def test_my_cool_test_method1():
    assert my_cool_test_method1() == "It works!"


def test_my_cool_test_method2():
    assert my_cool_test_method2() == "It works too!"
