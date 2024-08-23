from enum import Enum

test1 = dict(type='int', num_values=9)
print(test1["num_values"])


class FooAlgo:
    name = "Foo"


class BarAlgo:
    pass


the_type = FooAlgo
print(type(the_type))
print(the_type.__name__)


class Algo(Enum):
    FOO = FooAlgo
    BAR = BarAlgo

    def __str__(self):
        return self.value.__name__


value = Algo.BAR
print(value)

if not hasattr(FooAlgo, 'name'):
    print("does not Exists")


def func_foo(a, b, c, d=None):
    values = locals()
    i = 5
    print(values)


func_foo(1, 2, 3)
