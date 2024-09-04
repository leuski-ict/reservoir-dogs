class Foo(object):
    def __init__(self):
        pass

    @staticmethod
    def get_name():
        return __class__.__name__


class Bar(Foo):
    @staticmethod
    def get_name():
        return "Barrr"


class Car(Foo):
    pass


print(Foo.get_name())
print(Bar.get_name())
print(Car.get_name())


class GameEnvironmentNameDescriptor(object):
    def __get__(self, obj, type_):
        try:
            return type_.__dict__['_name']
        except KeyError:
            return type_.__name__


class Base(object):
    name = GameEnvironmentNameDescriptor()


class Sub(Base):
    pass


class Sub1(Sub):
    _name = "custom_sub"


class Sub2(Sub1):
    pass


print(Sub.name)
print(Sub1.name)
print(Sub2.name)