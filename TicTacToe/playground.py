test1 = dict(type='int', num_values=9)
print(test1["num_values"])

class Foo:
    pass


the_type = Foo
print(type(the_type))
print(the_type.__name__)