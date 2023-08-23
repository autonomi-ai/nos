from nos.common.metaclass import SingletonMetaclass


def test_singleton_metaclass():
    class Foo(metaclass=SingletonMetaclass):
        pass

    foo1 = Foo()
    foo2 = Foo()
    assert foo1 is foo2
