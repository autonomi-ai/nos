"""Common metaclasses for re-use."""


class SingletonMetaclass(type):
    """Singleton metaclass for instantation of a single instance of a class.

    For example:
        ```python
        class Foo(metaclass=SingletonMetaclass):
            ...
        ```
    """

    _instance = None

    def __call__(cls, *args, **kwargs):
        """Call the class constructor."""
        if cls._instance is None:
            cls._instance = super().__call__(*args, **kwargs)
        return cls._instance
