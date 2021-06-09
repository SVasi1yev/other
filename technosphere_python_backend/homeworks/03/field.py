import logging
import sys
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO
)


class Field:
    def __init__(self, type_, type_name, default_value=None):
        self.type_ = type_
        self.valid(default_value)
        self.type_name = type_name
        self.default_value = default_value
        self.value = default_value

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, obj, objtype=None):
        return obj.__dict__[self.name]

    def __set__(self, obj, value):
        self.valid(value)
        obj.__dict__[self.name] = value

    def valid(self, value):
        if value is None:
            return
        if not isinstance(value, self.type_):
            err_message = f'{type(value)} is not instance of {type(self.type_)}'
            logging.warning(err_message)
            raise TypeError(err_message)


class IntField(Field):
    def __init__(self, default_value):
        super().__init__(int, 'INTEGER', default_value=default_value)


class RealField(Field):
    def __init__(self, default_value):
        super().__init__(float, 'REAL', default_value=default_value)


class TextField(Field):
    def __init__(self, default_value):
        super().__init__(str, 'TEXT', default_value=default_value)
