import sqlite3
from field import Field
import logging
import sys
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO
)


DB_NAME = 'test.db'


class MetaTable(type):
    def __new__(mcs, name, bases, attrs):
        table = super().__new__(mcs, name, bases, attrs)
        if name != 'Table' and 'table_name' not in table.__dict__:
            err_message = f'Set table name in {name}'
            logging.warning(err_message)
            raise AttributeError(err_message)
        table.scheme = {}
        for k, v in table.__dict__.items():
            if isinstance(v, Field):
                table.scheme[k] = v
        if name != 'Table':
            with sqlite3.connect(DB_NAME) as con:
                cur = con.cursor()
                command = f"""
                    CREATE TABLE IF NOT EXISTS 
                    {table.table_name} 
                    ({', '.join(map(lambda x: x[0] + ' ' + x[1].type_name, table.scheme.items()))});
                """
                logging.info(command)
                cur.execute(command)
                con.commit()
        return table


class Table(metaclass=MetaTable):
    def __init__(self, **kwargs):
        for key in kwargs:
            if key not in self.scheme:
                err_message = f'{key} is not in scheme'
                logging.warning(err_message)
                raise AttributeError(f'{key} is not in scheme')
        for key in self.scheme:
            if key in kwargs:
                setattr(self, key, kwargs[key])
            else:
                setattr(self, key, self.scheme[key].default_value)

    def get_keys_values(self):
        keys = self.scheme.keys()
        values = []
        for k in keys:
            v = getattr(self, k)
            if isinstance(v, str):
                values.append('"' + v + '"')
            elif v is None:
                values.append('NULL')
            else:
                values.append(str(v))
        return keys, values

    def save(self):
        with sqlite3.connect(DB_NAME) as con:
            cur = con.cursor()
            keys, values = self.get_keys_values()
            command = f'''
                INSERT INTO {self.table_name} 
                ({', '.join(keys)}) VALUES 
                ({', '.join(values)});
            '''
            logging.info(command)
            cur.execute(command)
            con.commit()

    def update(self, **kwargs):
        with sqlite3.connect(DB_NAME) as con:
            cur = con.cursor()
            keys, values = self.get_keys_values()
            for key in self.scheme:
                if key in kwargs:
                    setattr(self, key, kwargs[key])
            command = f'''
                UPDATE {self.table_name} SET 
                {','.join(f'{k}={v}' if v != 'NULL' else f'{k} IS NULL' for k, v in kwargs.items())}
                WHERE
                {' AND '.join(f'{k}={v}' if v != 'NULL' else f'{k} IS NULL' for k, v in zip(keys, values))};
            '''
            logging.info(command)
            cur.execute(command)
            con.commit()

    def delete(self):
        with sqlite3.connect(DB_NAME) as con:
            cur = con.cursor()
            keys, values = self.get_keys_values()
            command = f'''
                DELETE FROM {self.table_name} 
                WHERE {' AND '.join(f'{k}={v}' if v != 'NULL' else f'{k} IS NULL' for k, v in zip(keys, values))};
            '''
            logging.info(command)
            cur.execute(command)
            con.commit()

    @classmethod
    def get(cls, **kwargs):
        if len(kwargs) == 0:
            return cls.all()
        else:
            with sqlite3.connect(DB_NAME) as con:
                cur = con.cursor()
                command = f'''
                    SELECT * FROM {cls.table_name} 
                    WHERE {' AND '.join(f'{k}={v}' if v != 'NULL' else f'{k} IS NULL' for k, v in kwargs.items())};
                '''
                logging.info(command)
                cur.execute(command)
                keys = cls.scheme.keys()
                res = []
                for e in cur:
                    args = {k: v for k, v in zip(keys, e)}
                    res.append(cls(**args))
                return res

    @classmethod
    def all(cls):
        with sqlite3.connect(DB_NAME) as con:
            cur = con.cursor()
            command = f'''
                SELECT * FROM {cls.table_name};
            '''
            logging.info(command)
            cur.execute(command)
            keys = cls.scheme.keys()
            res = []
            for e in cur:
                args = {k: v for k, v in zip(keys, e)}
                res.append(cls(**args))
            return res

    def __str__(self):
        keys, values = self.get_keys_values()
        return f"{self.__class__.__name__}({', '.join(f'{k}={v}' for k, v in zip(keys, values))})"

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return self.get_keys_values() == other.get_keys_values()

    def __ne__(self, other):
        return not self.__eq__(other)
