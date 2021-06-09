import unittest
import field
import table


class Users(table.Table):
    table_name = 'users'

    id_ = field.IntField(0)
    name = field.TextField('Petya')


class TestTable(unittest.TestCase):
    def setUp(self):
        for u in Users.all():
            u.delete()

    def test_save(self):
        petya = Users()
        petya.save()
        vasya = Users(id_=100, name='Vasya')
        vasya.save()
        res = Users.all()
        self.assertEqual(res, [Users(id_=0, name='Petya'), Users(id_=100, name='Vasya')])
        self.assertNotEqual(res, [Users(id_=0, name='Petya'), Users(id_=101, name='Vasya')])

    def test_update(self):
        petya = Users()
        petya.save()
        petya.update(id_=1)
        self.assertEqual(petya, Users(id_=1, name='Petya'))
        self.assertEqual(Users.all()[0], Users(id_=1, name='Petya'))

    def test_delete(self):
        petya = Users()
        petya.save()
        vasya = Users(id_=100, name='Vasya')
        vasya.save()
        petya.delete()
        res = Users.all()
        self.assertEqual(1, len(Users.all()))
        self.assertEqual(res, [Users(id_=100, name='Vasya')])

    def test_get(self):
        petya = Users(id_=1, name=None)
        petya.save()
        res = Users.get(id_=1)
        self.assertEqual(res[0], Users(id_=1, name=None))
