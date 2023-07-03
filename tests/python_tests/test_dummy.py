import unittest
import numpy as np

class TestDummy(unittest.TestCase):
    def test_upper(self):
        self.assertEqual("foo".upper(), "FOO")

    def test_isupper(self):
        self.assertTrue("FOO".isupper())
        self.assertFalse("Foo".isupper())

    def test_numpy(self):
        x = np.ones((2,2))
        x = 2 * x
        self.assertEqual(x[0,0], 2.0)

if __name__ == '__main__':
    unittest.main()
