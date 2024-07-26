<<<<<<< HEAD
import sys
sys.path.append("../BERT_geoparser")
#import BERT_geoparser
from utils import flatten, convert
import unittest
import numpy as np

class TestFlatten(unittest.TestCase):
    
    def setUp(self):
        pass
    
    def tearDown(self):
        pass
    
    def test_non_list_raises_error(self):
        arg = 'test'
        with self.assertRaises(TypeError) as context:
            flatten(arg)
        exception_msg = "lists should be type list."
        self.assertTrue(exception_msg in str(context.exception))
        
    def test_list_of_non_lists_raises_error(self):
        arg = ['t', 'e', 's', 't']
        with self.assertRaises(TypeError) as context:
            flatten(arg)
        exception_msg = "Component 0 of lists is type str, not type list."
        self.assertTrue(exception_msg in str(context.exception))
        
    def test_returns_flat_list(self):
        arg = [[1,2,3], [1,2,3]]
        out = flatten(arg)
        exp = [1,2,3,1,2,3]
        self.assertListEqual(out, exp)
        

class TestConvert(unittest.TestCase):
    
    def setUp(self):
        pass
    
    def tearDown(self):
        pass
    
    def test_invalid_arg_raises_error(self):
        arg = 3
        with self.assertRaises(TypeError) as context:
            convert(arg)
        exception_msg = "Input should be np.int32"
        self.assertTrue(exception_msg, str(context.exception))
        
    def assert_output_correct_type(self):
        arg = np.int32(3)
        out = convert(arg)
        self.assertTrue(isinstance(out, int))
if __name__ == '__main__':
=======
import sys
sys.path.append("../BERT_geoparser")
#import BERT_geoparser
from utils import flatten, convert
import unittest
import numpy as np

class TestFlatten(unittest.TestCase):
    
    def setUp(self):
        pass
    
    def tearDown(self):
        pass
    
    def test_non_list_raises_error(self):
        arg = 'test'
        with self.assertRaises(TypeError) as context:
            flatten(arg)
        exception_msg = "lists should be type list."
        self.assertTrue(exception_msg in str(context.exception))
        
    def test_list_of_non_lists_raises_error(self):
        arg = ['t', 'e', 's', 't']
        with self.assertRaises(TypeError) as context:
            flatten(arg)
        exception_msg = "Component 0 of lists is type str, not type list."
        self.assertTrue(exception_msg in str(context.exception))
        
    def test_returns_flat_list(self):
        arg = [[1,2,3], [1,2,3]]
        out = flatten(arg)
        exp = [1,2,3,1,2,3]
        self.assertListEqual(out, exp)
        

class TestConvert(unittest.TestCase):
    
    def setUp(self):
        pass
    
    def tearDown(self):
        pass
    
    def test_invalid_arg_raises_error(self):
        arg = 3
        with self.assertRaises(TypeError) as context:
            convert(arg)
        exception_msg = "Input should be np.int32"
        self.assertTrue(exception_msg, str(context.exception))
        
    def assert_output_correct_type(self):
        arg = np.int32(3)
        out = convert(arg)
        self.assertTrue(isinstance(out, int))
if __name__ == '__main__':
>>>>>>> a77f96d0820c52c0cbe32d1d300a3d607231739f
    unittest.main()