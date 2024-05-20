import sys
sys.path.append("../BERT_geoparser")
#import BERT_geoparser
from tokenizer import Tokenizer
import unittest
import pathlib
import shutil

class TestTokenizer(unittest.TestCase):

    def setUp(self):
        # rename any existing tokenizer data as 'old_{path'}
        paths = ['bert-large-cased', 'bert-base-cased', 'bert-base-uncased']
        for path in paths:
            p = pathlib.Path(path)
            if p.exists():
                p.rename(f'old_{p}')

    def test_size_exception(self):
        with self.assertRaises(Exception) as context:
            Tokenizer(size='foo')
        exception_msg = "<size> should be 'base', or 'large'."
        self.assertTrue(exception_msg in str(context.exception))
    
    def test_set_tokenizer_name_size(self):
        t = Tokenizer(size='large', cased=True)
        self.assertEqual(t.tokenizer_name, 'bert-large-cased')
        t = Tokenizer(size='base', cased=True)
        self.assertEqual(t.tokenizer_name, 'bert-base-cased')

    def test_set_tokenizer_name_cased(self):
        t = Tokenizer(size='base', cased=True)
        self.assertEqual(t.tokenizer_name, 'bert-base-cased')
        t = Tokenizer(size='base', cased=False)
        self.assertEqual(t.tokenizer_name, 'bert-base-uncased')
    
    def test_set_save_path_default(self):
        t = Tokenizer(size='base', cased=False,)
        expected = pathlib.Path('bert_base_uncased')
        self.assertEqual(t.save_path, expected)

    def test_set_save_path_non_default(self):
        t = Tokenizer(size='base', cased=False, save_path='testing')
        expected=pathlib.Path('testing')
        self.assertEqual(t.save_path, expected)

    def test_save_tokenizer(self):
        t = Tokenizer(size='base', cased=False, save_path='testing')
        t.save_tokenizer()
        directory = pathlib.Path('testing')
        dir_msg = 'not created expected directory'
        self.assertTrue(directory.exists(), msg=dir_msg)
        file1 = pathlib.Path('testing', 'special_tokens_map.json')
        file1_msg = 'not created special_tokens_map.json'
        self.assertTrue(file1.exists(), msg=file1_msg)
        file2 = pathlib.Path('testing', 'tokenizer_config.json')
        file2_msg = 'not created tokenizer_config.json'
        self.assertTrue(file2.exists(), msg=file2_msg)
        file3 = pathlib.Path('testing', 'vocab.txt')
        file3_msg = 'not created vocab.txt'
        self.assertTrue(file3.exists(), msg=file3_msg)

    def test_get_tokenizer_non_saved(self):
        t = Tokenizer(size='base', cased=False, save_path='testing')
        tokenizer = t.get_tokenizer()
        msg = 'tokenizer failed to load'
        self.assertTrue(tokenizer, msg) 
        msg = 'Did not produce expected directory'
        self.assertTrue(pathlib.Path('testing').exists(), msg)
        
    def tearDown(self):
        # remove the 'testing' path
        if pathlib.Path('testing').exists():
            shutil.rmtree('testing')
        # remove any saved tokenizers and replace with old tokenizers
        paths = ['bert-large-cased', 'bert-base-cased', 'bert-base-uncased']
        for path in paths:
            path = pathlib.Path(path)
            old_path = pathlib.Path(f'old_{path}')
            if (path.exists()) and (old_path.exists()):
                shutil.rmtree(path)
            if old_path.exists():
                pathlib.Path(old_path).rename(path)

if __name__ == '__main__':
    unittest.main()