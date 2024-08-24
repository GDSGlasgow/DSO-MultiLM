import sys
import unittest
sys.path.append('..')

from geo_llama.model import DummyModel
from geo_llama.gazetteer import DummyGazetteer
from geo_llama.main import GeoLlama


class TestGeoLlama(unittest.TestCase):
    
    def setUp(self):
        self.topo_model = DummyModel()
        self.topo_model.type='toponym'
        self.rag_model = DummyModel()
        self.rag_model.type='RAG'
        
        self.geo_llama=GeoLlama(self.topo_model, self.rag_model)
        # set the gazetteer to a dummy for testing
        self.geo_llama.gazetteer = DummyGazetteer()
    
    def tearDown(self):
        pass
    
    def test_initialized(self):
        geo_llama_nom = GeoLlama(self.topo_model, 
                                 self.rag_model, 
                                 gazetteer_source='nominatim')
        geo_llama_gnm = GeoLlama(self.topo_model, 
                                 self.rag_model,
                                 gazetteer_source='geonames')
        self.assertEqual(geo_llama_nom.gazetteer.gazetteer_source, 'nominatim')
        self.assertEqual(geo_llama_gnm.gazetteer.gazetteer_source, 'geonames')
        
    def test_get_matches(self):
        response = self.geo_llama.get_matches(['toponym_1', 'toponym_2'])
        
        expected_1 = {'name':'name_1',
                      'lat':0,
                      'lon':0,
                      'address':'display_name_1'}
        expected_2 = {'name':'name_2',
                      'lat':0,
                      'lon':0,
                      'address':'display_name_2'}
        
        self.assertTrue(len(response)==2)
        self.assertDictEqual(response[0], expected_1)
        self.assertDictEqual(response[1], expected_2)
        
    def test_get_location(self):
        """There isn't much to test here, mainly just that the correct model
        is called. Everything else should be handled by test_model.py"""
        response = self.geo_llama.get_location(toponym='test', text='test', matches='test')
        
        expected = [{'name':'Paris', 'latitude':-1.43, 'longitude':32.48},
                    {'name':'London','latitude':-0.45, 'longitude':48.3},
                    {'name':'New York', 'latitude':-94.34, 'longitue':'24.73'}]
        self.assertTrue(len(response)==len(expected))
        for res, exp in zip(response, expected):
            self.assertDictEqual(res, exp)
            
    def test_get_toponyms(self):
        """Again, we can only really check that the correct model is being used.
        """
        response = self.geo_llama.get_toponyms(text='test')
        expected = ['Paris', 'London', 'New York']
        self.assertCountEqual(response, expected)
        
    
    
        
if __name__=='__main__':
    unittest.main()