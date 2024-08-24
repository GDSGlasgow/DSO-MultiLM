import sys
import unittest
sys.path.append('..')
from geo_llama.gazetteer import Gazetteer

class TestGazetteer(unittest.TestCase):
    
    def setUp(self):
        pass
    
    def tearDown(self):
        pass
    
    def test_initialises(self):
        """Tests if the the correct gazetteer is initialised when a given 
        gazetteer_source is passed
        """
        # nominatim
        nom_gaz = Gazetteer(gazetteer_source='nominatim')
        base_url = nom_gaz.base_url
        expected = 'https://nominatim.openstreetmap.org/'
        self.assertEqual(base_url, expected)
        # nominatim
        gnm_gaz = Gazetteer(gazetteer_source='geonames',
                            geonames_username='test_name')
        base_url = gnm_gaz.base_url
        expected = 'http://api.geonames.org/'
        self.assertEqual(base_url, expected)
        self.assertEqual(base_url, expected)
        # initialized with empty cache
        self.assertDictEqual(gnm_gaz.cache, {})
    
    def test_cached_query(self):
        """Tests if the cache is used when required"""
        gaz = Gazetteer(gazetteer_source='nominatim')
        gaz.cache['test_key'] = 'test_value'
        out = gaz.query('test_key')
        expected = 'test_value'
        self.assertEqual(out, expected)
        
    def test_nominatim_user_agent_missing(self):
        """Tests if missing the user_agent on nominatim queries raises an error
        """
        gaz = Gazetteer(gazetteer_source='nominatim')
        # function to reproduce error
        def produce_error():
            gaz.query('test')
        self.assertRaises(ValueError, produce_error)
        
    def test_geonames_username_missing(self):
        """Tests if missing username causes geonames gazetteer to raise error"""
        gaz = Gazetteer(gazetteer_source='geonames')
        def produce_error():
            gaz.query('test')
        self.assertRaises(ValueError, produce_error)
        
    ### TODO: to add Mock APIs to test API calls
    
    def test_format_geonames_response(self):
        gaz = Gazetteer(gazetteer_source='geonames')
        input_response_1 = {'name':'name_1',
                          'adminName1':'state_1',
                          'countryName':'country_1',
                          'lat':0,
                          'lng':0}
        input_response_2 = {'name':'name_2',
                          'countryName':'country_2',
                          'lat':1,
                          'lng':1}
        input_response = [input_response_1, input_response_2]
        
        output = gaz.format_geonames_response(input_response)
        
        expected_1 = {'name':'name_1',
                      'lat':0,
                      'lon':0,
                      'display_name':'name_1, state_1, country_1'}
        expected_2 = {'name':'name_2',
                      'lat':1,
                      'lon':1,
                      'display_name':'name_2, , country_2'}
        
        self.assertTrue(len(output)==2)
        self.assertDictEqual(expected_1, output[0])
        self.assertDictEqual(expected_2, output[1])
    
    
        
if __name__=='__main__':
    unittest.main()
        