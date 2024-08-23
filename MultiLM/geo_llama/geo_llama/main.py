# standard library imports
# standard library imports
from datetime import datetime
import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
sys.path.append('.')
# local imports
from geo_llama.gazetteer import Gazetteer 

"""Uses the GeoLlama3-8b-toponym and GeoLlama3-7b-RAG models to extract and
geolocate toponyms from English text.

pipeline:
input text -> GeoLlama3-8b-toponym -> extracted toponyms -> Nominatim lookup 
-> GeoLlama3-7b-RAG model.

For each input text, we run the toponym extraction model once, then the RAG
model one further time for each extracted toponym. Text with a large number of
toponyms may take some time to fully parse.    
"""

class GeoLlama:
    def __init__(self,
                 topo_model, 
                 rag_model, 
                 gazetteer_source='nominatim',
                 geonames_username=None):
        """A class to handle the full geoparsing pipeline.

        Args:
            topo_model (geo_llama.model.TopoModel): the TopoModel object.
            rag_model (geo_llama.model.RAGModel): the RAGModel object.
            gazetteer_source (str, optional): The API used to find matches. Can 
                be 'nominatim' or 'geonames'. Defaults to 'nominatim'.
            geonames_username (_type_, optional): Required if using geonames as
                a gazetteer source (requires account). Defaults to None.
        """
        self.topo_model = topo_model
        self.rag_model = rag_model
        self.gazetteer = Gazetteer(gazetteer_source=gazetteer_source, 
                                   geonames_username=geonames_username)
        
    def geoparse(self, text:str)->dict:
        """Uses the specified topo_model and rag_model to estimate the location
        of all place names mentioned in the text. Returns a Json formatted
        dictionary.

        Args:
            text (str): The text to be geoparsed.

        Returns:
            dict: A json formatted dictionary with resolved locations. 
        """
        # Check that the input is a string
        if not isinstance(text, str):
            raise TypeError(f'text should be type str not {type(text)}')
        # extract the toponyms
        toponyms = self.get_toponyms(text)
        output = []
        # estimate location foreach toponym
        for toponym in toponyms:
            matches = self.get_matches(toponym)
            location = self.get_location(toponym=toponym, 
                                         text=text, 
                                         matches=matches)
            output.append(location)
        return output
    
    def get_toponyms(self, text:str)->list[str]:
        """Uses the specified topo_model to extract toponyms from the provided
        text. Returns a list of unique toponyms. These are validated to ensure
        all list elements can be found in the text.
        Args:
            text (str) : the text to be analysed.
        Returns:
            list[str] : a list of unique toponyms in the text
        """
        prompt = self.topo_model.toponym_prompt(text=text)
        output = self.topo_model.get_output(prompt=prompt['text'], 
                                            validation_data=text)
        return output['toponyms']
    
    def get_matches(self, toponym:str):
        user_agent = f'GeoLLama_req_{datetime.now().isoformat()}'
        print(toponym)
        raw_matches = self.gazetteer.query(toponym,user_agent)
        print(raw_matches)
        out = []
        for m in raw_matches:
            out.append({'name':m['name'], 
                        'lat':m['lat'], 
                        'lon':m['lon'], 
                        'address':m['display_name']})
        return out
    
    def get_location(self, toponym:str, text:str, matches:list[dict]):
        """Usees the specified RAG_model to identify the best candidate location
        for a toponym in a given peice of text.
        
        args:
            toponym (str) : The toponym to be resolved.
            text (str) : The text in which the toponym occurs.
            matches (list[dict]) : a list of candidate locations from gazetteer.
        """
        rag_prompt = self.rag_model.geoparse_prompt(toponym=toponym, 
                                                    text=text, 
                                                    matches=matches)
        output=self.rag_model.get_output(prompt=rag_prompt['text'],
                                         validation_data=matches)
        return output
        
        
    

        
        
            
            