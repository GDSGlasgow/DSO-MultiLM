# standard library imports
from datetime import datetime
# local imports
from geo_llama.gazetteer import Gazetteer 

"""Uses the GeoLlama3-7b-toponym and GeoLlama3-7b-RAG models to extract and
geolocate toponyms from English text.

pipeline:
input text -> GeoLlama3-7b-toponym -> extracted toponyms -> Nominatim lookup 
-> GeoLlama3-7b-RAG model.

For each input text, we run the toponym extraction model once, then the RAG
model one further time for each extracted toponym. Text with a large number of
toponyms may take some time to fully parse.    
"""

class GeoLlama:
    
    def __init__(self, topo_model, rag_model):
        self.topo_model = topo_model
        self.rag_model = rag_model
        self.gazetteer = Gazetteer(polygon=False, addressdetails=False)
        
    def geoparse(self, text:str)->dict:
        """Uses the specified topo_model and rag_model to estimate the location
        of all place names mentioned in the text. Returns a Json formatted
        dictionary.

        Args:
            text (str): The text to be geoparsed.

        Returns:
            dict: A json formatted dictionary with resolved locations. 
        """
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
        prompt = self.topo_model.toponym_prompt(text)
        output = self.topo_model.get_output(prompt['text'], text)
        return output['toponyms']
    
    def get_matches(self, toponym:str):
        user_agent = f'GeoLLama_req_{datetime.now().isoformat()}'
        raw_matches = self.gazetteer.query(toponym,user_agent)
        out = []
        for m in raw_matches:
            out.append({'name':m['name'], 
                        'lat':m['lat'], 
                        'lon':m['lon'], 
                        'address':m['display_name']})
        return out
    
    def get_location(self, toponym:str, text:str, matches:dict):
        rag_prompt = self.rag_model.geoparse_prompt(toponym=toponym, 
                                                    text=text, 
                                                    matches=matches)
        output=self.rag_model.get_output(rag_prompt['text'], text)
        return output
            
        
        
    

        
        
            
            