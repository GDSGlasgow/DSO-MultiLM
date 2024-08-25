# standard library imports
from datetime import datetime
import random
import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
sys.path.append('.')
# third party imports
from geopy.distance import distance
from geopy.geocoders import Nominatim
# local imports
from geo_llama.gazetteer import Gazetteer 
from geo_llama.plotting import plot_map

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
                 translate_model,
                 gazetteer_source='nominatim',
                 geonames_username=None):
        """A class to handle the full geoparsing pipeline.

        Args:
            topo_model (geo_llama.model.TopoModel): the TopoModel object.
            rag_model (geo_llama.model.RAGModel): the RAGModel object.
            translate_model (geo_llama.translator.Translator): translation.
            gazetteer_source (str, optional): The API used to find matches. Can 
                be 'nominatim' or 'geonames'. Defaults to 'nominatim'.
            geonames_username (_type_, optional): Required if using geonames as
                a gazetteer source (requires account). Defaults to None.
        """
        self.topo_model = topo_model
        self.rag_model = rag_model
        self.translator = translate_model
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
        #print(toponym)
        raw_matches = self.gazetteer.query(toponym,user_agent)
        #print(raw_matches)
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
        
    def translate(self, text):
        out = self.translator.translate(text, out_lang='en')
        return out


    def translate_name(self, name, coordinates):
        """We can't use the translator for the name because it tends to literally
        translate rather than preserve place names. Instead, we'll look the place up
        in Nominatim and return the english name of the first match.
        
        args:
            name (str) : the name of the toponym in the original language.
            coordinates (tuple[float,float]) : the location predicted by GeoLlama.
        returns:
            str : English name of cloasest location in Nominatim.
        """
        user_id = f'GeoLlama_{random.uniform(1000,10000)}'
        nom = Nominatim(user_agent=f'Geo-Llama_{user_id}')
        matches = nom.geocode(name, language='en', exactly_one=False)
        # get the match which is closest to the provided coordinates
        try:
            best = matches[0]
        except:
            return name + ' (unable to translate place name)'
        best_d = distance((best.latitude, best.longitude), coordinates)
        for m in matches:
            d = distance((m.latitude, m.longitude), coordinates)
            # check if best match
            if d < best_d:
                best = m
                best_d = d
        try:
            return best.address.split(',')[0]
        except IndexError as e:
            return name + ' (unable to translate place name)'


    def geoparse_pipeline(self, text:str, translation_option='With Translation'):
        """Uses the GeoLlama pipeline to geoparse to translate and geoparse the
        proivided text, and provide the output as processed HTML text with a 
        plotly map.
        
        args:
            text (str) : the text to be geoparsed.
            translation_option (str) : either 'With Translation' or 'Without Translation"
        return:
            tuple[str, str, plotly.map]
        """
        # translate text if required
        if translation_option=='With Translation':
            translated_text = self.translate(text)
            processed_text = translated_text['translation']
        else:
            processed_text = text

        # geoparse
        locations = self.geoparse(processed_text)
        # Create an HTML string with highlighted place names and tooltips
        translate_cache = {}
        for loc in locations:
            lat, lon = loc['latitude'], loc['longitude']
            # if the text has been translated, we don't need to translate the name
            if translation_option == 'With translation':
                name = loc['name']
            # if no translation we still want toponyms translated. Check cache first.
            elif loc['name'] in translate_cache.keys():
                name = translate_cache[loc['name']]
            # otherwise use translate_name()
            else:
                name = self.translate_name(loc['name'], (lat, lon))
                translate_cache.update({loc['name']:name})
            # Creating a tooltip for the place name with coordinates
            tooltip_html = f'<span style="background-color: yellow;" title="Toponym: {name} \n Coordinates: ({lat}, {lon})">{loc["name"]}</span>'
            processed_text = processed_text.replace(loc['name'], tooltip_html)

        # Generate the map plot
        mapped = plot_map(locations, translate_cache)

        return processed_text, mapped
        
    

        
        
            
            