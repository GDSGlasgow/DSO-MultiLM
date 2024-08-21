# standard library imports
from dataclasses import dataclass, field
from typing import List, Optional
# third party imports
from lxml import etree
# local import

"""Contains all the classes required to handle the training and testing 
articles.
"""

@dataclass
class Toponym:
    """A class to store the attributes associated wiht a toponym object
    
    attributes:
        phrase (str) : the word or phrase, e.g. "New York".
        start (int) : the start index of the toponym in the sentence.
        end (int) : the end index of the toponym in the sentence.
        latitude (float) : the latitude in decimal degrees.
        longitude (float) : the longitude in decimal degrees.
    """
    phrase: str
    start: int
    end: int
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    toponym_type: Optional[str] = None
        
    def __rep__(self):
        return self.phrase  
    
    def to_dict(self):
        try:
            out = {'name':str(self.phrase),
                'latitude':float(self.latitude),
                'longitude':float(self.longitude)
                }
        except TypeError as e:
            out = {'name':str(self.phrase),
                'latitude':None,
                'longitude':None
                }
        return out  

@dataclass
class Article:
    """The object is the base article class. It can be constructed either from a
    data object (article_xml) or by providing the text and toponyms manually.

    Attributes:
        article_xml (Optional[dict]) : The article data in its original format.
        text (Optional[str]) : the text associated with an article.
        toponyms (Optional[List[Toponym]]) : a list of topopnym objects in the article. 
    """
    article_xml: Optional[dict] = None
    text: Optional[str] = None
    toponyms: List[Toponym] = field(default_factory=list)
    
    def __post_init__(self):
        if self.article_xml is not None:
            self.validate_xml()
            self.text = self.extract_text()
            self.toponyms = self.get_toponyms()
    
    def validate_xml(self):
        raise NotImplementedError
    
    def extract_text(self):
        raise NotImplementedError
    
    def get_toponyms(self):
        raise NotImplementedError
    
    def to_dict(self, text_id):
        """Converts the articles and toponyms into a json in the format of the
        expected output of the gpt model.
        """
        out = {"id": str(text_id),
               "toponyms":[]}
        for toponym in self.toponyms:
            toponym_dict = toponym.to_dict()
            out['toponyms'].append(toponym_dict)
        return out
        

@dataclass
class LGLArticle(Article):
    """This inherits from the base Article class and is structured to handle both
    LGL and TR-News articles."""
    def __post_init__(self):
        super().__post_init__() 
        
    def validate_xml(self):
        """Validates the structure of the article XML"""
        if self.article_xml.find('text') is None:
            raise XMLValidationError("XML missing 'text' element")
        if self.article_xml.find('toponyms') is None:
            raise XMLValidationError("XML missing 'toponyms' element") 
           
    def get_toponyms(self):
        """Retrieves the toponyms from the xml article"""
        toponyms = []
        toponyms_xml = self.article_xml.toponyms
        for topo_data in toponyms_xml.iterchildren():
            try:
                toponym = Toponym(phrase=topo_data.phrase,
                                  start=topo_data.start,
                                  end=topo_data.end,
                                  latitude=topo_data.gaztag.lat,
                                  longitude=topo_data.gaztag.lon)
            except Exception as e:
                toponym = Toponym(phrase=topo_data.phrase,
                                  start=topo_data.start,
                                  end=topo_data.end)
            toponyms.append(toponym)
        return toponyms
    
    def extract_text(self):
        return self.article_xml.findtext('text')


@dataclass
class GeoVirusArticle(Article):
    """Inherits from Article and handles GeoVirus articles."""
    def __post_init__(self):
        super().__post_init__() 
        
    def validate_xml(self):
        if self.article_xml.find('text') is None:
            raise ValueError("XML missing 'text'")
        if self.article_xml.find('locations') is None:
            raise ValueError("XML missing 'locations'")
    
    def extract_text(self):
        return self.article_xml.findtext('text')
    
    def get_toponyms(self):
        toponyms = []
        toponyms_xml = self.article_xml.locations[0].location
        for topo_data in toponyms_xml:
            toponym = Toponym(
                phrase=topo_data.name,
                start=int(topo_data.start),
                end=int(topo_data.end),
                latitude=float(topo_data.lat),
                longitude=float(topo_data.lon)
            )
            toponyms.append(toponym)
        return toponyms
    
@dataclass
class News2024Article(Article):
    
    def __post_init__(self):
        super().__post_init__()
        
    def validate_xml(self):
        """This class useses a json object. Maintaining XML name to ensure
        compatibility with super class
        """
        if self.article_xml['text'] is None:
            raise ValueError("Dicitonary is missing 'text'.")
        if self.article_xml['toponyms'] is None:
            raise ValueError("Article does not contain toponyms.")
    
    def extract_text(self):
        return self.article_xml['text']
    
    def get_toponyms(self):
        toponyms = []
        toponyms_dict = self.article_xml.get('toponyms')
        for topo_data in toponyms_dict:
            toponym = Toponym(
                            phrase=str(topo_data['word']),
                            start=int(topo_data['start']),
                            end=int(topo_data['end']),
                            latitude=float(topo_data.get('lat', None)),
                            longitude=float(topo_data.get('lon', None)),
                            toponym_type=str(topo_data.get('type', None))
                            )
            
            toponyms.append(toponym)
        return toponyms
    
    
@dataclass
class WikTorArticle(Article):
    
    def __post_init__(self):
        super().__post_init__()
        
    def validate_xml(self):
        if self.article_xml.find('text') is None:
            raise ValueError("XML missing 'text'")
        if self.article_xml.find('toponymName') is None:
            raise ValueError("XML missing 'toponymName'")
        if self.article_xml.find('lat') is None:
            raise ValueError("XML missing 'lat'")
        if self.article_xml.find('lon') is None:
            raise ValueError("XML missing 'lon'")
        
    def extract_text(self):
        return self.article_xml.findtext('text')
    
    def get_toponyms(self):
        toponyms = []
        for toponym_xml in self.article_xml.toponymIndices:
            toponym = Toponym(phrase = self.article_xml.find('toponymName'),
                              start = toponym_xml.toponym.find('start'),
                              end = toponym_xml.toponym.find('end'),
                              latitude=self.article_xml.find('lat'),
                              longitude=self.article_xml.find('lon'))
            toponyms.append(toponym)
        return toponyms
    
    
class XMLValidationError(Exception):
    def __init__(self, message="XML format invalid."):
        self.message=message
        super().__init__(message)