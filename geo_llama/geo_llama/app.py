# standard library imports
import random
import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
# third party imports
import gradio as gr
from geopy.distance import distance
from geopy.geocoders import Nominatim
# local imports
from geo_llama.translator import Translator
from geo_llama.model import RAGModel, TopoModel
from geo_llama.main import GeoLlama
from geo_llama.plotting import plot_map

"""This script runs the full geoparsing pipeline using a Gradio web browser
based app. This script should be edited to reflect changes to the model name or
prompt templates.

Currently (19/08/24) there are 2 GeoLlama models available:
    * GeoLlama-3.1-8b-toponym and GeoLlama-3.1-8b-RAG
    * GeoLlama-8b-toponym and GeoLlama-8b-RAG
    
The 3.1 models use the newest version of Llama, the toponym model has also 
received more fine tuning. 
"""

def translate(text):
    out = translator.translate(text, out_lang='en')
    return out


def translate_name(name, coordinates):
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
    best = matches[0]
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


def geoparse(text:str, translation_option='With Translation'):
    """Uses the GeoLlama pipeline to geoparse the provided text.
    
    args:
        text (str) : the text to be geoparsed.
        translation_option (str) : either 'With Translation' or 'Without Translation"
    return:
        tuple[str, str, plotly.map]
    """
    # translate text if required
    if translation_option=='With Translation':
        translated_text = translate(text)
        processed_text = translated_text['translation']
    else:
        processed_text = text

    # geoparse
    locations = geo_llama.geoparse(processed_text)
    locations_str = ', '.join([x['name'] for x in locations])
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
            name = translate_name(loc['name'], (lat, lon))
            translate_cache.update({loc['name']:name})
        # Creating a tooltip for the place name with coordinates
        tooltip_html = f'<span style="background-color: yellow;" title="Toponym: {name} \n Coordinates: ({lat}, {lon})">{loc["name"]}</span>'
        processed_text = processed_text.replace(loc['name'], tooltip_html)

    # Generate the map plot
    mapped = plot_map(locations, translate_cache)

    return processed_text, locations_str, mapped


def main():
    # load the markdown with app info
    with open('data/config_files/app_info.txt', 'r') as f:
        app_info = f.read()
    # set up the gradio inputs and outputs
    input_text = gr.Textbox(label='Text')
    input_options = gr.Radio(
        label="Geoparse Mode",
        choices=["With Translation", "Without Translation"],
        value="With Translation",  # Default option
    )
    # outputs
    output1 = gr.Markdown()
    output2 = gr.Textbox(label='Toponyms')
    output3 = gr.Plot(label='Mapped Locations')
    demo = gr.Interface(fn=geoparse, inputs=[input_text, input_options], outputs=[output1, output2, output3], description=app_info)
    return demo

if __name__=='__main__':
    # specify models. We're using GeoLlama 3.1 here
    translator = Translator(model_size='1.2B')
    topo_model = TopoModel(model_name='JoeShingleton/GeoLlama-3.1-8b-toponym', 
                        prompt_path='data/prompt_templates/prompt_template.txt',
                        instruct_path='data/prompt_templates/topo_instruction.txt',
                        input_path=None,
                        config_path='data/config_files/model_config.json')

    rag_model = RAGModel(model_name='JoeShingleton/GeoLlama-3.1-8b-RAG', 
                        prompt_path='data/prompt_templates/prompt_template.txt',
                        instruct_path='data/prompt_templates/rag_instruction.txt',
                        input_path='data/prompt_templates/rag_input.txt',
                        config_path='data/config_files/model_config.json')

    geo_llama = GeoLlama(topo_model, rag_model)
    demo = main()
    demo.launch(share=True)

