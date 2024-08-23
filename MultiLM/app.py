# standard library
import sys
sys.path.append('MultiLM/geo_llama/')
import random
# third party
import gradio as gr
from geopy.distance import distance
from geopy.geocoders import Nominatim
# local
from rag_vision.vision_app import process_image
from rag_vision.GPT4o_class import GPT4o
from geo_llama.model import TopoModel, RAGModel
from geo_llama.translator import Translator
from geo_llama.main import GeoLlama
from geo_llama.plotting import plot_map

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

    return processed_text, mapped


def main(image, text, api_key, n_near, n_far, include_text, translate_option):
    """
    Args:
        image (_type_): _description_
        text (_type_): _description_
    """
    # Check if text is to be included in the image location estimation
    context_text = None
    if include_text.lower()=='yes':
        context_text=text
        
    ### process image
    img_coords, img_html = process_image(uploaded_file=image,
                                         openai_api_key=api_key,
                                         num_nearest_neighbors=n_near,
                                         num_farthest_neighbors=n_far, 
                                         context_text=context_text)
    ### process text
    processed_text, text_map = geoparse(text=text,
                                        translation_option=translate_option)
    
    return img_coords, img_html, processed_text, text_map
    
if __name__=='__main__':
    img_geo_locator = GPT4o(device="cpu")
    translator = Translator(model_size='1.2B')
    topo_model = TopoModel(model_name='JoeShingleton/GeoLlama-3.1-8b-toponym', 
                        prompt_path='MultiLM/geo_llama/data/prompt_templates/prompt_template.txt',
                        instruct_path='MultiLM/geo_llama/data/prompt_templates/topo_instruction.txt',
                        input_path=None,
                        config_path='MultiLM/geo_llama/data/config_files/model_config.json')

    rag_model = RAGModel(model_name='JoeShingleton/GeoLlama-3.1-8b-RAG', 
                        prompt_path='MultiLM/geo_llama/data/prompt_templates/prompt_template.txt',
                        instruct_path='MultiLM/geo_llama/data/prompt_templates/rag_instruction.txt',
                        input_path='MultiLM/geo_llama/data/prompt_templates/rag_input.txt',
                        config_path='MultiLM/geo_llama/data/config_files/model_config.json')

    geo_llama = GeoLlama(topo_model, rag_model)
    
    with gr.Blocks() as app:
        with gr.Row():
            with gr.Column():
                
                ### IMAGE INPUT ###
                image_input = gr.Image(label="Upload an image")
                # api key
                openai_api_key = gr.Textbox(label="API Key", placeholder="xxxxxxxxx", type="password")
                # nearest neighbour options
                with gr.Accordion("Advanced Options", open=False):
                    num_nearest_neighbors = gr.Number(label="Number of nearest neighbors", value=16)
                    num_farthest_neighbors = gr.Number(label="Number of farthest neighbors", value=16)
                    include_text = gr.Radio(label='Include text in image inference?',
                                            choices=['Yes', 'No'],
                                            value='No')
                    
                ### TEXT INPUT ###
                text_input = gr.Textbox(label='Text')
                translate_options = gr.Radio(label="Geoparse Mode",
                                             choices=["Without Translation", 
                                                 "With Translation"],
                                            value="Without Translation")



                submit = gr.Button("Submit")

            with gr.Column():
                text_output = gr.Markdown('Highlighted Toponyms')
                text_map = gr.Plot(label='Toponyms mapped')
                
            with gr.column()
                status = gr.Textbox(label="Predicted Location")
                img_outputs = gr.HTML(label="Generated Maps")  # Using HTML for correct map rendering
            
                
        submit.click(
            main,
            inputs=[
                image_input, # image to be processed
                text_input,  # text to be processed
                openai_api_key, # api key
                num_nearest_neighbors, 
                num_farthest_neighbors,
                include_text, # include text in image inference?
                translate_options # include translation?
            ],
            outputs=[status, img_outputs, text_output, text_map]
        )

    app.launch(share=True)