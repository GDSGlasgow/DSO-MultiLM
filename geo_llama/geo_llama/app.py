# standard library imports
import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
# third party imports
import gradio as gr
# local imports
from .translator import Translator
from .model import RAGModel, TopoModel
from .geo_llama import GeoLlama
from .plotting import plot_map

"""This script runs the full geoparsing pipeline using a Gradio web browser
based app. This script should be edited to reflect changes to the model name or
prompt templates.
"""

def setup():
    translator = Translator(model_size='1.2B')
    topo_model = TopoModel(model_name='JoeShingleton/GeoLlama_7b_toponym', 
                        prompt_path='data/prompt_templates/prompt_template.txt',
                        instruct_path='data/prompt_templates/topo_instruction.txt',
                        input_path=None,
                        config_path='data/config_files/model_config.json')

    rag_model = RAGModel(model_name='JoeShingleton/GeoLlama_7b_RAG', 
                        prompt_path='data/prompt_templates/prompt_template.txt',
                        instruct_path='data/prompt_templates/rag_instruction.txt',
                        input_path='data/prompt_templates/rag_input.txt',
                        config_path='data/config_files/model_config.json')
    
    return translator, topo_model, rag_model

def translate(text):
    out = translator.translate(text, out_lang='en')
    return out

def geoparse(text):
    translated_text = translate(text, translator)
    locations = geo_llama.geoparse(translated_text['translation'])
    locations_str = ', '.join([x['name'] for x in locations])
    mapped = plot_map(locations)
    return translated_text, locations_str, mapped

def main():
    translator, topo_model, rag_model = setup()
    geo_llama = GeoLlama(topo_model, rag_model)
    input = gr.Textbox(label='Text')
    output1 = gr.Textbox(label='Translation')
    output2 = gr.Textbox(label='Toponyms')
    output3 = gr.Plot(label='Mapped Locations')
    demo = gr.Interface(fn=geoparse, 
                        inputs=input, 
                        outputs=[output1, output2, output3])
    demo.launch()

if __name__=='__main__':
    main()
