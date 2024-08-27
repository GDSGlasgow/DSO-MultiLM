# standard library imports
import sys
import os
import pathlib

PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)
sys.path.append('multi-lm/')
sys.path.append('.')
# third party imports
import gradio as gr
# local imports
from rag_vision.vision_app import process_image
from rag_vision.GPT4o_class import GPT4o
from geo_llama.model import TopoModel, RAGModel
from geo_llama.translator import Translator
from geo_llama.main import GeoLlama


def main(image, text, api_key, n_near, n_far, include_text, translate_option):
    """
    Args:
        image (_type_): _description_
        text (_type_): _description_
    """
    # Check if text is to be included in the image location estimation
    context_text = None
    if include_text.lower() == 'yes':
        context_text = text

    ### process image
    img_coords, img_html = process_image(uploaded_file=image,
                                         openai_api_key=api_key,
                                         num_nearest_neighbors=n_near,
                                         num_farthest_neighbors=n_far,
                                         context_text=context_text)
    ### process text
    processed_text, text_map = geo_llama.geoparse_pipeline(text=text,
                                                           translation_option=translate_option)

    return img_coords, img_html, processed_text, text_map


if __name__ == '__main__':
    # set some paths
    data_path = pathlib.Path(PROJECT_PATH, 'multi-lm/geo_llama/data/')
    prompt_template_path = pathlib.Path(data_path, 'prompt_templates/prompt_template.txt')
    topo_instruct_path = pathlib.Path(data_path, 'prompt_templates/topo_instruction.txt')
    rag_instruct_path = pathlib.Path(data_path, 'prompt_templates/rag_instruction.txt')
    rag_input_path = pathlib.Path(data_path, 'prompt_templates/rag_input.txt')
    model_config_path = pathlib.Path(data_path, 'config_files/model_config.json')
    # open the app infor file
    with open(pathlib.Path(PROJECT_PATH, 'multi-lm/data/app_info.txt'), 'r') as f:
        app_info = f.read()
    # load geo_llama models
    translator = Translator(model_size='1.2B')
    topo_model = TopoModel(model_name='JoeShingleton/GeoLlama-3.1-8b-toponym',
                           prompt_path=prompt_template_path,
                           instruct_path=topo_instruct_path,
                           input_path=None,
                           config_path=model_config_path)

    rag_model = RAGModel(model_name='JoeShingleton/GeoLlama-3.1-8b-RAG',
                         prompt_path=prompt_template_path,
                         instruct_path=rag_instruct_path,
                         input_path=rag_input_path,
                         config_path=model_config_path)

    geo_llama = GeoLlama(topo_model, rag_model, translator)
    # set up logging files
    img_callback = gr.CSVLogger()
    txt_callback = gr.CSVLogger()

    # build UI
    with gr.Blocks() as app:
        gr.Markdown(app_info)
        with gr.Row():
            with gr.Column():
                ### IMAGE INPUT ###
                image_input = gr.Image(label="Upload an image")
                # api key
                openai_api_key = gr.Textbox(label="API Key", placeholder="xxxxxxxxx", type="password")
                # nearest neighbour options
                with gr.Accordion("Advanced Options", open=False):
                    num_nearest_neighbors = gr.Number(label="Number of similar images", value=16)
                    num_farthest_neighbors = gr.Number(label="Number of dissimilar images", value=16)
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
                # add feedback
                txt_flag_btn = gr.Button("Flag incorrect toponym location")
                txt_callback.setup([text_input, text_output], "flagged_text")
                txt_flag_btn.click(lambda *args: txt_callback.flag(list(args)), [text_input, text_output], None,
                                   preprocess=False)

            with gr.Column():
                status = gr.Textbox(label="Predicted Location")
                img_outputs = gr.HTML(label="Generated Maps")  # Using HTML for correct map rendering
                # add feedback
                img_flag_btn = gr.Button("Flag incorrect image location")
                img_callback.setup([image_input, text_input, img_outputs], "flagged_images")
                img_flag_btn.click(lambda *args: img_callback.flag(list(args)), [image_input, text_input, img_outputs],
                                   None, preprocess=False)

        submit.click(
            main,
            inputs=[
                image_input,  # image to be processed
                text_input,  # text to be processed
                openai_api_key,  # api key
                num_nearest_neighbors,
                num_farthest_neighbors,
                include_text,  # include text in image inference?
                translate_options  # include translation?
            ],
            outputs=[status, img_outputs, text_output, text_map],

        )
    app.launch(share=True)