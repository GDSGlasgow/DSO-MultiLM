import gradio as gr
import numpy as np
import torch
import folium
import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from io import BytesIO
from rag_vision.GPT4o_class import GPT4o

geo_locator = GPT4o(device="cuda" if torch.cuda.is_available() else "cpu")
# Function to handle the main processing logic
def process_image(uploaded_file, openai_api_key, num_nearest_neighbors, num_farthest_neighbors, context_text=None):
    if not openai_api_key:
        return "Please add your API key to continue.", None

    if uploaded_file is None:
        return "Please upload an image.", None

    # Use the set_image_app method to process the uploaded image
    geo_locator.set_image_app(
        file_uploader=uploaded_file,
        imformat='jpeg',
        use_database_search=True,  # Assuming you want to use the nearest/farthest neighbors
        num_neighbors=num_nearest_neighbors,
        num_farthest=num_farthest_neighbors,
        context_text=context_text
    )

    # Get the location from the OPENAI API
    coordinates = geo_locator.get_location(
        OPENAI_API_KEY=openai_api_key,
        use_database_search=True  # Assuming you want to use the nearest/farthest neighbors
    )

    lat_str, lon_str = coordinates.split(',')
    lat_str = lat_str.strip("() ")
    lon_str = lon_str.strip("() ")
    latitude = float(lat_str)
    longitude = float(lon_str)

    # Generate the prediction map
    prediction_map = folium.Map(location=[latitude, longitude], zoom_start=12)
    folium.Marker([latitude, longitude], tooltip='Img2Loc Location',
                  popup=f'latitude: {latitude}, longitude: {longitude}',
                  icon=folium.Icon(color="red", icon="map-pin", prefix="fa")).add_to(prediction_map)
    folium.TileLayer('cartodbpositron').add_to(prediction_map)

    # Generate the nearest neighbor map
    nearest_map = None
    if geo_locator.neighbor_locations_array:
        nearest_map = folium.Map(location=geo_locator.neighbor_locations_array[0], zoom_start=4)
        folium.TileLayer('cartodbpositron').add_to(nearest_map)
        for i in geo_locator.neighbor_locations_array:
            folium.Marker(i, tooltip=f'({i[0]}, {i[1]})',
                          icon=folium.Icon(color="green", icon="compass", prefix="fa")).add_to(nearest_map)

    # Generate the farthest neighbor map
    farthest_map = None
    if geo_locator.farthest_locations_array:
        farthest_map = folium.Map(location=geo_locator.farthest_locations_array[0], zoom_start=3)
        folium.TileLayer('cartodbpositron').add_to(farthest_map)
        for i in geo_locator.farthest_locations_array:
            folium.Marker(i, tooltip=f'({i[0]}, {i[1]})',
                          icon=folium.Icon(color="blue", icon="compass", prefix="fa")).add_to(farthest_map)

    # Convert maps to HTML representations
    prediction_map_html = map_to_html(prediction_map)
    nearest_map_html = map_to_html(nearest_map) if nearest_map else ""
    farthest_map_html = map_to_html(farthest_map) if farthest_map else ""

    # Create a combined HTML output for Gradio
    combined_html = f"""
    <div style="text-align: center;">
        <h3>Prediction Map</h3>
        {prediction_map_html}
        <div style="display: flex; justify-content: space-between; margin-top: 20px;">
            <div style="flex: 1; margin-right: 10px;">
                <h4>Nearest Neighbor Points Map</h4>
                {nearest_map_html}
            </div>
            <div style="flex: 1; margin-left: 10px;">
                <h4>Farthest Neighbor Points Map</h4>
                {farthest_map_html}
            </div>
        </div>
    </div>
    """

    # Return the coordinates (location information) and the combined HTML with maps
    return coordinates, combined_html


def map_to_html(map_obj):
    """
    Convert a Folium map to an HTML representation.
    """
    return map_obj._repr_html_()

def main():
    # Gradio Interface
    with gr.Blocks() as vision_app:
        with gr.Row():
            with gr.Column():
                uploaded_file = gr.Image(label="Upload an image")
                openai_api_key = gr.Textbox(label="API Key", placeholder="xxxxxxxxx", type="password")

                with gr.Accordion("Advanced Options", open=False):
                    num_nearest_neighbors = gr.Number(label="Number of nearest neighbors", value=16)
                    num_farthest_neighbors = gr.Number(label="Number of farthest neighbors", value=16)

                submit = gr.Button("Submit")

            with gr.Column():
                status = gr.Textbox(label="Predicted Location")
                maps_display = gr.HTML(label="Generated Maps")  # Using HTML for correct map rendering

        submit.click(
            process_image,
            inputs=[
                uploaded_file,
                openai_api_key,
                num_nearest_neighbors,
                num_farthest_neighbors
            ],
            outputs=[status, maps_display]
        )

    vision_app.launch()
    
if __name__ == '__main__':
        
    # Initialize the GPT4v2Loc object
    geo_locator = GPT4o(device="cuda" if torch.cuda.is_available() else "cpu")
    main()


