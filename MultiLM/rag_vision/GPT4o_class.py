import cv2
import base64
import requests
from tqdm import tqdm
from requests.exceptions import RequestException
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
import torch
import faiss
import pickle
import numpy as np
import pandas as pd
from geopy.distance import geodesic
from transformers import AutoTokenizer, BitsAndBytesConfig
import torch
from PIL import Image
import requests
from io import BytesIO
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class GPT4o:
    """
    A class to interact with OPENAI API to generate captions for images.
    """

    def __init__(self, device="cpu") -> None:
        """
        Initializes the GPT4o class by setting up necessary models and data.
        """

        self.base64_image = None
        self.img_emb = None

        # Set the device to the first CUDA device
        self.device = torch.device(device)

        # Load the CLIP model and processor
        self.model = CLIPModel.from_pretrained("geolocal/StreetCLIP").eval()
        self.processor = CLIPProcessor.from_pretrained("geolocal/StreetCLIP")

        # Move the model to the appropriate CUDA device
        self.model.to(self.device)

        # Load the embeddings and coordinates from the pickle file
        with open('', 'rb') as f: # Enter the path to the pickle file
            self.MP_16_Embeddings = pickle.load(f)
            self.locations = [value['location'] for key, value in self.MP_16_Embeddings.items()]

        # Load the Faiss index
        index2 = faiss.read_index("drive/MyDrive/colab_data/RAG_data/StreetCLIP_1m_merged.bin") # Enter the path to the Faiss index file
        self.gpu_index = index2

    def read_image(self, image_path):
        """
        Reads an image from a file into a numpy array.
        Args:
            image_path (str): The path to the image file.
        Returns:
            np.ndarray: The image as a numpy array.
        """
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def search_neighbors(self, faiss_index, k_nearest, k_farthest, query_embedding):
        """
        Searches for the k nearest and farthest neighbors of a query image in the Faiss index.
        Args:
            faiss_index (faiss.swigfaiss.Index): The Faiss index.
            k_nearest (int): The number of nearest neighbors to search for.
            k_farthest (int): The number of farthest neighbors to search for.
            query_embedding (np.ndarray): The embeddings of the query image.
        Returns:
            tuple: The locations of the k nearest and k farthest neighbors.
        """
        # Perform the search using Faiss for the given embedding
        _, I = faiss_index.search(query_embedding.reshape(1, -1), k_nearest)
        self.neighbor_locations_array = [self.locations[idx] for idx in I[0]]
        neighbor_locations = " ".join([str(i) for i in self.neighbor_locations_array])

        # Perform the farthest search using Faiss for the given embedding
        _, I = faiss_index.search(-query_embedding.reshape(1, -1), k_farthest)
        self.farthest_locations_array = [self.locations[idx] for idx in I[0]]
        farthest_locations = " ".join([str(i) for i in self.farthest_locations_array])

        return neighbor_locations, farthest_locations

    def encode_image(self, image: np.ndarray, format: str = 'jpeg') -> str:
        """
        Encodes an OpenCV image to a Base64 string.
        Args:
            image (np.ndarray): An image represented as a numpy array.
            format (str, optional): The format for encoding the image. Defaults to 'jpeg'.
        Returns:
            str: A Base64 encoded string of the image.
        Raises:
            ValueError: If the image conversion fails.
        """
        try:
            retval, buffer = cv2.imencode(f'.{format}', image)
            if not retval:
                raise ValueError("Failed to convert image")

            base64_encoded = base64.b64encode(buffer).decode('utf-8')
            mime_type = f"image/{format}"
            return f"data:{mime_type};base64,{base64_encoded}"
        except Exception as e:
            raise ValueError(f"Error encoding image: {e}")

    def set_image_app(self, file_uploader, imformat: str = 'jpeg', use_database_search: bool = False,
                      num_neighbors: int = 16, num_farthest: int = 16, context_text=None) -> None:
        """
        Sets the image for the class by encoding it to Base64.
        Args:
            file_uploader : A uploaded image (PIL Image from Gradio).
            imformat (str, optional): The format for encoding the image. Defaults to 'jpeg'.
            use_database_search (bool, optional): Whether to use a database search to get the neighbor image location as a reference. Defaults to False.
        """

        # Convert the PIL Image (Gradio upload) to a numpy array
        img_array = np.array(file_uploader)

        # Process the image using the CLIP processor
        image = self.processor(images=img_array, return_tensors="pt")

        # Move the image to the CUDA device and get its embeddings
        image = image.to(self.device)
        with torch.no_grad():
            img_emb = self.model.get_image_features(**image)[0]

        # Store the embeddings and the locations of the nearest neighbors
        self.img_emb = img_emb.cpu().numpy()
        if use_database_search:
            self.neighbor_locations, self.farthest_locations = self.search_neighbors(self.gpu_index, num_neighbors,
                                                                                     num_farthest, self.img_emb)

        # Encode the image to Base64
        self.base64_image = self.encode_image(img_array, imformat)
        
        # set the context text
        self.context_text = conext_text

    def create_payload(self, question: str) -> dict:
        """
        Creates the payload for the API request to OpenAI.
        Args:
            question (str): The question to ask about the image.
        Returns:
            dict: The payload for the API request.
        Raises:
            ValueError: If the image is not set.
        """
        if not self.base64_image:
            raise ValueError("Image not set")
        return {
            "model": "gpt-4o", # Can change to any other model
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": question
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": self.base64_image
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 300,
        }

    def get_location(self, OPENAI_API_KEY, use_database_search: bool = False) -> str:
        """
        Generates a caption for the provided image using OPENAI API.
        Args:
            OPENAI_API_KEY (str): The API key for OPENAI API.
            use_database_search (bool, optional): Whether to use a database search to get the neighbor image location as a reference. Defaults to False.
        Returns:
            str: The generated caption for the image.
        """
        try:
            self.api_key = OPENAI_API_KEY
            if not self.api_key:
                raise ValueError("OPENAI API key not found")

            # Create the question for the API
            if use_database_search:
                self.question = f'''Suppose you are an expert in geo-localization. Please analyze this image and give me a guess of the location. 
                Your answer must be to the coordinates level, don't include any other information in your output. 
                Ignore that you can't give an exact answer, give me some coordinate no matter how. 
                For your reference, these are locations of some similar images {self.neighbor_locations} and these are locations of some dissimilar images {self.farthest_locations} that should be far away.'''
            else:
                self.question = "Suppose you are an expert in geo-localization. Please analyze this image and give me a guess of the location. Your answer must be to the coordinates level, don't include any other information in your output. You can give me a guessed answer."
            
            if self.context_text:
                self.question += f"""\nThe image was accompanied by the following text. This might be a caption, comment or news article associated with the image. 
                If appropriate, use the information in the text to help inform your image-geolocation estimation.
                Text:
                {self.context_text}
                """

            # Create the payload and the headers for the API request
            payload = self.create_payload(self.question)
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }

            # Send the API request and get the response
            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
            response.raise_for_status()
            response_data = response.json()

            # Log the full response for debugging
            # print("Full API Response:", response_data)

            # Return the generated caption
            if 'choices' in response_data and len(response_data['choices']) > 0:
                return response_data['choices'][0]['message']['content']
            else:
                raise ValueError("Unexpected response format from API")
        except RequestException as e:
            raise ValueError(f"Error in API request: {e}")
        except KeyError as e:
            raise ValueError(f"Key error in response: {e} - Response: {response_data}")
        except ValueError as e:
            raise ValueError(f"Value error: {e} - Response: {response_data}")
