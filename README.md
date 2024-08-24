# DSO-MultiLM
## Multi-Lingual, Multi-Modal Location Estimation

This software package provides a tool for geo-location of image and textual data. The package uses Large Language Models and Retrieval Augmented Generation to provide accurate location estimation. 

## Components

### GeoLlama text geoparsing
 An AI pipeline which provides geoparsing of multi-lingual texts using retrieval augmented generation. The model is current set up to use two fine-tuned Llama-3.1-8b modela for toponym extraction and toponym resolution (GeoLlama-3.1-8b-toponym and GeoLlama-3.1-8b-RAG, respectively), however other LLMs will also work. 

 The model can be run with and without a pre-translation step, which uses an M2M100 model to translate the text  into English before geoparsing. Since Llama-3.1 has native multi-lingual capacity, however, this is not generally required (and not reccomended) unless a specific need for translation is required.Other LLMs with less multi-lingual functionality may require this step to be implemented. 

 Included in the `geo_llama` folder is all the source code required to produce the fine-tuned models using the `unsloth` package, and the data required for fine-tuning and testing. Notebooks detailing the fine-tuning and testing processes are within the `multi-lm/geo_llama/notebooks` folder.


 ### Vision-RAG image geocoding
 A RAG based pipeline which uses a Vision Language Model (VLM) to estimate location from images. The model produces an embedding for a given image, which is then compared to an embedding database of geo-tagged images. The geo-tags of the most similar and most dissimilar images are then extracted and used to help construct the prompt for the VLM. Optionally, text accompanying the image can also be included in the prompt to aid inference. 

 The pipeline is currently structured to use GPT-4 as the VLM, however other models will also work. 

## System Requirements
The GeoLlama model is run locally and requires 12 GB of GPU RAM on a Linux OS. The model requires access to the internet to retrieve information from the OpenStreetMaps API.

The Vision-RAG model requires the use of an Open-AI API key and access to the embeddings dataset and required at least 20GB of system RAM if sing the full 5M embedding database. 

## Installation
The package and all its dependencies can be installed from command line:
```$ python setup.py install```

Please see `requiremnets.txt` for a full list of system requirements.

## Usage
The package uses a Gradio UI. This can be launched by running:
```$ python multi_lm/app.py```

Instructions for using the app are provided on launch. A flagging system has been implemented to store information for when the app fails to predict locations accurately. These are saved in `flagged_images` and `flagged_text` folders. 

## Testing
Tests for the GeoLlama are available and are currently running OK. The Vision-RAG model does not include unittests yet.

## Results

### Text-based geoparsing
 The text-based model has been tested on 47 geo-tagged news articles. The news articles were all published between May and June 2024, and so are outside of the training window for the Llama-3.1 model.

 **Toponym Extraction**
These results are compared against GPT-4o zero-shot and the Edinburgh Geoparser. 

| Metric    | Multi-LM | GPT-4o | Edinburgh |
| ----------|----------|--------|-----------|
| Precision | 0.803    | 0.852  | 0.647     |
| Recall    | 0.736    | 0.717  | 0.641     |
| F1        | 0.744    | 0.779  | 0.644     |

**Toponym Resolution**
We have compared the toponym resolution results against GPT-4o zero-shot, GPT-4o with retrieval augmented generation, and the Edinburgh Geoparser.

| Metric             | Multi-LM | GPT-4o | GPT-4o RAG | Edinburgh |
| ------------------ |----------|--------|------------|-----------|
| Mean distance (KM) | 230.7    | 286.4  | 154.3      | 734.9     |
| Median distance(KM)| 0.01     | 3.483  | 0.006      | 13.490    |
| Accuracy @ 1km     | 0.702    | 0.281  | 0.724      | 0.431     |
| Accuracy @ 161km   | 0.885    | 0.903  | 0.938      | 0.658     |

**Non-English text**
We also tested the model on 15 non-English news articles (5 simplified Chinese, 5 traditional Chinese, 5 French). The toponym extraction and resolution results are provided for the model with and without the pre-translation step. The model generally performs better (and more quickly) without pre-translation. 
 

| Metric    | Without translation | With translation | 
| ----------|---------------------|------------------|
| Precision | 0.733               | 0.648            |
| Recall    | 0.793               | 0.747            |
| F1        | 0.762               | 0.694            |


| Metric             | Without translation | with translation |
| ------------------ |---------------------|------------------|
| Mean distance (KM) | 21.2                | 27.7             |
| Median distance(KM)| 0.034               | 0.034            |
| Accuracy @ 1km     | 0.640               | 0.625            |
| Accuracy @ 161km   | 0.960               | 0.925            |

### Image geocoding
The image geocoding model has been tested against the Img2GPS3K dataset. We have not performed any quantative testing on the multi-modal aspect of the model. The results on the benchmark set are below.

We have tested the framework with the Neva22b VLM and the GPT-4o VLM, and compared to GeoCLIP, Translocator and PIGEOTTO models. 

| Level                    | Neva22b | GPT-4o | GeoCLIP | Translocator |PIGEOTTO|
|--------------------------|---------|--------|---------|--------------|--------|
| Street ($\leq$ 1km)      | 0.058   | 0.106  | 0.141   | 0.118        | 0.113  |
| City ( $\leq$ 25km)      | 0.229   | 0.323  | 0.345   | 0.313        | 0.367  |
| Region ($\leq$ 200km)    | 0.356   | 0.450  | 0.507   | 0.467        | 0.538  |
| Country ($\leq$ 750km)   | 0.539   | 0.609  | 0.697   | 0.589        | 0.724  |
| Continent ($\leq$ 2000km)| 0.724   | 0.771  | 0.838   | 0.801        | 0.853  |

