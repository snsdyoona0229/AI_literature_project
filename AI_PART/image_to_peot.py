import cv2
import torch
from PIL import Image
import matplotlib.pyplot as plt
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from transformers import BertTokenizer, GPT2LMHeadModel, TextGenerationPipeline,AutoTokenizer,BertTokenizerFast
import os
import sys
import shutil
from subprocess import call
import webbrowser
sys.path.append('/literature_project/AI_PART')
import geration_poet
import peot_Classification_model


import numpy as np
import cv2
from matplotlib import pyplot as plt
import googletrans


# Get Model
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
# Get Image Feature Extractor
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
# Get Tokenizer Model
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# Apply model on GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# The maximum length the generated tokens can have
max_length = 16
#  Number of beams for beam search
num_beams = 4
# Generation Config
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

def predict_step(image_paths):
    images = []
    for image_path in image_paths:
        img = Image.open(image_path)
        if img.mode != "RGB":
            img = img.convert(mode="RGB")

        images.append(img)

    # Feature Extractor
    pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    # Apply model
    output_ids = model.generate(pixel_values, **gen_kwargs)

    # Get text tokens
    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    return preds


def image_to_text_gpt(image,category):
    poet_style = ["創世紀詩詞風格", "原住民詩詞風格", "客家詩詞風格", "新月詩詞風格","新詩詩詞風格","現代詩詞風格","笠詩詞風格","藍星詩詞風格"]
    translator = googletrans.Translator()
    translation = translator.translate(predict_step([image])[0], dest='zh-tw')
    context = geration_poet.peot_generation(str(category),translation.text)

    return context
 
