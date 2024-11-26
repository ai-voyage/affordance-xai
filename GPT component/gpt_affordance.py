import base64
import gzip
import json
import glob
import random

import numpy as np
import openai
import os
import requests
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())

openai.api_key = 'your_key'
print("api loadded")


def parse_json_gz(json_gz_file):
    with gzip.open(json_gz_file, 'rt', encoding='utf-8') as f:
        data = json.load(f)

    return data


def get_completion4_vision(prompt, model="gpt-4-vision-preview"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0,  # this is the degree of randomness of the model's output
        max_tokens=300,
    )
    return response.choices[0].message["content"]


# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def encode_image_heatmap(heatmap):
    return base64.b64encode(heatmap).decode('utf-8')


# Function to extract the object name
def extract_object_name(filename):
    parts = filename.split('_')[:-3]
    # If the name consists of two words like 'food_can', join them back
    name_parts = [part for part in parts if not part.isdigit()]
    return '_'.join(name_parts)


# root_dir = "images/GPT experiments/DeiT_2"
root_dir = "data/GPT_experiments/ablative"

gzCounter = len(glob.glob1(root_dir, "*.gz"))

files = glob.glob1(root_dir, "*.gz")
# file = random.choice(files)

# Random selection
# for i in range(len(files)):
#     file_name = random.choice(files)

# Manual selection
for file_name in files:
    data = parse_json_gz(os.path.join(root_dir, file_name))

    img_file = file_name.split(".")[0]
    image_path = os.path.join(root_dir, img_file + "_.png")
    # Getting the base64 string
    base64_image = encode_image(image_path)
    object_name = extract_object_name(file_name)
    print(object_name)
    labels = data["predicted_labels"]
    print(labels)
    manual_label = labels[1]
    vision_prompt = [
        {"type": "text",
         "text": "Your task is to provide a description for the heatmaps of objects that are related to the task of "
                 "affordance learning. I will provide the object category and their "
                 "affordance labels, your task will be to describe the predicted affordance labels based on uploaded "
                 f"heatmaps in one sentence. This is the heatmap of a {object_name} that shows predicted affordances of {manual_label}."

         },
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            },
        },
    ]

    vision_response = get_completion4_vision(vision_prompt)
    print(vision_response)
    print(data["filename"])
    print(data["predicted_labels"])
    entry = {"filename": file_name,
             "object_name": object_name,
             "response": vision_response,
             # "method": method,
             "predicted_labels": labels,
             }
    gzip_name = f"GPT_{file_name.split('.')[0]}_GPT" + ".json"

    gzip_path = os.path.join(root_dir, gzip_name)
    print(gzip_name)
    with open(gzip_path, 'wt', encoding='utf-8') as outfile:
        json.dump(entry, outfile)
        print("saved json")


print("end")