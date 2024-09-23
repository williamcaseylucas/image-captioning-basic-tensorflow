import os
import pickle
import numpy as np
from tqdm.notebook import tqdm
import tensorflow as tf
from tensorflow import keras
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.utils import to_categorical, plot_model
from keras.layers import Input, Dense, LSTM, Embedding, Dropout, add


def extract_features_from_dataset():
    # extract features from image
    features = {}
    directory = os.path.join(BASE_DIR, "Images")

    for image_name in tqdm(os.listdir(directory)):
        # load image from file
        image_path = os.path.join(directory, image_name)
        image = load_img(image_path, target_size=(224, 224))
        # convert image pixels from PIL to numpy array
        image = img_to_array(image)
        # reshape data for model
        image = image[None, :, :, :]
        # preprocess image for VGG
        image = preprocess_input(image)
        feature = model.predict(image, verbose=0)
        # get image id
        image_id = image_name.split(".")[0]
        features[image_id] = feature

    # store features in pickle
    pickle.dump(features, open(os.path.join(WORKING_DIR, "features.pkl"), "wb"))

    return features


def get_image_caption_pairs():
    img_id_to_captions = {}  # img_id: list[captions]
    for line in tqdm(captions_doc.split("\n")):
        # Skip erroneous lines
        if len(line) < 2:
            continue

        tokens = line.split(",")
        image_name, caption = (
            tokens[0],
            tokens[1:],
        )  # can return multiple captions that are not joined
        image_id = image_name.split(".")[0]
        caption = " ".join(caption)
        # convert caption list to string

        if image_id not in img_id_to_captions:
            img_id_to_captions[image_id] = []

        img_id_to_captions[image_id].append(caption)

    return img_id_to_captions


def preprocess_captions_in_dictionary(img_id_to_captions):
    """
    Modify captions such that:
    1. words of length 1 are removed
    2. everything is lowercase
    3. remove digits, special characters, etc
    4. delete extra spaces
    5. add start and end tokens
    """
    # preprocess text data
    for id, captions in img_id_to_captions.items():
        for i, caption in enumerate(captions):
            # convert to lowercase | delete digits, special characters, etc
            caption = caption.lower().replace("[^A-Za-z]", "")
            # delete extra spaces
            caption = caption.replace("\s+", " ")
            # reduce caption to not include words of length 1
            caption = " ".join([w for w in caption.split() if len(w) > 1])
            # add start and end tokens
            caption = "<start> " + caption + " <end>"
            captions[i] = caption
    return img_id_to_captions


def get_all_captions_from_dictionary(img_id_to_captions):
    all_captions = []
    for key in img_id_to_captions:
        for caption in img_id_to_captions[key]:
            all_captions.append(caption)

    return all_captions


# Check for TensorFlow GPU access
print(
    f"TensorFlow has access to the following devices:\n{tf.config.list_physical_devices()}"
)

BASE_DIR = "../../ml-research/Personal Research/datasets/flickr8k"
WORKING_DIR = "./"

# Extract Image features
model = VGG16()
# Restructure model, removes linear layer
model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
# summarize
model.summary()

# features = extract_features_from_dataset()

with open(os.path.join(WORKING_DIR, "features.pkl"), "rb") as f:
    features = pickle.load(f)

# Load captions data
with open(os.path.join(BASE_DIR, "captions.txt"), "r") as f:
    next(f)  # first line is 'image caption' which we can skip over
    captions_doc = f.read()  # {ID}.jpg,{SOME_TEXT}

img_id_to_captions = get_image_caption_pairs()
img_id_to_captions = preprocess_captions_in_dictionary(img_id_to_captions)
all_captions = get_all_captions_from_dictionary(img_id_to_captions)


print(len(all_captions))
