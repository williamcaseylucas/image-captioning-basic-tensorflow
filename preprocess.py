import os
import pickle
import numpy as np
from tqdm.notebook import tqdm
import tensorflow as tf
from tf.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical, plot_model, load_img, img_to_array
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
from collections import defaultdict
from typing import List, Dict, Tuple, Generator


def extract_features_from_dataset() -> Dict[str, np.ndarray]:
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


def get_image_caption_pairs() -> Dict[str, List[str]]:
    img_id_to_captions = defaultdict(list)  # img_id: list[captions]
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

        img_id_to_captions[image_id].append(caption)

    return img_id_to_captions


def preprocess_captions_in_dictionary(img_id_to_captions) -> Dict[str, List[str]]:
    """
    Modify captions such that:
    1. words of length 1 are removed
    2. everything is lowercase
    3. remove digits, special characters, etc
    4. delete extra spaces
    5. add start and end tokens
    """
    # preprocess text data
    for _, captions in img_id_to_captions.items():
        for i, caption in enumerate(captions):
            # convert to lowercase | delete digits, special characters, etc
            caption = caption.lower().replace("[^A-Za-z]", "")
            # delete extra spaces
            caption = caption.replace("\s+", " ")
            # reduce caption to not include words of length 1
            caption = " ".join([w for w in caption.split() if len(w) > 1])
            # add start and end tokens
            caption = "startseq " + caption + " endseq"
            captions[i] = caption
    return img_id_to_captions


def get_all_captions_from_dictionary(img_id_to_captions) -> List[List[str]]:
    return [c for captions in img_id_to_captions.values() for c in captions]


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
    features: Dict[str, np.ndarray] = pickle.load(f)

# Load captions data
with open(os.path.join(BASE_DIR, "captions.txt"), "r") as f:
    next(f)  # first line is 'image caption' which we can skip over
    captions_doc = f.read()  # {ID}.jpg,{SOME_TEXT}

img_id_to_captions = get_image_caption_pairs()
img_id_to_captions = preprocess_captions_in_dictionary(img_id_to_captions)
all_captions = get_all_captions_from_dictionary(img_id_to_captions)

# Tokenize text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_captions)
# dict of word to int val
vocab_size = len(tokenizer.word_index) + 1

# get max length of caption available
max_length = max(len(caption.split()) for caption in all_captions)

# train test split
image_ids = list(img_id_to_captions.keys())
split = int(len(image_ids) * 0.9)
train_ids = image_ids[:split]
test_ids = image_ids[split:]


# Create data generator to get data in batches
def data_generator(
    ids: List[str],
    img_id_to_captions: Dict[str, List[str]],
    features: Dict[str, np.ndarray],
    tokenizer: Tokenizer,
    max_length: int,
    vocab_size: int,
    batch_size: int,
):
    X1, X2, y = [], [], []
    size = 0

    while True:
        for key in ids:
            size += 1
            captions_for_img = img_id_to_captions[key]
            for caption in captions_for_img:
                # encode sequence from letters to numbers
                seq = tokenizer.texts_to_sequences([caption])[0]
                # split sequence into X, y pairs
                for i in range(1, len(seq)):
                    # split into input and output pairs
                    in_seq, out_seq = seq[:i], seq[i]
                    # pad input sequence
                    in_seq = pad_sequences([in_seq], maxlen=max_length, padding="post")[
                        0
                    ]
                    # encode output sequence (like one-hot)
                    out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]

                    # Given image features, encoded text, predict output sequence

                    # Change features from (1, 4096) to (4096,)
                    X1.append(features[key].flatten())
                    X2.append(in_seq)
                    y.append(out_seq)

            if size == batch_size:
                X1, X2, y = np.array(X1), np.array(X2), np.array(y)
                yield (X1, X2), y
                X1, X2, y = [], [], []
                size = 0


# Model creation
def construct_model():
    tf.config.list_physical_devices("GPU")

    # For encoder image feature model
    inputs1 = Input(shape=(4096,))
    feat1 = Dropout(0.5)(inputs1)
    feat2 = Dense(256, activation="relu")(feat1)

    # sequence feature layers
    inputs2 = Input(shape=(max_length,))
    # mask_zero -> padding sequences
    # ensures Embedding counts value 0 as skippable
    seq1 = Embedding(input_dim=vocab_size, output_dim=256, mask_zero=True)(inputs2)
    seq2 = Dropout(0.4)(seq1)
    seq3 = LSTM(256)(seq2)

    # decoder model
    decoder1 = add([feat2, seq3])
    decoder2 = Dense(256, activation="relu")(decoder1)
    outputs = Dense(vocab_size, activation="softmax")(decoder2)

    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss="categorical_crossentropy", optimizer="adam")

    return model


model = construct_model()

plot_model(model, show_shapes=True)

epochs = 15
batch_size = 64
steps = len(train_ids) // batch_size

for i in range(epochs):
    generator = data_generator(
        train_ids,
        img_id_to_captions,
        features,
        tokenizer,
        max_length,
        vocab_size,
        batch_size,
    )

    # fit for one epoch
    model.fit(generator, epochs=1, steps_per_epoch=steps, verbose=1)

model.save("best_model.h5")

model.load_weights("best_model.h5")


# Generate captions for image
def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


def predict_caption(model, image, tokenizer, max_length):
    # Add start tag for generation process
    in_text = "startseq"

    # Iterate over the max length of sequence
    for i in range(max_length):
        # Encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # Pad the sequence by the max length of the possible caption (1, 35)
        sequence = pad_sequences([sequence], max_length, padding="post")
        # Get the prediction
        # (1, 4096), (1, 35)
        yhat = model.predict([image, sequence], verbose=0)
        # Get predicted word for the current timestep
        yhat = np.argmax(yhat)

        # Convert index to word
        word = idx_to_word(yhat, tokenizer)
        # Stop if word not found
        if word is None:
            break
        # Append word as input for generating next word
        in_text += " " + word
        # Stop if we reach end tag
        if word == "endseq":
            break
    return in_text


from nltk.translate.bleu_score import corpus_bleu

len(train_ids), len(test_ids)
# Validate with test data
actual, predicted = [], []
for key in tqdm(test_ids):
    # get actual caption
    captions = img_id_to_captions[key]
    # predict the caption
    y_pred = predict_caption(model, features[key], tokenizer, max_length)
    words_in_captions = [caption.split() for caption in captions]
    y_pred = y_pred.split()

    # store actual and predicted captions
    actual.append(words_in_captions)
    predicted.append(y_pred)

# calcuate BLEU score
print("BLEU-1 score: ", corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
print("BLEU-2 score: ", corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))

img_id_to_captions["101669240_b2d3e7f17b"][0].split()
features["101669240_b2d3e7f17b"].shape

# Visualize results
from PIL import Image
import matplotlib.pyplot as plt

# image_name = "101669240_b2d3e7f17b.jpg"
image_name = "1012212859_01547e3f17.jpg"
image_id = image_name.split(".")[0]
image_path = os.path.join(BASE_DIR, "Images", image_name)
image = Image.open(image_path)

captions = img_id_to_captions[image_id]

print("-------------------------Actual captions---------------------------------")
for caption in captions:
    print(caption)
print("-------------------------Predicted captions---------------------------------")

y_pred = predict_caption(model, features[image_id], tokenizer, max_length)
print(y_pred)
