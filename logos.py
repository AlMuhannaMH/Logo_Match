import streamlit as st
from PIL import Image
import cv2
import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os

# Load pre-trained VGG16 model without top layers (only feature extraction)
vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Function to preprocess and extract features from an image
def extract_features(image, model):
    # Load and resize image
    
    if image is None:
        print(f"Error: Unable to read image '{image}'")
        return None
    image = cv2.resize(image, (224, 224))
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = preprocess_input(image)  # Preprocess image for VGG16
    
    # Extract features using VGG16
    features = model.predict(image)
    features_flattened = features.flatten()  # Flatten features to 1D array
    return features_flattened

# Function to calculate cosine similarity between two feature vectors
def calculate_similarity(features1, features2):
    similarity = cosine_similarity([features1], [features2])
    return similarity[0][0]

with open('features.pickle', 'rb') as f:
    features = pickle.load(f)

def find_similar_logos(logo_features, features, top_n):
    similarity_scores = []
    for i in features:
        similarity = calculate_similarity(logo_features, i[0])
        # Append similarity score and image number to the list
        similarity_scores.append((similarity, i[1]))
    # Sort similarity scores by cosine similarity (ascending order)
    similarity_scores.sort(key=lambda x: -x[0])
    top_images = similarity_scores[:top_n]
    return top_images
# Set the title of the app
st.title("Similar Logos Finder")

# Add an uploader widget to the sidebar
st.sidebar.title("Upload Company Logo")
uploaded_files = st.sidebar.file_uploader("Choose images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
num = st.text_input("Enter number of images to display:", 5)
num = int(num)

# Check if any files have been uploaded
if uploaded_files:
    st.write("Uploaded Logo:")
    
    for uploaded_file in uploaded_files:
        # Open the image
        image = Image.open(uploaded_file)
        image_cv = np.array(image.convert('RGB'))  # Convert to RGB format
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)
        logo_feature = extract_features(image_cv, vgg16)
        similar_logos = find_similar_logos(logo_feature, features, num)
        
    # Display the images
    for rank, (score, image_filename) in enumerate(similar_logos, start=1):
        image_path = os.path.join("./your_folder", image_filename)
        image = Image.open(image_path)
        st.image(image, use_column_width=True)
else:
    st.write("No images uploaded yet. Please upload images to display them.")
