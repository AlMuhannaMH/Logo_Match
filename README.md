# **Logo Similarity Detection**
This project enables finding similar images based on logo similarity using pre-trained VGG16 model features.

Logo Similarity Detection utilizes computer vision techniques to extract features from images using a VGG16 model. It then calculates cosine similarity scores to find and display similar logos from a dataset.

## **Features**
- Feature Extraction: Uses VGG16 to extract image features.
- Cosine Similarity: Calculates similarity scores for logo comparison.
- Streamlit App: Interactive interface to upload and find similar logos.
- Flexible: Supports multiple image formats (JPG, JPEG, PNG).

## **Setup**
### Clone the Repository:

    git clone https://github.com/deepmbhatt/Logo_Match.git

### Install Dependencies:

    pip install -r requirements.txt
    
### Download Pre-trained Model:

Ensure VGG16 model weights are downloaded automatically with tensorflow.keras.
Usage

## **Extract Features:**

- Use extract_features.py to preprocess images and save feature vectors first.
  
## **Find Similar Logos:**

- Utilize logos.py and the Streamlit app to upload a logo and discover similar logos.


License
Distributed under the MIT License. See LICENSE for more information.

## **ScreenShot**
![image](https://github.com/user-attachments/assets/ebfd35b0-7f07-401c-abec-0c7f19feab28)
![image](https://github.com/user-attachments/assets/f9bbd3ff-b591-478d-bb44-b5c57fa53582)


