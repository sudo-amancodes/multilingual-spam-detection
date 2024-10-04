# Social Media Multilingual Spam Detector

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/flask-2.0+-blue.svg)

A robust and efficient **Social Media Multilingual Spam Detector** built with Python and Flask. This application leverages machine learning techniques to identify and filter spam content across various languages on social media platforms.

## Table of Contents

- [Features](#features)
- [Demo](#demo)
- [Technologies Used](#technologies-used)
- [Dataset](#dataset)
- [Prerequisites](#prerequisites)
- [Steps](#steps)


## Features

- **Multilingual Support**: Detects spam in multiple languages.
- **Web Interface**: User-friendly interface built with Flask.
- **Real-time Detection**: Instantly analyzes and flags spam content.
- **Scalable Architecture**: Easily scalable to handle large volumes of data.
- **Extensible**: Modular design allows for easy integration of additional features.

## Demo

![App Screenshot](screenshots/demo.png)

*Screenshot of the Social Media Multilingual Spam Detector interface.*

## Technologies Used

- **Python 3.8+**
- **Flask**: Web framework for building the application.
- **Scikit-learn**: Utilized for machine learning algorithms.
  - `CountVectorizer`: For text feature extraction.
  - `MultinomialNB`: Naive Bayes classifier for spam detection.
- **Kaggle**: Source of the dataset used for training and evaluation.
- **HTML/CSS/JavaScript**: Front-end development.
- **Bootstrap**: For responsive design.

## Dataset

The project uses the [Social Media Spam Collection](https://www.kaggle.com/datasets) from Kaggle, which contains labeled spam and non-spam messages in multiple languages. This dataset was instrumental in training and evaluating the effectiveness of the spam detection model.

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

## Steps

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/social-media-spam-detector.git
   cd social-media-spam-detector
    ```
2. **Create a virtual environment**
    ```bash
    python -m venv venv
    ```
3. **Activate the virtual environment**
    * Windows
        ```bash
        venv\Scripts\activate
        ```
    * macOS/Linux
        ```bash
        source venv/bin/activate
        ```

4. **Install the required packages**
    ```bash
    pip install -r requirements.txt
    ```
5. **Download the dataset**
    
    Download the dataset from Kaggle and place it in the sms_spam_collection/SMSSpamCollection file.
6. **Train the model**
    ```bash
    python train_model.py
    ```
7. **Run the application**
    ```bash
    flask run
    ```
8. **Access the app**
    
    Open your browser and navigate to http://127.0.0.1:5000/
