# Action-Recognition-using-hybrid-LSTMs
## Overview
This project acts as a toy problem for my studienarbeit project.

This project demonstrates how to use Long Short-Term Memory (LSTM) neural networks for action recognition in videos. It uses hybrid deep learning models along with computer vision techniques to analyze video sequences, recognize patterns, and classify different actions.

The project focuses on action recognition using the UCF50 dataset. The hybrid approach combines convolutional neural networks (CNNs) for spatial feature extraction and LSTMs for temporal sequence modeling. The model learns to recognize actions by analyzing both the visual content and the sequence of frames in the video.

## Features

- **Data Processing:** Extracts video frames and processes them for input into the model.
- **Model Architecture:** Combines CNNs for spatial feature extraction and LSTMs for sequence learning.
- **Training and Evaluation:** Trains the hybrid LSTM model on the UCF50 dataset and evaluates its accuracy.
- **Action Recognition:** Classifies human actions based on video input.

## Technologies Used

- **Python**
- **TensorFlow/Keras**: For building and training the LSTM-based neural network.
- **OpenCV**: For video frame extraction and preprocessing.
- **Scikit-Learn**: For dataset preparation and splitting.
- **MoviePy**: For handling video data.
- **Pandas & NumPy**: For data manipulation.
- **Matplotlib**: For visualizing results.

## Installation

To get started with this project, follow the steps below:

1. **Clone the repository:**
    ```bash
    git clone https://github.com/Vigneshwar-KR/Action-Recognition-using-hybrid-LSTMs.git
    ```

2. **Install the required dependencies:**
    ```bash
    pip install tensorflow opencv-contrib-python youtube-dl moviepy pydot
    pip install git+https://github.com/TahaAnwar/pafy.git#egg=pafy
    ```

3. **Download the UCF50 dataset:**
   You can download the UCF50 dataset from [here](https://www.crcv.ucf.edu/data/UCF50.php) or directly via the code.

   ```bash
   # Inside the notebook or script
   !wget --no-check-certificate https://www.crcv.ucf.edu/data/UCF50.rar
   ```

4. **Extract the dataset:**
   If using a Jupyter notebook or Colab:
   ```python
   import os
   if not os.path.exists('UCF50'):
       !unrar x UCF50.rar UCF50/
   ```

5. **Run the Notebook:**
   You can now run the `LSTM.ipynb` notebook to train the model.

## Usage

1. **Frame Extraction:**
   The tool uses OpenCV to extract frames from videos. Frames are resized and normalized for input into the neural network.

2. **Model Training:**
   The hybrid model combines convolutional layers with LSTM layers. It is trained on the UCF50 dataset, which contains 50 different action categories.

3. **Evaluation:**
   After training, the model's performance is evaluated using test data. Accuracy and loss metrics are plotted for analysis.
