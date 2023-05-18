# Damage-Detection-System
###### This repository contains a Python application that utilizes Mask R-CNN and TensorFlow to detect damages in cars from uploaded images. The application provides a user interface where users can upload an image and view the processed image with identified damages.

# Prerequisites
* Python 3.7   
* Conda

# Installation
1. Clone the repository
2. Create a Conda virtual environment: conda create -n damage-detection python=3.7
3. Activate the virtual environment: conda activate damage-detection
4. Install the required dependencies: pip install -r requirements.txt

# Usage
1. Navigate to the project directory  
2. Start the application: python newapp.py  
3. Open your web browser and go to 'http://localhost:5000' to access the application.  
4. On the application interface, upload an image that you want to detect damages on.  
5. The application will process the image using the trained model and display the result with the detected damages.

# Development
If you prefer to develop the application using the Spyder IDE, follow these additional steps

1. Open the Conda environment with Spyder:  
  conda activate damage-detection  
  spyder
2. In Spyder, open the newapp.py file located in the project directory.
3. Modify and customize the code as needed.
4. Run the application from Spyder's integrated terminal or use the Run button in the IDE.

# Project Structure
newapp.py : Python code for the damage detection application.  
models/ : Folder containing the trained model for damage detection.  
templates/ : Folder containing the HTML template for the application interface.  
static/images : Folder containing output image
requirements.txt : List of Python dependencies required by the application.

# License
This project is licensed under the MIT License.
