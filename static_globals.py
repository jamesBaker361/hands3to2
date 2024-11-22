import platform
import os
import sys

MAC_PATH="/Users/jbaker15/Desktop/hands3to2"
WINDOWS_PATH="\\Users\\jlbak\\hands3to2"
IMAGE_FOLDER="blender_images"

script_directory = os.path.dirname(os.path.abspath(__file__))
FOLDER=os.path.join(script_directory, IMAGE_FOLDER)
os.makedirs(FOLDER,exist_ok=True)
print(f"Script is located in: {script_directory}")
metadata_dir=os.path.join(script_directory, "metadata")
os.makedirs(metadata_dir,exist_ok=True)