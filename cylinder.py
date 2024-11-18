import bpy
import json
import math
import os
import mathutils
from mathutils import Vector
import bpy_extras
import numpy as np
import re
import sys
sys.path.append("\\Users\\jlbak\\hands3to2")
sys.path.append("/Users/jbaker15/Desktop/hands3to2")
from screenshot_data import *
from generate_valid_camera_angles import reset,generate_camera_positions,SHITTY,make_cylinder
from math import degrees
from config import *

distance_samples=1
light_samples=1
scale_samples=1
angle_step=90


import re
import platform
from PIL import Image, ImageDraw
# Set camera parameters
camera = bpy.context.scene.camera
#light=bpy.data.objects["MainLight"]
scene = bpy.context.scene

#light.data.shadow_soft_size=10

using_mac=True

if platform.system() == "Darwin":
    print("Running on macOS")
    folder="/home/jlb638/hands3to2/blender_images"
elif platform.system() == "Windows":
    print("Running on Windows")
    using_mac=False
    folder="\\Users\\jlbak\\hands3to2\\blender_images\\"
else:
    print(f"Running on another OS {platform.system()}")

os.makedirs(folder,exist_ok=True)


def add_dots_to_image(image_path, coords1, coords2, radius=5):
    """
    Opens an image, adds two dots at specified coordinates, and saves the modified image.
    
    Parameters:
    - image_path: Path to the input image.
    - coords1, coords2: Tuples (x, y) representing the coordinates where dots will be drawn.
    - radius: The radius of the dots to be drawn. Default is 5.
    - output_path: Path where the modified image will be saved.
    """
    # Open the image
    image = Image.open(image_path)

    width, height = image.size
    # Create a drawing context
    draw = ImageDraw.Draw(image)
    coords1=[width*coords1[0], height*coords1[1]]
    coords2=[width*coords2[0], height*coords2[1]]

    # Draw the first dot (coords1)
    draw.ellipse([coords1[0] - radius, coords1[1] - radius, coords1[0] + radius, coords1[1] + radius], fill="red")

    # Draw the second dot (coords2)
    draw.ellipse([coords2[0] - radius, coords2[1] - radius, coords2[0] + radius, coords2[1] + radius], fill="blue")

    # Save the modified image
    image.save(image_path)



def world_to_screen(world_coords):
    """
    Convert world coordinates to screen coordinates using the camera.

    Parameters:
    - world_coords: A mathutils.Vector with world coordinates (X, Y, Z).

    Returns:
    - (x, y): Screen coordinates (2D).
    """
    # Get the active camera
    camera = bpy.context.scene.camera

    # Get the 3D region and region_data (view3d context)
    region = bpy.context.region
    rv3d = bpy.context.region_data

    # Convert world coordinates to camera view coordinates
    view_coords = bpy_extras.object_utils.world_to_camera_view(
        bpy.context.scene, camera, Vector(world_coords)
    )

    # Convert camera view coordinates to 2D screen space
    screen_coords = mathutils.Vector((
        (view_coords.x + 1.0) * 0.5 * region.width,  # Normalize X to screen space
        (view_coords.y + 1.0) * 0.5 * region.height  # Normalize Y to screen space
    ))
    
    print("view coords ",view_coords)

    return (view_coords.x,1-view_coords.y)

def rescale_to_unit_box(obj):
    # Make sure the object is selected and active
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    
    # Calculate the bounding box dimensions
    bbox_corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    min_corner = Vector((min(v[0] for v in bbox_corners),
                         min(v[1] for v in bbox_corners),
                         min(v[2] for v in bbox_corners)))
    max_corner = Vector((max(v[0] for v in bbox_corners),
                         max(v[1] for v in bbox_corners),
                         max(v[2] for v in bbox_corners)))
    bbox_size = max_corner - min_corner
    
    # Calculate scale factor to fit in a 1x1x1 box
    max_dimension = max(bbox_size)  # Find the largest dimension
    scale_factor = 1.0 / max_dimension
    
    # Apply scale factor
    obj.scale = (scale_factor, scale_factor, scale_factor)
    
    # Apply the scale transformation
    bpy.ops.object.transform_apply(scale=True)



def quadrant_angle(x, y):
    # Calculate base angle in degrees
    theta = math.degrees(math.atan2(abs(x), abs(y)))

    # Apply quadrant-based conditions
    if x < 0 and y > 0:
        return theta
    elif x < 0 and y < 0:
        return 180 - theta
    elif x > 0 and y < 0:
        return 180 + theta
    elif x > 0 and y > 0:
        return 360 - theta
    else:
        return 0  # Handle the origin if necessary

new_camera_params = {
           # "location": (5.0, -5.0, 5.0),   # Change to your desired location
           # "rotation_euler": (1.0, 0.0, 0.78),  # Change to your desired rotation in radians
            "focal_length": 35.0,   # Adjust focal length as desired
            "sensor_width": 24.0,   # Typical sensor width in mm
            "sensor_height":24.0,
            "clip_start": 0.01,
            "clip_end": 1000.0,
            "angle_x":0.90,
            "angle_y":0
}

def cleanup(substring:str):
    for obj in bpy.data.objects:
        if substring in obj.name:
            # Deselect all objects and select the target object
            bpy.ops.object.select_all(action='DESELECT')
            obj.select_set(True)
            
            # Delete the selected object
            bpy.ops.object.delete()
            print(f"Deleted: {obj.name}")

# import each scene, based on the filename, add it to a collection with its name, then delete it
# also import each character


collection_name="CameraCollection"
if collection_name in bpy.data.collections:
    # Set new_collection to the existing collection
    new_collection = bpy.data.collections[collection_name]
    print(f"Collection '{collection_name}' already exists.")
else:
    # Create a new collection
    new_collection = bpy.data.collections.new(collection_name)
    bpy.context.scene.collection.children.link(new_collection)
    print(f"Collection '{collection_name}' created.")

tracker_collection_name="TrackerCollection"
if tracker_collection_name in bpy.data.collections:
    # Set new_collection to the existing collection
    tracker_collection = bpy.data.collections[tracker_collection_name]
    print(f"Collection '{tracker_collection_name}' already exists.")
else:
    # Create a new collection
    tracker_collection = bpy.data.collections.new(tracker_collection_name)
    bpy.context.scene.collection.children.link(tracker_collection)
    print(f"Collection '{tracker_collection_name}' created.")
#cleanup(SHITTY)
print([c for c in bpy.data.collections])

testing=True
# Set render engine to Cycles
bpy.context.scene.render.engine = 'CYCLES'
bpy.context.scene.render.resolution_percentage = 100
bpy.context.scene.cycles.use_denoising = True

if testing:
    # Set resolution to 128x128
    bpy.context.scene.render.resolution_x = 128
    bpy.context.scene.render.resolution_y = 128
    
    # Optional: Lower samples for faster rendering
    bpy.context.scene.cycles.samples = 10  # Adjust as needed
    

    # Set max light bounces to 2
    bpy.context.scene.cycles.max_bounces = 2

else:
    # Set resolution to 128x128
    bpy.context.scene.render.resolution_x = 512
    bpy.context.scene.render.resolution_y = 512
    
    # Optional: Lower samples for faster rendering
    bpy.context.scene.cycles.samples = 100  # Adjust as needed
    

    # Set max light bounces to 2
    bpy.context.scene.cycles.max_bounces = 3


location1=Vector((0.0000, 2.0000, 0.5285))
location2= Vector((0.0000, 2.0000, 0.4500))


make_cylinder(location1, location2,new_collection)