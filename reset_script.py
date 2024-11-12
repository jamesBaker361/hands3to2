import bpy
import json
import math
import os
import mathutils
from mathutils import Vector
import sys
sys.path.append("\\Users\\jlbak\\hands3to2")
from screenshot_data import *
import re

# Set camera parameters
camera = bpy.context.scene.camera
light=bpy.data.objects["MainLight"]

#light.data.shadow_soft_size=10

def toggle_hide(obj,value:bool):
    # Check if the obj is a collection
    if isinstance(obj, bpy.types.Collection):
        # Recursively call toggle_hide on each child in the collection
        obj.hide_viewport=value
        obj.hide_render=value
        for child in obj.objects:
            toggle_hide(child,value)
        for sub_collection in obj.children:
            toggle_hide(sub_collection,value)
    else:
        # Toggle hide_viewport for individual objects
        obj.hide_viewport=value
        obj.hide_render=value
    bpy.context.view_layer.update()

def reset(obj_name:str,value:bool):
    print(f"resetting {obj_name} to {value}")
    obj_name=re.sub(r'\d+', '', obj_name)
    try:
        bpy.data.objects[obj_name].hide_viewport=value
        bpy.data.objects[obj_name].hide_render=value
    except:
        collection=bpy.data.collections[obj_name]
        collection.hide_viewport=value
        collection.hide_render=value
        toggle_hide(collection,value)

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
print([c for c in bpy.data.collections])
for obj_name in [k for k in character_dict.keys()]+[k for k in scene_camera_params_dict.keys()]:
    reset(obj_name,False)
bpy.context.view_layer.update()