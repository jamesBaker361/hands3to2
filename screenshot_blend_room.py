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
from generate_valid_camera_angles import reset,generate_camera_positions,SHITTY
from math import degrees
from config import *


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
    max_dimension = max(v[2] for v in bbox_corners)-min(v[2] for v in bbox_corners)
    
    # Calculate scale factor to fit in a 1x1x1 box
    #max_dimension = max(bbox_size)  # Find the largest dimension
    scale_factor = 1.0 / max_dimension
    
    # Apply scale factor
    obj.scale = (scale_factor, scale_factor, scale_factor)

    print(f"scale factor is {scale_factor}")
    
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
tracker_collection_name="TrackerCollection"
character_collection_name="CharacterCollection"


def get_collection(collection_name:str):
    if collection_name in bpy.data.collections:
        # Set new_collection to the existing collection
        new_collection = bpy.data.collections[collection_name]
        print(f"Collection '{collection_name}' already exists.")
    else:
        # Create a new collection
        new_collection = bpy.data.collections.new(collection_name)
        bpy.context.scene.collection.children.link(new_collection)
        print(f"Collection '{collection_name}' created.")

    return new_collection

new_collection=get_collection(collection_name)
tracker_collection=get_collection(tracker_collection_name)
character_collection=get_collection(character_collection_name)


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




for c in character_dict.keys():
    if c not in bpy.data.objects:
        bpy.ops.wm.obj_import(filepath=f"\\Users\\jlbak\\hands3to2\\characters\\{c}\\{c}.obj")
    obj=bpy.data.objects[c]
    for collection in obj.users_collection:
        collection.objects.unlink(obj)

    character_collection.objects.link(obj)

for obj_name in [k for k in character_dict.keys()]:
    reset(obj_name,True)
bpy.context.view_layer.update()
class BreakOutException(Exception):
    pass
start=0
try:
    scene_mesh_name="room"
    scene_params=scene_camera_params_dict[scene_mesh_name]
    for s,location in enumerate(scene_params.object_location_list):
        location_count=0
        try:
            for scale_step in range(scale_samples+1):
                scale=scene_params.scale_range[0]+(scene_params.scale_range[1]-scene_params.scale_range[0])*(float(scale_step)/float(scale_samples))
                for constraint in camera.constraints:
                    camera.constraints.remove(constraint)
                print(f"scale {scale}")
                bpy.ops.object.empty_add(type='PLAIN_AXES', location=Vector((location[0], location[1], location[2]+scale/2)))
                empty_target = bpy.context.object
                empty_target.location=(location[0],location[1],location[2]+scale/2)
                object_name="TrackingEmpty"
                empty_target.name = object_name  # Name the Empty for easier identification
                for collection in empty_target.users_collection:
                    collection.objects.unlink(empty_target)
                    
                # Link the object to the new collection
                tracker_collection.objects.link(empty_target)

                constraint = camera.constraints.new(type='TRACK_TO')
                constraint.target = empty_target
                constraint.track_axis = 'TRACK_NEGATIVE_Z'  # Camera points down -Z by default
                constraint.up_axis = 'UP_Y'  # Y-axis is usually the upward direction
                for distance_step in range(distance_samples+1):
                    distance=scene_params.distance_range[0]+(scene_params.distance_range[1]-scene_params.distance_range[0])*(float(distance_step)/float(distance_samples))
                    print(f"\t distance {distance}")
                    location_vector=Vector((location[0],location[1],location[2]))
                    camera_position_list=generate_camera_positions(location_vector,distance,angle_step,scale,False,False,new_collection)
                    for light_step in range(light_samples+1):
                        light_energy=scene_params.light_range[0]+(scene_params.light_range[1]-scene_params.light_range[0])*(light_step/light_samples)
                        #light.data.energy=light_energy
                        print(f"\t\t light {light_energy}")
                        for c,camera_pos in enumerate(camera_position_list):
                            print(f"\t\t\tposition {camera_pos}")
                            camera.location=camera_pos
                            for character in character_dict:
                                
                                print(f"\t\t\t\t character{character}")
                                reset(character,False)
                                
                                character_obj=bpy.data.objects[character]
                                print("inital charcater location",character_obj.location)
                                character_obj.scale=scale *character_obj.scale
                                #bpy.ops.object.transform_apply(scale=True)
                                character_obj.rotation_euler=character_dict[character].rotation
                                character_obj.rotation_euler[2] = 0  # Apply the angle to the Z-axis
                                # Adjust the object's location based on its bottom point
                                bbox_corners = [character_obj.matrix_world @ mathutils.Vector(corner) for corner in character_obj.bound_box]
                                min_z = min(corner.z for corner in bbox_corners)  # Find the minimum Z to get the bottom

                                # Offset the object's location so its bottom is at desired_location
                                offset_z = character_obj.location.z - min_z
                                character_obj.location = (location[0], location[1],  location[2]+ offset_z)
                                axis=character_dict[character].axis

                                camera_object_distance=camera.location-character_obj.location

                                relative_rotation=math.radians(quadrant_angle(camera_object_distance[0], camera_object_distance[1]))

                                print(f"{camera.location} -{character_obj.location} = {camera_object_distance} ")
                                print(f"relative rotation {relative_rotation}")

                                

                                # Rotate the object around the axis to align with the camera
                                character_obj.rotation_euler.rotate_axis(axis, relative_rotation)  # Apply the opposite to align
                                for rotation in range(0,360,character_angle_step):
                                    print("character location",character_obj.location)
                                    character_obj.rotation_euler.rotate_axis(axis,math.radians(character_angle_step))
                                    os.makedirs(f"{folder}\\{scene_mesh_name}\\{character}",exist_ok=True)
                                    start+=1
                                    location_count+=1
                                    if start>limit or location_count> limit_per_location:
                                        raise BreakOutException

                                    bpy.context.scene.render.filepath = f"{folder}\\{scene_mesh_name}\\{character}\\{distance}_{s}_{c}_{light_energy}_{rotation}_{scale}.png"
                                    bpy.context.scene.render.image_settings.file_format = 'PNG'
                                    

                                    # Render and save the screenshot from the camera's perspective
                                    bpy.ops.render.render(write_still=True)
                                    #bpy.ops.screen.screenshot(bpy.context.scene.render.filepath)

                                    bpy.context.view_layer.update()
                                    x,y=world_to_screen(location)
                                    x_1,y_1=world_to_screen((location[0], location[1], location[2] + scale))
                                    #add_dots_to_image(bpy.context.scene.render.filepath,(x,y),(x_1,y_1))
                                    print(f"x,y= {x},{y} x_1,y_1 {x_1},{y_1}")
                                    print("Screenshot saved to:", bpy.context.scene.render.filepath)
                                    #raise BreakOutException
                                reset(character,True)
                
                #cleanup(object_name)
                
        except BreakOutException:
            if start>=limit:
                raise BreakOutException        
    reset(scene_mesh_name,True)
except BreakOutException:
    reset(scene_mesh_name,False)

reset(scene_mesh_name,False)
