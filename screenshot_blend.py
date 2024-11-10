import bpy
import json
import math
import os
import mathutils
from mathutils import Vector
import sys
sys.path.append("\\Users\\jlbak\\hands3to2")
from screenshot_data import *

# Set camera parameters
camera = bpy.context.scene.camera
light=bpy.data.objects["MainLight"]

#light.data.shadow_soft_size=10

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
    try:
        bpy.data.objects[obj_name].hide_set(True)
    except:
        collection=bpy.data.collections[obj_name]
        for obj in collection.objects:
            obj.hide_set(True)

for scene_mesh_name,scene_params in scene_camera_params_dict.items():
    try:
        scene_obj=bpy.data.objects[scene_mesh_name]
        scene_obj.hide_set(False)
    except:
        collection=bpy.data.collections[obj_name]
        for obj in collection.objects:
            obj.hide_set(False)
    for s,camera_params in enumerate(scene_params.camera_locations_and_rotations):
        
        # Apply the new camera parameters
        camera.location = camera_params[:3]
        camera.rotation_euler = camera_params[3:]
        camera.data.lens = new_camera_params["focal_length"]
        camera.data.sensor_width = new_camera_params["sensor_width"]
        camera.data.sensor_height=new_camera_params["sensor_height"]
        camera.data.clip_start = new_camera_params["clip_start"]
        camera.data.clip_end = new_camera_params["clip_end"]
        #camera.output
        #camera.data.angle_x=new_camera_params["angle_x"]
        #camera.data.angle_y=new_camera_params["angle_y"]
        
        
        step=45
        for l,light_params in enumerate(scene_params.light_locations_and_rotations):
            for character in character_dict:
                folder=f"\\Users\\jlbak\\hands3to2\\{scene_mesh_name}\\{character}"
                os.makedirs(folder,exist_ok=True)
                character_obj=bpy.data.objects[character]
                rescale_to_unit_box(character_obj)
                character_obj.scale=(scene_params.object_scale,scene_params.object_scale,scene_params.object_scale)
                character_obj.rotation_euler=character_dict[character]

                # Calculate the direction from the object to the camera in the XY plane
                direction_to_camera = bpy.context.scene.camera.location - character_obj.location
                direction_to_camera.z = 0  # Ignore the Z component to only rotate in the XY plane

                # Normalize the direction vector
                direction_to_camera.normalize()

                # Get the angle between the object's current forward direction and the direction to the camera
                angle = math.atan2(direction_to_camera.y, direction_to_camera.x)

                # Set the object's rotation around the Z axis
                obj.rotation_euler[2] = angle  # Apply the angle to the Z-axis

                # Update the scene
                bpy.context.view_layer.update()


                character_obj.hide_set(False)

                desired_location=scene_params.object_location_and_rotation[:3]
                # Adjust the object's location based on its bottom point
                bbox_corners = [character_obj.matrix_world @ mathutils.Vector(corner) for corner in character_obj.bound_box]
                min_z = min(corner.z for corner in bbox_corners)  # Find the minimum Z to get the bottom

                # Offset the object's location so its bottom is at desired_location
                offset_z = desired_location[2] - min_z
                character_obj.location = (desired_location[0], desired_location[1], character_obj.location.z + offset_z)

                #character_obj.location=scene_params.object_location_and_rotation[:3]
                #character_obj.rotation_euler=scene_params.object_location_and_rotation[3:]
                
                rotation_degrees = (0, step, 0)  # Set rotation in radians (X, Y, Z)
                light.location=tuple([p for p in light_params[:3]])
                light.rotation_euler=tuple([r for r in light_params[3:6]])
                light.data.energy=light_params[6]
                    #rotation_radians = tuple(math.radians(deg) for deg in rotation_degrees)
                rotation_radians=mathutils.Euler([math.radians(deg) for deg in rotation_degrees], 'XYZ')
                for angle in range(0,90,step):
                    
                
                    character_obj.rotation_euler.rotate_axis("Y",math.radians(step))

                    # Set render settings for the screenshot
                    bpy.context.scene.render.filepath = f"{folder}\\{s}_{l}_{angle}.png"
                    bpy.context.scene.render.image_settings.file_format = 'PNG'
                    

                    # Render and save the screenshot from the camera's perspective
                    bpy.ops.render.render(write_still=True)
                    #bpy.ops.screen.screenshot(bpy.context.scene.render.filepath)


                    print("Screenshot saved to:", bpy.context.scene.render.filepath)
                character_obj.hide_set(True)
                #character_obj.scale=(0.0000001,0.0000001,0.0000001)
    try:
        scene_obj=bpy.data.objects[scene_mesh_name]
        scene_obj.hide_set(True)
    except:
        collection=bpy.data.collections[obj_name]
        for obj in collection.objects:
            obj.hide_set(True)
