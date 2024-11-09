import bpy
import json
import math
import os
import mathutils
import sys
sys.path.append("\\Users\\jlbak\\hands3to2")
from screenshot_data import *

# Set camera parameters
camera = bpy.context.scene.camera
light=bpy.data.objects["MainLight"]

#light.data.shadow_soft_size=10


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

for obj_name in character_list+[k for k in scene_camera_params_dict.keys()]:
    bpy.data.objects[obj_name].hide_set(True)

for scene_mesh_name,scene_params in scene_camera_params_dict.items():
    scene_obj=bpy.data.objects[scene_mesh_name]
    scene_obj.hide_set(False)
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
            for character in character_list:
                folder=f"\\Users\\jlbak\\hands3to2\\{scene_mesh_name}\\{character}"
                os.makedirs(folder,exist_ok=True)
                character_obj.scale=(scene_params.object_scale)
                character_obj=bpy.data.objects[character]
                character_obj.hide_set(False)
                character_obj.location=scene_params.object_location_and_rotation[:3]
                character_obj.rotation_euler=scene_params.object_location_and_rotation[3:]
                
                rotation_degrees = (0, step, 0)  # Set rotation in radians (X, Y, Z)
                light.location=tuple([p for p in light_params[:3]])
                light.rotation_euler=tuple([r for r in light_params[3:6]])
                light.data.energy=light_params[6]
                    #rotation_radians = tuple(math.radians(deg) for deg in rotation_degrees)
                rotation_radians=mathutils.Euler([math.radians(deg) for deg in rotation_degrees], 'XYZ')
                for angle in range(0,45,step):
                    
                
                    character_obj.rotation_euler.rotate_axis("Y",math.radians(step))

                    # Set render settings for the screenshot
                    bpy.context.scene.render.filepath = f"{folder}\\{s}_{l}_{angle}.png"
                    bpy.context.scene.render.image_settings.file_format = 'PNG'
                    

                    # Render and save the screenshot from the camera's perspective
                    bpy.ops.render.render(write_still=True)
                    #bpy.ops.screen.screenshot(bpy.context.scene.render.filepath)


                    print("Screenshot saved to:", bpy.context.scene.render.filepath)
