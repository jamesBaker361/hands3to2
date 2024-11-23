import bpy
import json
import math
import os
import mathutils
from mathutils import Vector
import bpy_extras
import sys 
import random
import subprocess



def install_package(package_name):
    """Install a Python package using Blender's bundled Python."""
    python_executable = sys.executable  # Blender's Python interpreter
    try:
        # Ensure pip is installed
        subprocess.check_call([python_executable, "-m", "ensurepip", "--upgrade"])
        # Upgrade pip
        subprocess.check_call([python_executable, "-m", "pip", "install", "--upgrade", "pip"])
        # Install the required package
        subprocess.check_call([python_executable, "-m", "pip", "install", package_name])
        print(f"Package '{package_name}' installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Failed to install package '{package_name}': {e}")



try:
    import numpy as np
except ModuleNotFoundError:
    install_package("numpy")
import re
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append("\\Users\\jlbak\\hands3to2")
sys.path.append("/Users/jbaker15/Desktop/hands3to2")
from screenshot_data import *
from generate_valid_camera_angles import reset,generate_camera_positions,SHITTY
from math import degrees
from config import *
from static_globals import *

with open(os.path.join(script_directory,"output.txt"), "w") as file:
    sys.stdout = file  # Redirects print() output to the file


    import re
    import platform
    try:
        from PIL import Image, ImageDraw
    except ModuleNotFoundError:
        install_package("pillow")
    # Delete all cameras in the scene
    for obj in list(bpy.data.objects):  # Use a copy of the list to avoid modification during iteration
        if obj.type == 'CAMERA':
            try:
                bpy.data.objects.remove(obj, do_unlink=True)
                print(f"Deleted camera: {obj.name}")
            except ReferenceError:
                print("already removed")

    # Create a new camera
    bpy.ops.object.camera_add(location=(0, 0, 10))  # Adjust location as needed
    camera = bpy.context.object  # The newly created camera becomes the active object

    # Rename the camera to "Camera"
    camera.name = "Camera"

    # Set the new camera as the scene's active camera
    bpy.context.scene.camera = camera

    # Modify the lens property (focal length)
    camera.data.lens = 25  # Set the focal length to 25mm

    print(f"Created new camera '{camera.name}' with lens set to {camera.data.lens}mm.")
    #light=bpy.data.objects["MainLight"]
    scene = bpy.context.scene




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

    def rescale_to_unit_box(obj,target_height=1.0):
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
        
        x_dist=max(v[0] for v in bbox_corners)-min(v[0] for v in bbox_corners)
        y_dist=max(v[1] for v in bbox_corners)-min(v[1] for v in bbox_corners)
        
        max_dimension = max(v[2] for v in bbox_corners)-min(v[2] for v in bbox_corners)
        print(f"max x {x_dist} y {y_dist} z {max_dimension}")
        # Calculate scale factor to fit in a 1x1x1 box
        #max_dimension = max(bbox_size)  # Find the largest dimension
        scale_factor = target_height / max_dimension
        
        # Apply scale factor
        obj.scale = (scale_factor, scale_factor, scale_factor)

        print(f"scale factor is {scale_factor}")
        
        # Apply the scale transformation
        #bpy.ops.object.transform_apply(scale=True)

    def init_gpu():
        # Set the render engine to Cycles
        bpy.context.scene.render.engine = 'CYCLES'

        # Enable GPU rendering
        bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'METAL'  # Use 'CUDA', 'OPTIX', or 'HIP'
        bpy.context.scene.cycles.device = 'GPU'

        # Enable all available GPUs
        for device in bpy.context.preferences.addons['cycles'].preferences.get_devices():
            for gpu in device:
                gpu.use = True

        # Optional: Set tile size for optimal GPU rendering
        bpy.context.scene.render.tile_x = 256
        bpy.context.scene.render.tile_y = 256

        # Optional: Enable denoising
        bpy.context.scene.cycles.use_denoising = True

        print("GPU rendering configured successfully!")



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

    import bpy
    import mathutils




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

    def clean_collection(collection_name:str):
        # Get the collection
        collection = bpy.data.collections.get(collection_name)

        if collection:
            # Loop through all objects in the collection and delete them
            for obj in collection.objects:
                bpy.data.objects.remove(obj, do_unlink=True)
            
            print(f"All objects in the collection '{collection_name}' have been deleted.")
        else:
            print(f"Collection '{collection_name}' not found.")

    new_collection=get_collection(collection_name)
    tracker_collection=get_collection(tracker_collection_name)
    character_collection=get_collection(character_collection_name)

    clean_collection(tracker_collection_name)
    clean_collection(character_collection_name)


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
        character_dict={"Koala_Quad":CharacterParameters([math.pi/2,0,0],"Y"),"Duck_Quad":CharacterParameters([math.pi/2,0,0],"Y")}
        scene_camera_params_dict={
        "bedroom":SceneParameters([[0.4,1.5,0.72]],[0.2,1.5],[5,10],[0.5,4]),
        "office":SceneParameters([[3,3,0.75],
                                [5,3,0.75],
                                [0.3,3,0.75],
                                [2,3.5,0.01],
                                [7,4,0.01]],[0.2,1.0],[5,10],[0.5,3]),
        "room":SceneParameters([
            #[-0.5,-1,0.11],
                            # [0.5,2.2,0.41],
                            # [-1,-0.25,0.21],
                            # [-2.75,0.25,0.57],
                                [-3.5,0.25,0.65],
                                [-3.9,-1.3,0.65],
                            # [-3,-1.3,0.65],
                            # [-1.4,-2.1,0.415],
                                [-1.3,-2.1,0.045],
                            # [-1,-2.2,0.045],
                            # [-5.2,-1.2,0.65],
                            # [-5.1,0.1,0.65],
                            # [1,-0.8,0.2],
                                [1.2,-2.05,0.37],
                            # [0.5,-2.28,1.61],
                            # [0,2,0.55]
                                ],[0.95,1.1],[.05,1],[0.5,2.0])                         
        }

    else:
        scale_samples=4
        light_samples=1
        angle_step=15
        distance_samples=4
        character_angle_step=120
        limit=10000
        limit_per_location=16
        random_angles=4
        # Set resolution to 128x128
        bpy.context.scene.render.resolution_x = 512
        bpy.context.scene.render.resolution_y = 512
        
        # Optional: Lower samples for faster rendering
        bpy.context.scene.cycles.samples = 80  # Adjust as needed
        

        # Set max light bounces to 2
        bpy.context.scene.cycles.max_bounces = 3




    

    for obj_name in [k for k in character_dict.keys()]:
        reset(obj_name,True)
    bpy.context.view_layer.update()
    try:
        init_gpu()
    except:
        print("init gpu failed")
    class BreakOutException(Exception):
        pass
    start=0

    uid= str(random.randint(10000, 99999))
    with open(os.path.join(metadata_dir, f"img_metadata_{uid}.csv"),"w+") as write_file:

        write_file.write("path,character,x,y,x1,y1,angle")
        try:
            scene_mesh_name="room"
            scene_params=scene_camera_params_dict[scene_mesh_name]
            location_list=scene_params.object_location_list
            random.shuffle(location_list)
            for s,location in enumerate(location_list):
                print(f" location to be set = {location}")
                location_count=0
                try:

                    for scale_step in range(scale_samples+1):
                        #scale=scene_params.scale_range[0]+(scene_params.scale_range[1]-scene_params.scale_range[0])*(float(scale_step)/float(scale_samples))
                        scale=round(random.uniform(scene_params.scale_range[0],scene_params.scale_range[1]),2)
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
                            #distance=scene_params.distance_range[0]+(scene_params.distance_range[1]-scene_params.distance_range[0])*(float(distance_step)/float(distance_samples))
                            distance=round(random.uniform(scene_params.distance_range[0],scene_params.distance_range[1]),2)
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
                                    before_rotation=camera.rotation_euler
                                    #for constraint in camera.constraints:
                                    #    camera.constraints.remove(constraint)
                                    camera.rotation_euler=before_rotation
                                    character_keys=[k for k in character_dict.keys()]
                                    random.shuffle(character_keys)
                                    for character in character_keys:
                                        print(f"\t\t\t\t character{character}")
                                        reset(character,False)
                                        
                                        if character not in bpy.data.objects:
                                            filepath=os.path.join(script_directory, "characters",character, f"{character}.obj")
                                            bpy.ops.wm.obj_import(filepath=filepath)
                                        character_obj=bpy.data.objects[character]
                                        for collection in character_obj.users_collection:
                                            collection.objects.unlink(character_obj)

                                        character_collection.objects.link(character_obj)
                                        character_obj.location=(0,0,0)
                                        rescale_to_unit_box(character_obj,scale)
                                        character_obj.rotation_euler=character_dict[character].rotation
                                        character_obj.rotation_euler[2] = 0  # Apply the angle to the Z-axis
                                        # Adjust the object's location based on its bottom point
                                        
                                        # Offset the object's location so its bottom is at desired_location
                                        
                                        #print(f"character_obj.location.z {character_obj.location.z} - min_z {min_z} = {offset_z}")
                                        #print(f"location {character_obj.location}")

                                        bbox_corners = [ character_obj.matrix_world @ mathutils.Vector(corner) for corner in character_obj.bound_box]
                                        min_z = min(corner.z for corner in bbox_corners)  # Find the minimum Z to get the bottom
                                        lowest=0
                                        lowest_corner=bbox_corners[0]
                                        for corner in bbox_corners:
                                            if corner.z<lowest_corner.z:
                                                corner=lowest_corner
                                        print(f"matrix world before {character_obj.matrix_world}")
                                        print(f"lowest corner before mpving {lowest_corner}")
                                        character_obj.location = (location[0], location[1],  location[2]-min_z)
                                        #character_obj.matrix_world.translation=Vector((location[0], location[1],  location[2]))
                                        #print(f"location {character_obj.location}")
                                        axis=character_dict[character].axis

                                        camera_object_distance=camera.location-character_obj.location

                                        relative_rotation=math.radians(quadrant_angle(camera_object_distance[0], camera_object_distance[1]))

                                        #create_bounding_box(character_obj)

                                        print(f"{camera.location} -{character_obj.location} = {camera_object_distance} ")
                                        print(f"relative rotation {relative_rotation}")


                                        # Rotate the object around the axis to align with the camera
                                        character_obj.rotation_euler.rotate_axis(axis, relative_rotation)  # Apply the opposite to align
                                        for rotation in [r for r in range(0,360,character_angle_step)]+[random.randint(0,360) for _ in range(random_angles)]:
                                            print("character location",character_obj.location)
                                            character_obj.rotation_euler.rotate_axis(axis,math.radians(rotation))
                                            character_folder=os.path.join(FOLDER, scene_mesh_name, character)
                                            os.makedirs(character_folder, exist_ok= True)
                                            #os.makedirs(f"{folder}\\{scene_mesh_name}\\{character}",exist_ok=True)
                                            start+=1
                                            location_count+=1
                                            if start>limit or location_count> limit_per_location:
                                                raise BreakOutException
                                            file_name=f"{distance}_{rotation}_{scale}.png"
                                            bpy.context.scene.render.filepath = os.path.join(character_folder, file_name)
                                            
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
                                            write_file.write(f"\n{bpy.context.scene.render.filepath},{character},{x},{y},{x_1},{y_1},{rotation}")
                                            character_obj.rotation_euler.rotate_axis(axis,-math.radians(rotation))
                                            #raise BreakOutException
                                        
                                        reset(character,True)

                        
                        #cleanup(object_name)
                        
                except BreakOutException:
                    if start>=limit:
                        raise BreakOutException        
            #clean_collection(tracker_collection_name)
            reset(scene_mesh_name,True)
        except BreakOutException:
            reset(scene_mesh_name,False)

        reset(scene_mesh_name,False)
