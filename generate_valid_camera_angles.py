import bpy
import math
import mathutils
from mathutils import Vector
import re
import platform
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from static_globals import *

camera = bpy.context.scene.camera

SHITTY="shitty_camera"

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
        try:
            collection=bpy.data.collections[obj_name]
            collection.hide_viewport=value
            collection.hide_render=value
            toggle_hide(collection,value)
        except KeyError:
            print(f"{obj_name}  not found")

def is_unobstructed(camera_location, target_location):
    # Cast a ray from the camera to the target
    #print(f"\t\tobstruction {target_location} - {camera_location} = {target_location - camera_location}")
    direction = (target_location - camera_location).normalized()
    result, location, normal, index, obj, matrix = bpy.context.scene.ray_cast(bpy.context.view_layer.depsgraph, camera_location, direction)
    #if result:
        
        #print(f"hit {obj.name}  at location {location} camera is at {camera_location}")
    # If result[0] is True, it means the ray hit something, and the view is obstructed
    return not result


def make_cylinder(location1, location2,new_collection):
    midpoint = (location1 + location2) / 2

    # Calculate the distance between the two points (to set the cylinder length)
    distance = (location2 - location1).length

    # Create a cylinder
    bpy.ops.mesh.primitive_cylinder_add(radius=0.01, depth=distance, location=midpoint)  # Adjust radius as needed
    cylinder = bpy.context.object
    cylinder.name = "RulerCylinder"

    # Calculate the orientation (rotation) of the cylinder to align it between the two points
    direction = location2 - location1
    rotation_quaternion = direction.to_track_quat('Z', 'Y')
    cylinder.rotation_euler = rotation_quaternion.to_euler()
    
    for collection in cylinder.users_collection:
        collection.objects.unlink(cylinder)

    new_collection.objects.link(cylinder)

def generate_camera_positions(object_location, radius, angle_step,scale=1,make_cameras:bool=False,cylinder=False,new_collection=None):
    positions = []
    
    # Loop through different angles to generate camera positions around the object
    for azimuth in range(0, 360, angle_step):  # Azimuth angle (angle around the object)
        for elevation in range(-90, 90, angle_step):  # Elevation angle (angle above/below the object)
            
            # Convert spherical coordinates to Cartesian coordinates
            theta = math.radians(azimuth)
            phi = math.radians(elevation)

            #print(f"\t\t{theta} {phi}")
            
            
            
            
            # Camera position on the sphere at a given distance
            x = object_location.x + radius * math.cos(phi) * math.sin(theta)
            y = object_location.y + radius * math.cos(phi) * math.cos(theta)
            z = object_location.z + radius * math.sin(phi)
            camera_location = Vector((x, y, z))
            
            direction = object_location - camera_location

            #print(f"{object_location} - {camera_location} = {direction}")
            rot_quat = direction.to_track_quat('-Z', 'Y')
            #camera.rotation_euler = rot_quat.to_euler()
            
            
            location_unobstructed=is_unobstructed(camera_location,object_location)
            location_above=(Vector((object_location.x,object_location.y,object_location.z+scale)))
            location_above_unobstructed=is_unobstructed(camera_location,location_above)

            if cylinder:
                make_cylinder(camera_location,object_location,new_collection)
            if make_cameras:
                bpy.ops.object.camera_add(location=camera_location)
                new_camera = bpy.context.object  # Get the new camera object
                for collection in new_camera.users_collection:
                    collection.objects.unlink(new_camera)
                    
                # Link the object to the new collection
                new_collection.objects.link(new_camera)
                
                # Rename the camera to distinguish it from others
                new_camera.name = f"{SHITTY}"
                
                # (Optional) Set any properties or constraints for each camera here
                # Example: Set rotation, focal length, etc.
                new_camera.location = camera_location  # Adjust as needed
                #new_camera.rotation_euler=camera.rotation_euler


            if location_unobstructed and location_above_unobstructed:
                # Store valid camera positions
                positions.append(camera_location)
                
    return positions

if __name__=="__main__":
    # Example usage:
    reset("budgie",True)
    reset("room",False)
    object_location = Vector((-3, -1, 0.75))  # The object's location in world coordinates
    bpy.ops.object.empty_add(type='PLAIN_AXES', location=object_location)
    empty_target = bpy.context.object
    empty_target.name = "TrackingEmpty"  # Name the Empty for easier identification

    constraint = camera.constraints.new(type='TRACK_TO')
    constraint.target = empty_target
    constraint.track_axis = 'TRACK_NEGATIVE_Z'  # Camera points down -Z by default
    constraint.up_axis = 'UP_Y'  # Y-axis is usually the upward direction

    radius = 2  # The distance from the object
    angle_step = 30  # Step size in degrees for azimuth and elevation

    valid_camera_positions = generate_camera_positions(object_location, radius, angle_step)
    reset("budgie",False)
    # Create cameras at valid positions and point them at the object
    for i, cam_location in enumerate(valid_camera_positions):
        
        # Create a new camera
        camera.location=cam_location
        
        # Make the camera point towards the object
        direction = object_location - cam_location

        # Calculate the rotation for the camera to face the target
        # Set the camera to look down the negative Z-axis and up along the Y-axis
        rot_quat = direction.to_track_quat('-Z', 'Y')
        #camera.rotation_euler = rot_quat.to_euler()
        # Set render settings for the screenshot
        bpy.context.scene.render.filepath =  os.path.join(FOLDER, f"pic_{i}.png") #f"{FOLDER}\\pic_{i}.png"
        bpy.context.scene.render.image_settings.file_format = 'PNG'
        

        # Render and save the screenshot from the camera's perspective
        bpy.ops.render.render(write_still=True)
        
        bpy.ops.object.camera_add(location=cam_location)
        new_camera = bpy.context.object  # Get the new camera object
        
        # Rename the camera to distinguish it from others
        new_camera.name = f"Camera_{i+1}"
        
        # (Optional) Set any properties or constraints for each camera here
        # Example: Set rotation, focal length, etc.
        new_camera.location = cam_location  # Adjust as needed
        new_constraint = new_camera.constraints.new(type='TRACK_TO')
        new_constraint.target = empty_target
        new_constraint.track_axis = 'TRACK_NEGATIVE_Z'  # Camera points down -Z by default
        new_constraint.up_axis = 'UP_Y'  # Y-axis is usually the upward direction
