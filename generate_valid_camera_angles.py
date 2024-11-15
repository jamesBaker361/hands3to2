import bpy
import math
import mathutils
from mathutils import Vector
import os
import sys
sys.path.append("\\Users\\jlbak\\hands3to2")
folder="\\Users\\jlbak\\hands3to2\\camera_test\\"
from screenshot_blend import reset
os.makedirs(folder,exist_ok=True)
camera = bpy.context.scene.camera

def is_unobstructed(camera_location, target_location):
    # Cast a ray from the camera to the target
    direction = (target_location - camera_location).normalized()
    result = bpy.context.scene.ray_cast(bpy.context.view_layer.depsgraph, camera_location, direction)
    
    # If result[0] is True, it means the ray hit something, and the view is obstructed
    return not result[0]

def generate_camera_positions(object_location, radius, angle_step, min_distance, max_distance):
    positions = []
    
    # Loop through different angles to generate camera positions around the object
    for azimuth in range(0, 360, angle_step):  # Azimuth angle (angle around the object)
        for elevation in range(-90, 90, angle_step):  # Elevation angle (angle above/below the object)
            
            # Convert spherical coordinates to Cartesian coordinates
            theta = math.radians(azimuth)
            phi = math.radians(elevation)
            
            
            
            
            # Camera position on the sphere at a given distance
            x = object_location.x + radius * math.cos(phi) * math.sin(theta)
            y = object_location.y + radius * math.cos(phi) * math.cos(theta)
            z = object_location.z + radius * math.sin(phi)
            camera_location = Vector((x, y, z))
            direction = object_location - camera_location
            rot_quat = direction.to_track_quat('-Z', 'Y')
            #camera.rotation_euler = rot_quat.to_euler()
            
            
            print(camera_location, is_unobstructed(camera_location, object_location))

            if is_unobstructed(camera_location, object_location):
                # Store valid camera positions
                positions.append(camera_location)
                
    return positions

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
min_distance = 5  # Minimum distance the camera can be placed from the object
max_distance = 50  # Maximum distance the camera can be placed from the object

valid_camera_positions = generate_camera_positions(object_location, radius, angle_step, min_distance, max_distance)
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
    bpy.context.scene.render.filepath = f"{folder}\\pic_{i}.png"
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
