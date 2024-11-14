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
            
            print(camera_location, is_unobstructed(camera_location, object_location))

            if is_unobstructed(camera_location, object_location):
                # Store valid camera positions
                positions.append(camera_location)
                
    return positions

# Example usage:
reset("budgie",True)
object_location = Vector((3, 3, 1))  # The object's location in world coordinates
radius = 2  # The distance from the object
angle_step = 15  # Step size in degrees for azimuth and elevation
min_distance = 5  # Minimum distance the camera can be placed from the object
max_distance = 50  # Maximum distance the camera can be placed from the object

valid_camera_positions = generate_camera_positions(object_location, radius, angle_step, min_distance, max_distance)

# Create cameras at valid positions and point them at the object
for i, cam_location in enumerate(valid_camera_positions):
    reset("budgie",False)
    # Create a new camera
    camera.location=cam_location
    
    # Make the camera point towards the object
    direction = object_location - cam_location
    rot_quat = direction.to_track_quat('Z', 'Y')
    camera.rotation_euler = rot_quat.to_euler()
    print(cam_location,rot_quat.to_euler())
    
    # Set render settings for the screenshot
    bpy.context.scene.render.filepath = f"{folder}\\pic_{i}.png"
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    

    # Render and save the screenshot from the camera's perspective
    bpy.ops.render.render(write_still=True)

# Now all cameras at valid positions have been created and are pointing at the object
