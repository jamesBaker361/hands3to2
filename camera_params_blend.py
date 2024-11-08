import bpy
import math

# Find the active 3D view in the current window screen areas
for area in bpy.context.window.screen.areas:
    if area.type == 'VIEW_3D':
        # Access the 3D View space data
        space = area.spaces.active
        
        # Ensure we're in perspective mode
        if space.region_3d.view_perspective == 'PERSP':
            region_3d = space.region_3d

            # Print the viewport location, rotation, and approximate FOV
            print("Location:", region_3d.view_location)
            print("Rotation (radians):", region_3d.view_rotation.to_euler())
            print("Rotation (degrees):", tuple(math.degrees(a) for a in region_3d.view_rotation.to_euler()))

            print("params for dict",[*region_3d.view_location]+[*region_3d.view_rotation.to_euler()])

            # Calculate approximate FOV based on view_camera_zoom
            fov_x = region_3d.view_camera_zoom * math.radians(50) / 4.0
            fov_y = region_3d.view_camera_zoom * math.radians(50) / 4.0

            print("Approx. Field of View X (radians):", fov_x)
            print("Approx. Field of View Y (radians):", fov_y)
            
            break
