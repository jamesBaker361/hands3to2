import bpy
import json

# Set camera parameters
camera = bpy.context.scene.camera

# Define new camera parameters
new_camera_params = {
    "location": (5.0, -5.0, 5.0),   # Change to your desired location
    "rotation_euler": (1.0, 0.0, 0.78),  # Change to your desired rotation in radians
    "focal_length": 35.0,   # Adjust focal length as desired
    "sensor_width": 36.0,   # Typical sensor width in mm
    "clip_start": 0.1,
    "clip_end": 1000.0
}

# Apply the new camera parameters
camera.location = new_camera_params["location"]
camera.rotation_euler = new_camera_params["rotation_euler"]
camera.data.lens = new_camera_params["focal_length"]
camera.data.sensor_width = new_camera_params["sensor_width"]
camera.data.clip_start = new_camera_params["clip_start"]
camera.data.clip_end = new_camera_params["clip_end"]

# Set render settings for the screenshot
bpy.context.scene.render.filepath = "\\Users\\jlbak\\hands3to2\\screenshot.png"
bpy.context.scene.render.image_settings.file_format = 'PNG'

# Render and save the screenshot from the camera's perspective
bpy.ops.render.render(write_still=True)



print("Screenshot saved to:", bpy.context.scene.render.filepath)
