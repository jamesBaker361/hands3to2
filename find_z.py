import bpy
import mathutils
import os
import sys
import random

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append("\\Users\\jlbak\\hands3to2")
sys.path.append("/Users/jbaker15/Desktop/hands3to2")

print(" ")


# Define the list of (x, y) coordinates
xy_pairs = [(-3.5, 0.25), 
            (-3.5, 1),
            (-2, -1),
            (-2,2),
            (-3,2),
            (-1,2),
            (0.6,2.1),
            (-1,0),
            (1.115,-0.5),
        (-0.25, -1),
        (1.75,2.5)]  # Replace with your actual coordinates

# Function to find the Z-coordinate of the tallest bounding box below 3
def get_tallest_z_under_3(x, y):
    max_z = float('-inf')  # Start with the lowest possible value
    for obj in bpy.data.objects:
        if obj.type in {'MESH', 'CURVE', 'SURFACE', 'META'}:  # Consider only relevant object types
            # Get the world-space bounding box
            bbox = [obj.matrix_world @ mathutils.Vector(corner) for corner in obj.bound_box]
            
            # Check if any corner of the bounding box overlaps the (x, y) coordinate
            x_max=max([corner.x for corner in bbox])
            y_max=max([corner.y for corner in bbox])
            x_min=min([corner.x for corner in bbox])
            y_min=min([corner.y for corner in bbox])
            _max_z=max([corner.z for corner in bbox])+.025
            if x>= x_min and x <= x_max:
                if y >= y_min and y <= y_max:
                    if _max_z<2:
                        max_z = max(max_z, _max_z)
    return max_z if max_z != float('-inf') else None  # Return None if no valid Z is found


def get_random_point(x_min,x_max,y_min,y_max):
    return (round(random.uniform(x_min, x_max),3),round(random.uniform(y_min, y_max),3))


box_bounds=[(-2.5,1.5,-2,3),(-5,-3,-2,1.5)]
# Loop through each (x, y) pair and find the tallest Z under 3
results = {}
for b in box_bounds:
    for k in range(5):
        xy_pairs.append(get_random_point(*b))
for x, y in xy_pairs:
    tallest_z = get_tallest_z_under_3(x, y)
    results[(x, y)] = tallest_z

# Print results
for (x, y), z in results.items():
    if z is not None:
        print(f" [{x}, {y}, {round(z,3)}],")
