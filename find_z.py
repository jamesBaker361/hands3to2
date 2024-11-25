import bpy

# Define the list of (x, y) coordinates
xy_pairs = [(-3.5, 0.25), (-3.5, 1), (-2, -1),
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
            bbox = [obj.matrix_world @ bpy.mathutils.Vector(corner) for corner in obj.bound_box]
            
            # Check if any corner of the bounding box overlaps the (x, y) coordinate
            for corner in bbox:
                if corner.x == x and corner.y == y and corner.z < 3:
                    max_z = max(max_z, corner.z)
    return max_z if max_z != float('-inf') else None  # Return None if no valid Z is found

# Loop through each (x, y) pair and find the tallest Z under 3
results = {}
for x, y in xy_pairs:
    tallest_z = get_tallest_z_under_3(x, y)
    results[(x, y)] = tallest_z

# Print results
for (x, y), z in results.items():
    if z is not None:
        print(f"The tallest Z below 3 at ({x}, {y}) is {z}")
    else:
        print(f"No valid Z below 3 found at ({x}, {y})")
