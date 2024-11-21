import os
import trimesh

# Directory containing .obj files
input_directory = os.getcwd()
output_directory = os.getcwd()

# Create output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Iterate over all files in the input directory
for filename in os.listdir(input_directory):
    if filename.endswith(".obj"):
        # Load the mesh
        file_path = os.path.join(input_directory, filename)
        mesh = trimesh.load(file_path)

        # Get the current bounding box dimensions
        bounding_box = mesh.bounding_box_oriented.bounds
        box_size = bounding_box[1] - bounding_box[0]
        
        # Find the scale factor to fit within a 1x1x1 box
        scale_factor = 1.0 / max(box_size)
        
        # Scale the mesh
        mesh.apply_scale(scale_factor)
        
        # Center the mesh within the unit box
        mesh.vertices -= mesh.bounding_box.centroid

        # Save the scaled mesh
        output_path = os.path.join(output_directory, filename)
        mesh.export(output_path)
        print(f"Rescaled and saved: {output_path}")

print("Rescaling complete.")
