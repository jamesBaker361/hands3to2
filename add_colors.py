from PIL import Image
import numpy as np

def load_texture(texture_path):
    # Load the texture image as a numpy array
    texture_image = Image.open(texture_path)
    if texture_image.mode != 'RGB':
        texture_image = texture_image.convert('RGB')
    texture_data = np.array(texture_image)
    print(texture_data.shape)
    return texture_data, texture_image.size

def get_color_from_texture(u, v, texture_data, texture_width, texture_height):
    # Convert normalized (u, v) to pixel coordinates
    x = int(u * (texture_width - 1))  # Horizontal coordinate
    y = int((1 - v) * (texture_height - 1))  # Vertical coordinate (flip v)
    # Get the color at (x, y) position as RGB tuple
    return texture_data[y, x][:3]  # Use RGB only, ignore alpha if present

def parse_obj_with_texture_coords(obj_path):
    vertices = []
    texture_coords = []
    faces = []

    with open(obj_path, "r") as file:
        for line in file:
            parts = line.strip().split()
            if parts[0] == "v":
                vertices.append(tuple(map(float, parts[1:4])))  # Store vertex
            elif parts[0] == "vt":
                u, v = map(float, parts[1:3])
                texture_coords.append((u, v))  # Store texture coordinate
            elif parts[0] == "f":
                # Store face with vertex/texture indices
                face = []
                for vertex_data in parts[1:]:
                    vertex_indices = vertex_data.split('/')
                    face.append((int(vertex_indices[0]), int(vertex_indices[1])))
                faces.append(face)
    return vertices, texture_coords, faces

def save_obj_with_vertex_colors(output_path, vertices, vertex_colors, faces):
    with open(output_path, "w") as file:
        # Write vertices with colors
        for (x, y, z), (r, g, b) in zip(vertices, vertex_colors):
            file.write(f"v {x} {y} {z} {r/255.0} {g/255.0} {b/255.0}\n")

        # Write texture coordinates
        for u, v in texture_coords:
            file.write(f"vt {u} {v}\n")

        # Write faces with vertex/texture indices
        for face in faces:
            face_str = " ".join([f"{vi}/{ti}" for vi, ti in face])
            file.write(f"f {face_str}\n")

# Paths to your files
obj_path = "bird.obj"
texture_path = "Textures/Bird_Quad_Diffuse.png"
output_path = "colored_bird.obj"

# Load texture image and extract dimensions
texture_data, (texture_width, texture_height) = load_texture(texture_path)
#print(texture_data)

# Parse OBJ file to get vertices, texture coordinates, and faces
vertices, texture_coords, faces = parse_obj_with_texture_coords(obj_path)

# Generate vertex colors by mapping texture coordinates to colors
vertex_colors = [
    get_color_from_texture(u, v, texture_data, texture_width, texture_height)
    for u, v in texture_coords
]

# Save the new OBJ file with vertex colors
save_obj_with_vertex_colors(output_path, vertices, vertex_colors, faces)

print("New OBJ file saved with vertex colors.")
