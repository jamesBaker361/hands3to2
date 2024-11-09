import re
from PIL import Image
import numpy as np

def parse_mtl(mtl_path):
    texture_paths = {}
    with open(mtl_path, 'r') as mtl_file:
        for line in mtl_file:
            line = line.strip()
            if line.startswith('map_Kd'):
                texture_paths['diffuse'] = line.split()[1]
            elif line.startswith('map_Ns'):
                texture_paths['specular'] = line.split()[1]
            elif line.startswith('map_bump'):
                texture_paths['bump'] = line.split()[1]
    return texture_paths

def parse_obj(obj_path):
    vertices = []
    texture_coords = []
    faces = []
    
    with open(obj_path, 'r') as obj_file:
        for line in obj_file:
            if line.startswith('v '):
                vertices.append(list(map(float, line.strip().split()[1:])))
            elif line.startswith('vt '):
                texture_coords.append(list(map(float, line.strip().split()[1:])))
            elif line.startswith('f '):
                face = []
                for vertex in line.strip().split()[1:]:
                    v, vt = vertex.split('/')[:2]
                    face.append((int(v) - 1, int(vt) - 1))  # zero-indexed
                faces.append(face)
    return vertices, texture_coords, faces

def get_texture_color(image, uv):
    width, height = image.size
    u = int((uv[0]) * width) % width
    v = int((1- uv[1]) * height) % height  # Flip v-coordinate
    return np.array(image.getpixel((u, v)))[:3]/255  # Only RGB

def create_obj_with_vertex_colors(output_path, vertices, faces, texture_coords, diffuse_map):
    with open(output_path, 'w') as obj_out:
        # Write vertices with colors
        for vertex, uv_index in zip(vertices, texture_coords):
            color = get_texture_color(diffuse_map, uv_index)
            vertex_with_color = vertex + color.tolist()
            obj_out.write(f"v {' '.join(map(str, vertex_with_color))}\n")
        
        # Write texture coordinates
        for uv in texture_coords:
            obj_out.write(f"vt {' '.join(map(str, uv))}\n")
        
        # Write faces
        for face in faces:
            face_str = " ".join([f"{v+1}/{vt+1}" for v, vt in face])
            obj_out.write(f"f {face_str}\n")

def main():
    # Paths to files
    mtl_path = 'bird.mtl'
    obj_path = 'bird.obj'
    output_obj_path = 'new_bird.obj'
    
    # Parse the MTL and OBJ files
    texture_paths = parse_mtl(mtl_path)
    vertices, texture_coords, faces = parse_obj(obj_path)
    
    # Load the diffuse texture map
    diffuse_map = Image.open(texture_paths['diffuse'])
    if diffuse_map.mode != 'RGB':
        diffuse_map = diffuse_map.convert('RGB')
    
    # Create new OBJ with vertex colors
    create_obj_with_vertex_colors(output_obj_path, vertices, faces, texture_coords, diffuse_map)
    print(f"New OBJ with vertex colors saved to {output_obj_path}")

if __name__ == '__main__':
    main()
