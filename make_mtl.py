from collections import defaultdict

def parse_obj(obj_path):
    vertices = []
    faces = []
    with open(obj_path, 'r') as obj_file:
        for _,line in enumerate(obj_file):
            parts = line.split()
            if len(parts)<1:
                continue
            if parts[0] == 'v':  # Vertex with color
                x, y, z = map(float, parts[1:4])
                colors= parts[4:7] # Assuming RGB values are integers
                #print(colors)
                colors=[int(255*float(c)) for c in colors]
                r,g,b=colors
                vertices.append((x, y, z, r, g, b))
            elif parts[0] == 'f':  # Face
                face = [int(p.split('/')[0]) - 1 for p in parts[1:]]
                faces.append(face)
    return vertices, faces

def write_mtl(mtl_path, color_to_material):
    with open(mtl_path, 'w') as mtl_file:
        for color, material_name in color_to_material.items():
            r, g, b = color
            mtl_file.write(f"newmtl {material_name}\n")
            mtl_file.write(f"Kd {r / 255.0} {g / 255.0} {b / 255.0}\n")
            mtl_file.write("illum 1\n\n")

def write_obj_with_mtl(obj_path, output_obj_path, vertices, faces, color_to_material):
    with open(output_obj_path, 'w') as out_obj:
        out_obj.write(f"mtllib {output_obj_path}.mtl\n")
        # Write vertices
        for x, y, z, r, g, b in vertices:
            out_obj.write(f"v {x} {y} {z} {r} {g} {b}\n")
        
        # Write faces with material assignments
        for face in faces:
            # Get color of the first vertex in face to determine material
            color = (vertices[face[0]][3], vertices[face[0]][4], vertices[face[0]][5])
            material_name = color_to_material[color]
            out_obj.write(f"usemtl {material_name}\n")
            out_obj.write("f " + " ".join(str(idx + 1) for idx in face) + "\n")

def process_obj_to_mtl(obj_path, output_obj_path, output_mtl_path):
    vertices, faces = parse_obj(obj_path)
    
    # Map colors to unique material names
    color_to_material = {}
    for vertex in vertices:
        color = vertex[3], vertex[4], vertex[5]
        if color not in color_to_material:
            color_to_material[color] = f"material_{color[0]}_{color[1]}_{color[2]}"
    
    # Write the MTL file
    write_mtl(output_mtl_path, color_to_material)
    
    # Write the new OBJ file with material assignments
    write_obj_with_mtl(obj_path, output_obj_path, vertices, faces, color_to_material)

# Usage
obj_path = 'full_mesh.obj'  # Path to the input OBJ file
output_obj_path = 'output_with_materials.obj'  # Path for the new OBJ file with materials
output_mtl_path = 'output_with_materials.mtl'  # Path for the new MTL file
process_obj_to_mtl(obj_path, output_obj_path, output_mtl_path)
