from plyfile import PlyData

# Load the binary PLY file
with open("gsplat.ply", "rb") as file:
    plydata = PlyData.read(file)

print(plydata)

# Access vertex data
vertices = plydata['vertex']
#print(vertices.data)   # Print all vertex data

# Example: Get specific property arrays
x_coords = vertices['x']
y_coords = vertices['y']
z_coords = vertices['z']
normals = [(nx, ny, nz) for nx, ny, nz in zip(vertices['nx'], vertices['ny'], vertices['nz'])]

# Print a sample
#print(f"Sample vertex coordinates: {x_coords[:5]}, {y_coords[:5]}, {z_coords[:5]}")
print(len(vertices.data[0]))