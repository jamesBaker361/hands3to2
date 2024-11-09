import pyvista as pv
import numpy as np

# Download an example mesh (e.g., the bunny mesh)
obj_path="new_bird.obj"
mesh = pv.read(obj_path)


# Create a plotter object to manage multiple views
plotter = pv.Plotter(off_screen=True)
scalars=[]
with open(obj_path, 'r') as obj_file:
    for _,line in enumerate(obj_file):
        parts = line.split()
        if len(parts)<1:
            continue
        if parts[0] == 'v':  # Vertex with color
            x, y, z = map(float, parts[1:4])
            colors= parts[4:7] # Assuming RGB values are integers
            #print(colors)
            colors=[float(c) for c in colors]
            [r,g,b]=colors
            #colors=[b,g,r]
            scalars.append(colors)

# Add the mesh to the plotter
#mesh.point_data["colors"]=np.array(scalars)

if "colors" in mesh.point_data:  # Adjust 'colors' to the actual name of the color data array
    mesh.active_scalars_name = "colors"
    plotter = pv.Plotter()
    plotter.add_mesh(mesh, scalars="colors")
    plotter.show()
else:
    print("No vertex colors found in the OBJ file.")
    print(len(scalars))
    print(mesh.point_data)
    mesh.point_data["RGB"]=np.array(scalars)
    plotter.add_mesh(mesh,scalars="RGB",rgb=True)

# Set up different camera positions (elevation, azimuth, roll)
views = [
    (1, 1, 1),   # View 1: Diagonal view from the top
    (0, 0, 1),   # View 2: Top-down view
    (1, 0, 0),   # View 3: Side view
    (0, 1, 0),   # View 4: Front view
    (1,0,1),
    (0,1,1),
    (1,1,0),
    (0,-1,0),
    (-1,0,0),
    (0,0,-1),
    (-1,-1,-1)
]

# Render the mesh from different perspectives
for i, view in enumerate(views):
    plotter.camera_position = view
    plotter.show(auto_close=False)  # Show but don't close the plotter to keep rendering

    # Save each perspective as an image
    plotter.screenshot(f"render_view_{i + 1}.png")

# Close the plotter once all renders are done
plotter.close()
