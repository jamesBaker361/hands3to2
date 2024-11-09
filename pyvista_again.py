import pyvista as pv
import numpy as np
import math

def vertex_mesh(obj_path):
    mesh = pv.read(obj_path)
    scalars=[]
    with open(obj_path, 'r') as obj_file:
        for _,line in enumerate(obj_file):
            parts = line.split()
            if len(parts)<1:
                continue
            if parts[0] == 'v':  # Vertex with color
                x, y, z = map(float, parts[1:4])
                if len(parts)>4:
                    colors= parts[4:7] # Assuming RGB values are integers
                    #print(colors)
                    colors=[float(c) for c in colors]
                    [r,g,b]=colors
                    #colors=[b,g,r]
                    scalars.append(colors)
    if len(scalars)>0:
        mesh.point_data["RGB"]=np.array(scalars)
    return mesh

class SceneParameters:
    def __init__(self,object_location:tuple, camera_locations_and_rotations:list) -> None:
        self.object_location=object_location #x,y,z
        self.camera_locations_and_rotations=camera_locations_and_rotations #[(x,y,z,x,y,z)]

character_list=["budgie"]
scene_camera_params_dict={
    "eagle":SceneParameters([0,0,.019],[1, 1, 0, 0, 0, 0])
}

def rotate_x_y(x,y,degrees):
    theta=math.pi*degrees/180
    new_x=(math.cos(theta)*x)-(math.sin(theta)*y)
    new_y=(math.sin(theta)*x)+(math.cos(theta)*y)
    return new_x,new_y

for scene_mesh_name,scene_params in scene_camera_params_dict.items():
    for character in character_list:
        
        for angle in range(0,360,45):
            plotter=pv.Plotter(off_screen=True)
            scene_mesh=vertex_mesh(scene_mesh_name+".obj").rotate_y(-angle)
            #scene_mesh.rotate_x(angle)
            #scene_mesh.rotate_z(angle)

            plotter.import_obj(character+".obj",character+".mtl")
            #character_mesh.translate(scene_params.object_location)
            #character_mesh.rotate_y(angle)
            #rotate the camera and the scene opposite angle
            camera_location=scene_params.camera_locations_and_rotations[:3]
            new_x,new_z=rotate_x_y(camera_location[0],camera_location[2],angle)
            camera_location[0]=new_x
            camera_location[2]=new_z
            print(camera_location)
            plotter.camera_position = [camera_location, [0,0,0], scene_params.camera_locations_and_rotations[3:]]
            plotter.camera.zoom(0.5)
            plotter.show(auto_close=False)  # Show but don't close the plotter to keep rendering
            try:
                plotter.add_mesh(scene_mesh,scalars="RGB",rgb=True)
            except KeyError:
                plotter.add_mesh(scene_mesh)
            
            # Save each perspective as an image
            plotter.screenshot(f"render_view_{angle}.png")