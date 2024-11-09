import math

class SceneParameters:
    def __init__(self,object_location_and_rotation:tuple, camera_locations_and_rotations:list,light_locations_and_rotations:list) -> None:
        self.object_location_and_rotation=object_location_and_rotation #x,y,z,x,y,z
        self.camera_locations_and_rotations=camera_locations_and_rotations #[(x,y,z,x,y,z)]
        self.light_locations_and_rotations=light_locations_and_rotations #[(x,y,z,x,y,z,power)]

character_list=["budgie"]
scene_camera_params_dict={
    "dungeon":SceneParameters([0,-0.025,-.01,math.pi/2,0,math.pi/2],
                              [[0, 0, 0, math.pi *70/180, 0, math.pi*185/180]],
                              [[.1, 0, 0, 0,0,0,5000]])}