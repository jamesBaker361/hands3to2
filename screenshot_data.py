import math
from math import pi
class SceneParameters:
    def __init__(self,object_location_list:list, scale_range:list,light_range:list,distance_range:list):
        self.object_location_list=object_location_list
        self.scale_range=scale_range
        self.light_range=light_range
        self.distance_range=distance_range

class CharacterParameters:
    def __init__(self,rotation:list,axis:str) -> None:
        self.rotation=rotation
        self.axis=axis

character_dict={"Raccoon_Quad":CharacterParameters([math.pi/2,0,0],"Y")}
scene_camera_params_dict={
    "bedroom":SceneParameters([[0.4,1.5,0.72]],[0.2,1.5],[5,10],[0.5,4]),
    "office":SceneParameters([[3,3,0.75],
                              [5,3,0.75],
                              [0.3,3,0.75],
                              [2,3.5,0.01],
                              [7,4,0.01]],[0.2,1.0],[5,10],[0.5,3]),
    "room":SceneParameters([
        #[-0.5,-1,0.11],
                            [0.5,2.2,0.41],
                            [-1,-0.25,0.21],
                            [-2.75,0.25,0.57],
                            [-3.5,0.25,0.65],
                            [-3.9,-1.3,0.65],
                            [-3,-1.3,0.65],
                            [-1.4,-2.1,0.415],
                            [-1.3,-2.1,0.045],
                            [-1,-2.2,0.045],
                            [-5.2,-1.2,0.65],
                            [-5.1,0.1,0.65],
                            [1,-0.8,0.2],
                            [1.2,-2.05,0.37],
                            [0.5,-2.28,1.61],
                            [0,2,0.55]
                            ],[0.5,5.0],[5,10],[0.1,10.0])                         
    }

test_scene_camera_params_dict={
    "Sofa":SceneParameters([-8,-5,3.2, 0,0,0],
                             [[-19.355,-15.395,6.1234,math.pi/2,0,1.78*math.pi]],
                             [[5, 5, 5, 0,0,0,7.5]],
                             5.0),
        "room":SceneParameters([-0.5,-1,0.1,0,0,0],
                               [[-3,1,2,.38*math.pi, 0,1.3*math.pi]],
                               [[5, 5, 5, 0,0,0,7.5],
                                #[5, 5, 5, 0,0,0,2.5]
                                ],
                               0.25
                               ),   
}