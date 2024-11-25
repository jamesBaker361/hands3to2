import math
from math import pi
from static_globals import *
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

directory=os.path.join(script_directory, "characters")
characters=[d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
print(characters)
character_dict={c:CharacterParameters([math.pi/2,0,0],"Y") for c in characters}
scene_camera_params_dict={
    "bedroom":SceneParameters([[0.4,1.5,0.72]],[0.2,1.5],[5,10],[0.5,4]),
    "office":SceneParameters([[3,3,0.75],
                              [5,3,0.75],
                              [0.3,3,0.75],
                              [2,3.5,0.01],
                              [7,4,0.01]],[0.2,1.0],[5,10],[0.5,3]),
    "room":SceneParameters([
          [-3.5, 0.25, 3.174],
 [-3.5, 1, 3.174],
 [-2, -1, 3.174],
 [-2, 2, 3.174],
 [-3, 2, 3.174],
 [-1, 2, 3.174],
 [0.6, 2.1, 3.174],
 [-1, 0, 3.174],
 [1.115, -0.5, 3.174],
 [-0.25, -1, 3.174],
 [1.75, 2.5, 3.174]],[0.25,0.75],[5,10],[0.75,3.0])                         
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