import math
from math import pi
class SceneParameters:
    def __init__(self,object_location_list:list, scale_range:list,light_range:list,distance_range:list):
        self.object_location_list=object_location_list
        self.scale_range=scale_range
        self.light_range=light_range
        self.distance_range=distance_range

character_dict={"budgie":[math.pi/2,0,0],"emu":[math.pi/2,0,0]}
scene_camera_params_dict={
    "room":SceneParameters([[-0.5,-1,0.5],[0,2,0.55]],[0.2,0.4],[5,10],[1,3])
                              
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