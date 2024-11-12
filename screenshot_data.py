import math

class SceneParameters:
    def __init__(self,object_location_and_rotation:tuple, camera_locations_and_rotations:list,light_locations_and_rotations:list,object_scale:float=1.0) -> None:
        self.object_scale=object_scale
        self.object_location_and_rotation=object_location_and_rotation #x,y,z,x,y,z
        self.camera_locations_and_rotations=camera_locations_and_rotations #[(x,y,z,x,y,z)]
        self.light_locations_and_rotations=light_locations_and_rotations #[(x,y,z,x,y,z,power)]

character_dict={"budgie":[math.pi/2,0,0],"emu":[math.pi/2,0,0]}
scene_camera_params_dict={
    "dungeon":SceneParameters([0,-0.025,-.01,math.pi/2,0,math.pi/2],
                              [[0, 0, 0, math.pi *70/180, 0, math.pi*185/180]],
                              [[.1, 0, 0, 0,0,0,5000]]),
      "Sofa":SceneParameters([-8,-5,3.2, 0,0,0],
                             [[-19.355,-15.395,6.1234,math.pi/2,0,1.78*math.pi]],
                             [[5, 5, 5, 0,0,0,7.5]],
                             5.0),
        "room":SceneParameters([-0.5,-1,0.1,0,0,0],
                               [[-3,1,2,.38*math.pi, 0,1.3*math.pi]],
                               [[5, 5, 5, 0,0,0,7.5],
                                [5, 5, 5, 0,0,0,2.5]
                                ],
                               0.25
                               ),           
        "room0":SceneParameters([-3.25,-1,0.7,0,0,0],
                               [[-4,1.65,1.7,0.44*math.pi, 0,1.05*math.pi],
                                [-2.5,1.5,1.9,0.39*math.pi, 0, 0.94*math.pi],
                                [-1.8,-1,1.9,math.pi/3,0,0.417*math.pi]],
                               [[5, 5, 5, 0,0,0,7.5],
                                [5, 5, 5, 0,0,0,2.5]
                                ],
                               0.25
                               ),
        "room1":SceneParameters([-5.2,-1,0.7,0,0,0],
                               [[-4,1.65,1.7,0.44*math.pi, 0,1.05*math.pi],
                                ],
                               [[5, 5, 5, 0,0,0,7.5],
                                [5, 5, 5, 0,0,0,2.5]
                                ],
                               0.25
                               )    
                              
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