class SceneParameters:
    def __init__(self,object_location:tuple, camera_locations_and_rotations:list) -> None:
        self.object_location=object_location #x,y,z
        self.camera_locations_and_rotations=camera_locations_and_rotations #[(x,y,z,x,y,z)]

    

parameters={
    "dungeon":SceneParameters(
        (0,0,-0.019),[-0.0146, -0.2257, -0.1036,66.40694450, 0, 175.929594]
    )
}