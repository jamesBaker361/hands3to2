angle_step=45
import math
import bpy
rad=3
for azimuth in range(0, 360, angle_step):  # Azimuth angle (angle around the object)
    for elevation in range(0, 90, angle_step):  # Elevation angle (angle above/below the object)
        theta = math.radians(azimuth)
        phi = math.radians(elevation)

        #print(f"\t\t{theta} {phi}")




        # Camera position on the sphere at a given distance
        x =  rad* math.cos(phi) * math.sin(theta)
        y = rad* math.cos(phi) * math.cos(theta)
        z =  rad* math.sin(phi)

        location= (round(x,2),round(y,2),round(z,2))

        empty = bpy.data.objects.new(object_data=None)

        # Set the location of the empty object
        empty.location = location

        # Link the empty object to the current scene
        bpy.context.collection.objects.link(empty)