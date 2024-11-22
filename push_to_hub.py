from datasets import Dataset
from PIL import Image
from static_globals import *
import csv
# Open the CSV file

map={
    "image":[],
    "character":[],
    "x":[],
    "y":[],
    "x_1":[],
    "y_1":[],
    "angle":[]
}

print("a")

repo_id="jlbaker361/blender_animals"

for csv_file in [f for f in os.listdir(metadata_dir) if f.endswith('.csv')]:
    with open(os.path.join(metadata_dir, csv_file), mode='r') as file:

        csv_reader = csv.reader(file)
        
        # Skip the header if necessary
        next(csv_reader)
        
        # Read each row in the CSV
        for j,row in enumerate(csv_reader):
            image=Image.open(row[0])
            character=row[1]
            x=row[2]
            y=row[3]
            x_1=row[4]
            y_1=row[5]
            angle=row[6]
            map["image"].append(image)
            map["character"].append(character)
            map["x"].append(x)
            map["y"].append(y)
            map["x_1"].append(x_1)
            map["y_1"].append(y_1)
            map["angle"].append(angle)

            if j%100==0 and j >100:
                Dataset.from_dict(map).push_to_hub(repo_id)
        Dataset.from_dict(map).push_to_hub(repo_id)