import datasets import Da
from PIL import Image
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

with open('img_metadata.csv', mode='r') as file:
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
