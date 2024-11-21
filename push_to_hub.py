import datasets
from PIL import Image
import csv
# Open the CSV file

map={
    "image":[],
    "character":[],
    "x":[],
    "y":[],
    "x_1":[],
    "y_1":[]
}

with open('img_metadata.csv', mode='r') as file:
    csv_reader = csv.reader(file)
    
    # Skip the header if necessary
    next(csv_reader)
    
    # Read each row in the CSV
    for row in csv_reader:
        print(row)  # Each row is a list