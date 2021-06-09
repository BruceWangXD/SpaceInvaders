import pandas as pd
import os

path = "/Users/darap/Documents/School/University/2021, Sem. 1/DATA3888/Aqua10/Datasets/Good Data - Sandeep no errors/"

while True:
    file = input("Name of the file: ")

    if file[-4::] != ".txt":
        print("Please input a text file.")
        continue

    labels_dat = pd.read_csv(path+file, sep=",\t", skiprows=1)
    labels_dat.columns = ["label", "time"]
    dist_labs = sorted(list(set(labels_dat.label)))
    conversion_dict = {}
    #print(dist_labs)
    i = 0
    while i < len(dist_labs):
        converted = (input(str(dist_labs[i]) + " = ").strip()).lower()
        #print("Converted: ", converted)
        conversion_dict[dist_labs[i]] = converted
        
        #print("Labels: ", labels_dat["label"])
        i += 1
    labels_dat["label"] = labels_dat["label"].replace(conversion_dict)
    #os.remove(file+path)
    #instead of removing the file, create a new file with "_CONV.txt" at end

    #print("Labels: ", labels_dat["label"])