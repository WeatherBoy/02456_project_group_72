import json
import os
import numpy as np

# I had the JSON file in a neighbouring directory I called data:
# print(f"Current working directory: {os.getcwd()}") # for seeing working directory
DATA_PATH = "data/reddit_casual.json"

# Opening JSON file
f = open(DATA_PATH)
  
# returns JSON object as a dictionary
data = json.load(f)
data = data[:10] # for testing
new_data = []

# lambda function for processing string
stringProcessing = lambda x : x.replace('\"', "").encode('ascii', 'ignore').decode('ascii')
  
# Iterating through the json list
for i in data:
    # Initialize string variables to handle first case
    # - No character-value will be NaN
    # - No text-value will be: 'error_muffin_not_found'
    character = "NaN"
    previous = "error_muffin_not_found"
    
    new_structure = []

    for j in i["lines"]:
        if character == j["character"]:
            previous = previous + " " + j["text"]
        else:
            if(previous != "error_muffin_not_found"):
                new_structure.append(stringProcessing(previous))
            previous = j["text"] #last thing said
        
        character = j["character"] #last person talking
    new_structure.append(stringProcessing(previous))


    for l in range(len(new_structure)-1):
       new_data.append([new_structure[l], new_structure[l+1]]) 
       
#print("New data: ",new_data)
for indx, elem in enumerate(new_data):
    print(f"\n\nPair number: {indx}")
    print("Message: ")
    print(elem[0])
    print("\nResponse: ")
    print(elem[1])

new_data = np.array(new_data)
print(new_data.shape)
  
# Closing file
f.close()

