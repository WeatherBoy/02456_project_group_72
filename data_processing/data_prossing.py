### Extract all json conversations to get overview of content.
import json
import os
import numpy as np

# I had the JSON file in a neighbouring directory I called data:
print(f"Current working directory: {os.getcwd()}") # for seeing working directory
DATA_PATH = "./data/reddit_casual.json"
NEW_DATA_PATH = "./data/json_extract.csv"

# Opening JSON file
f = open(DATA_PATH)
data = json.load(f)
outfile = open(NEW_DATA_PATH,'w',encoding="utf8")

# The JSON format is abit confusing to look at so here one conversation is extraced to be one line.
for linedict in data:                                   # The data is a list of dicts
    for converlist in linedict.values():                # Every dict is a conversaion with key = "line" and value = conversation list
        for messagedict in converlist:                  # Every elemet in the list is a dict with key = charater and text, and value = num-charater and "dialog"
            for chare,dialog in messagedict.items():    # Loop over all the chaters and their dialog 
                if "§" in str(dialog):
                    dialog = dialog.replace("§","")
                outfile.write("{}".format(dialog).replace("\n",""))
                outfile.write("§")
        outfile.write('\n')
            #outfile.write(message.values())
        #print("{}:;:".format(value))
    #outfile.write("{}:;:{}".format(keys,value))
outfile.close()


# lambda function for processing string, removes quotation marks and unicode-smileys
stringProcessing = lambda x : x.replace('\"', "").encode('ascii', 'ignore').decode('ascii')
infile = open("./data/json_extract.csv",'r',encoding="utf8")

#print(infile)

new_structure =[]
for line in infile:
    line = line.split('§')
    MRpaircount = 0
    MRpairadded = 0
    character = "NaN"
    message = ""
    response = ""
    for i in range(len(line)):
        if i ==0:
            #capture start charater
            character = line[i]
            
        elif i%2==0:
            
            #check if new charater
            if character != line[i]:
                character = line[i]
                MRpaircount += 1
                
                if MRpaircount == MRpairadded + 2:
                    MRpairadded += 1
                    new_structure.append([message,response])
                
                
                message = response
                response = ''

        elif i%2==1:
            #capture new text
            response = response + line[i] + " "
        

new_data = []
count=0

for l in range(len(new_structure)-1):
    
    if [new_structure[l][0], new_structure[l][1]] not in new_data:
        new_data.append([new_structure[l][0], new_structure[l][1]])
        

    count+=1
    print(str(count) + " / " + str(len(new_structure)))




with open("./data/mr_reddit_casual.csv",'w',encoding="utf8") as outfile:
    outfile.write('Message§Response\n')
    for i in range(len(new_data)):
        outfile.write('{}§{}\n'.format(stringProcessing(new_data[i][0]),stringProcessing(new_data[i][1])))









    
    

#     for j in i["lines"]:
#         if character == j["character"]:
#             previous = previous + " " + j["text"]
#         else:
#             if(previous != "error_muffin_not_found"):
#                 new_structure.append(stringProcessing(previous))
#             previous = j["text"] #last thing said
        
#         character = j["character"] #last person talking
#     new_structure.append(stringProcessing(previous))

#     #remove dublicates 
#     for l in range(len(new_structure)-1):
#         if [new_structure[l], new_structure[l+1]] not in new_data:
#             new_data.append([new_structure[l], new_structure[l+1]]) 

# with open(NEW_DATA_PATH,'w') as outfile:
#     outfile.write('Message;:;Response\n')
#     for i in range(len(new_data)):
#         outfile.write('{};:;{}\n'.format(new_data[i][0],new_data[i][1]))



# #%%
# import json
# import os
# import numpy as np

# # TODO:
# # - Check whether the amount of saved messages & responses makes sense
# #       (right now it looks like there are duplicates)
# # - Save the file as something we can use in a dataloader


# # I had the JSON file in a neighbouring directory I called data:
# print(f"Current working directory: {os.getcwd()}") # for seeing working directory
# #%%
# DATA_PATH = "../data/reddit_casual.json"
# NEW_DATA_PATH = "../data/processed_reddit_casual.csv"
# TESTING = True

# # Opening JSON file
# f = open(DATA_PATH)
  
# # returns JSON object as a dictionary
# data = json.load(f)

# # if TESTING:
# #     # ingenius code... yes I know
# #     data = data[:30]
    
# new_data = []


# def stringProcessing(string_to_process : str):
#     import re
    
#     # Removes those unicode-smileys
#     string_to_process = string_to_process.encode('ascii', 'ignore').decode('ascii')
    
#     # This removes special characters i.e. quotation marks and smileys, but sadly also
#     # parentheses.
#     # What it basicly says is:
#     # 'anything that isn't:
#     #   a letter, a capital letter, a number, an exclamation mark, a question mark,
#     #   an apostrophe, a comma or a full stop
#     # replace it with a space'
#     string_to_process = re.sub(r"[^a-zA-Z0-9!?',.]+", ' ', string_to_process)
    
#     return string_to_process


# # lambda function for processing string, removes quotation marks and unicode-smileys
# stringProcessing = lambda x : x.replace('\"', "").encode('ascii', 'ignore').decode('ascii')
  
# convertion_count = 0
# # Iterating through the json list
# for i in data:
#     convertion_count += 1
#     print(convertion_count)
#     # Initialize string variables to handle first case
#     # - No character-value will be NaN
#     # - No text-value will be: 'error_muffin_not_found'
#     character = "NaN"
#     previous = "error_muffin_not_found"
    
#     new_structure =[]

#     for j in i["lines"]:
#         if character == j["character"]:
#             previous = previous + " " + j["text"]
#         else:
#             if(previous != "error_muffin_not_found"):
#                 new_structure.append(stringProcessing(previous))
#             previous = j["text"] #last thing said
        
#         character = j["character"] #last person talking
#     new_structure.append(stringProcessing(previous))

#     #remove dublicates 
#     for l in range(len(new_structure)-1):
#         if [new_structure[l], new_structure[l+1]] not in new_data:
#             new_data.append([new_structure[l], new_structure[l+1]]) 


# # #get uniq entries
# # uniq_list = []
# # for message in new_data:
# #     #check if a entry has been seen before
# #     if message not in uniq_list:
# #         uniq_list.append(message)
        

# # if TESTING:
# #     # once more, ingenious testing
# #     for indx, elem in enumerate(uniq_list):
# #         print(f"\n\nPair number: {indx} ")
# #         print("Message: ")
# #         print(elem[0])
# #         print("\nResponse: ")
# #         print(elem[1])

# # new_data = np.array(new_data)
# # print(f"shape of new data: {new_data.shape}")
  
# # Closing file
# f.close()


# %%




