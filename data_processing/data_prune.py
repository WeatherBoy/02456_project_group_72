import pandas as pd
import numpy as np 
import math

# Remove nan and None and text with /r/ 

#df = pd.read_csv('data/reddit_conversations.csv',sep = '\"',index_col = False, engine = 'python')
csv_dir = "./data/mr_reddit_casual.csv"
df = pd.read_csv(csv_dir,sep ='§',index_col = False, engine = 'python', quoting=3)

drop_idx = []

#print(df.head())
#print(df['Response'].isnull().sum())
#print(len(df[df['Response'] == None]))
#print(df['Message'][34920])
#print('')
#print(df['Response'][34920])
countr = 0
counti = 0

for i in range(len(df)):
    
    if pd.isna(df['Response'][i]) or  pd.isna(df['Message'][i]): #type(df['Response'][i]) == float or type(df['Message'][i]) == float:
        #print(df['Response'][i])
        drop_idx.append(i)
        print('missing message or response')
    else:
        
        if '/r/' in df['Response'][i] or '/r/' in df['Message'][i] :
            #print(df['Response'][i])
            drop_idx.append(i)
            countr += 1
            
        if '[' in df['Response'][i]:
            df['Response'][i] = df['Response'][i].replace('[','')
            df['Response'][i] = df['Response'][i].replace(']','')
        
        if '[' in df['Message'][i]:
            df['Message'][i] = df['Message'][i].replace('[','')
            df['Message'][i] = df['Message'][i].replace(']','')
        # If the text is in multiple lines. 
        try:
            int(df['Message'][i][0])

            if df['Message'][i][1] == '.':
                counti += 1
                #print('Message: ' + df['Message'][i])
                #print('Response: ' + df['Response'][i] + '\n')
                drop_idx.append(i)

               

        except ValueError:
            continue

#print(countr, counti) 
#print(len(df))       

print(f"Removed {len(drop_idx)} lines")   
df.drop(df.index[drop_idx], inplace = True)
df.to_csv(csv_dir.replace('.csv','_new.csv'), sep = '§', index = False)


