import pandas as pd
from torch.utils.data import Dataset, DataLoader



class reditDataset(Dataset):
    def __init__(self, split, csv_dir):
        # read data
        csv = pd.read_csv(csv_dir,sep = '§', engine='python')
        self.csv = csv[csv['split'] == split]


    def __len__(self):
        return len(self.csv)

    def __getitem__(self,idx):
        #data = {}
        message = self.csv.iloc[idx]['Message']
        response = self.csv.iloc[idx]['Response']

        return message, response, [message,response]



if __name__ == '__main__':
    csv_dir = "./data/reddit_casual_split.csv"
   
    dataset = reditDataset('Train', csv_dir)
    loader = DataLoader(dataset)
    data = next(iter(dataset))
    #print(type(data['message']))
    #print(data['response'])
    max_lengh = 0
    for message,_,_ in loader:
        
        if len(message[0].split(' ')) > max_lengh:
            max_lengh = len(message[0].split(' '))
    #    print(*data['message'])
    #    print(*data['response'])
    #    print('')
    print(max_lengh)
        
    
