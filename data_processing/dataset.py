import pandas as pd
from torch.utils.data import Dataset, DataLoader


class reditDataset(Dataset):
    def __init__(self, split, csv_dir):
        # read data
        csv = pd.read_csv(csv_dir,sep = '\"')
        self.csv = csv[csv['split'] == split]


    def __len__(self):
        return len(self.csv)

    def __getitem__(self,idx):
        data = {}
        data['message'] = self.csv.iloc[idx]['Message']
        data['response'] = self.csv.iloc[idx]['Response']

        return data



if __name__ == '__main__':
    csv_dir = "./data/reddit_conversations.csv"
   
    dataset = reditDataset('Train', csv_dir)
    loader = DataLoader(dataset)
    data = next(iter(dataset))
    print(data['message'])
    print(data['response'])

    #for data in loader:
    #    print(data['message'])