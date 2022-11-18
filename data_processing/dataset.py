import pandas as pd
from torch.utils.data import Dataset


class reditDataset(Dataset):
    def __init__(self, split, csv_dir):
        # read data
        csv = pd.read_csv(csv_dir,sep = '"')
        self.csv = csv[csv['split'] == split]


    def __len__(self):
        return len(self.csv)

    def __getitem__(self,idx):
        data = {}
        data['message'] = self.csv.loc[idx, 'Message']
        data['response'] = self.csv.los[idx, 'Response']

        return data



if __name__ == '__main__':
    csv_dir = "./data/processed_reddit_casual.csv"

    csv = pd.read_csv(csv_dir, sep= ';:;')
    print(csv)