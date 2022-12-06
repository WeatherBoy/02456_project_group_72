import pandas as pd
from torch.utils.data import Dataset, DataLoader



class redditDataset(Dataset):
    def __init__(self, split, max_length, csv_dir):
        # read data
        csv = pd.read_csv(csv_dir,sep = '§', engine='python')
        # Remove pairs that is too long.
        num_init = len(csv)
        get_word_lenght = lambda x: len(x.split(' '))
        num_message = csv['Message'].map(get_word_lenght)
        num_response = csv['Response'].map(get_word_lenght)
        csv = csv[pd.DataFrame([num_message,num_response]).max() < max_length]
        
        if split == 'Train':
            print(f'Word length {max_length} trimmed: {100*(num_init - len(csv))/num_init:.2f} %')
       
       # Select the right split
        self.csv = csv[csv['split'] == split]


    def __len__(self):
        return len(self.csv)

    def __getitem__(self,idx):
        #data = {}
        message = self.csv.iloc[idx]['Message']
        response = self.csv.iloc[idx]['Response']

        return message, response



if __name__ == '__main__':
    csv_dir = "./data/reddit_casual_split.csv"
    '''
    csv = pd.read_csv(csv_dir,sep = '§', engine='python')
    # Remove pairs that is too long.
    num_init = len(csv)
    get_word_lenght = lambda x: len(x.split(' '))
    num_message = csv['Message'].map(get_word_lenght)
    num_response = csv['Response'].map(get_word_lenght)      
    #csv.loc[:,'max_words'] = csv.loc([:,[num_message,num_response]]).max(axis=1)        
    #csv = csv[csv['num_message'].max(axis=1) < 200 or csv['num_response'].max(axis=1)]
    print(csv[pd.DataFrame([num_message,num_response]).max()< 200])
    
    '''
    dataset = reditDataset('Train',200, csv_dir)
    loader = DataLoader(dataset, batch_size=4)
    print(len(loader))
    
    #data = next(iter(dataset))
    #print(type(data['message']))
    #print(data['response'])
    max_lengh = 0
    
    #for message, response in loader:
    #    print('check')
        #print("message size %s"%type(message))
        #print("Reponse size %d"%response.size())
        
        #if len(message[0].split(' ')) > max_lengh:
        #    max_lengh = len(message[0].split(' '))
    #    print(*data['message'])
    #    print(*data['response'])
    #    print('')