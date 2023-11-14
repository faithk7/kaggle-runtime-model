import torch


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.data = []
        self.targets = []
        self.create_dataset()
    
    
if __name__ == '__main__':
    print("testing out the dataset creation")
    data_dir = "/Users/kaiqu/Desktop/kaggle-runtime-optimization/dataset"
    dataset = Dataset(data_dir)

