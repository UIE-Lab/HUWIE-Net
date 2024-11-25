import copy
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
    
class UIEBD(Dataset):
    def __init__(self, data_type, konf):
        
        self.train_data_idx = [0, 800]
        self.val_data_idx = [800, 890]
        self.test_data_idx = [800, 890]
        
        if konf == 1:
            self.annotations_file_path = '../../Data/UIEBD/UIEBD_random_shuffle_1.txt'
        elif konf ==2:
            self.annotations_file_path = '../../Data/UIEBD/UIEBD_random_shuffle_2.txt'
        elif konf ==3:
            self.annotations_file_path = '../../Data/UIEBD/UIEBD_random_shuffle_3.txt'
        elif konf ==4:
            self.annotations_file_path = '../../Data/UIEBD/UIEBD_random_shuffle_4.txt'        
        elif konf ==5:
            self.annotations_file_path = '../../Data/UIEBD/UIEBD_random_shuffle_5.txt'
        elif konf ==6:
            self.annotations_file_path = '../../Data/UIEBD/UIEBD_random_shuffle_6.txt'
        elif konf ==7:
            self.annotations_file_path = '../../Data/UIEBD/UIEBD_random_shuffle_7.txt'
        elif konf ==8:
            self.annotations_file_path = '../../Data/UIEBD/UIEBD_random_shuffle_8.txt'            
        elif konf ==9:
            self.annotations_file_path = '../../Data/UIEBD/UIEBD_random_shuffle_9.txt'            
        elif konf ==10:
            self.annotations_file_path = '../../Data/UIEBD/UIEBD_random_shuffle_10.txt'
            
        self.raw_data_root = '../../Data/UIEBD/raw/'
        self.gt_data_root = '../../Data/UIEBD/gt/'
        self.data_type = data_type
                
        t = []
        with open(self.annotations_file_path, 'r') as f:
            data_list = f.readlines()
            for data in data_list:
                data = data.split('\n')[0]
                t.append({'raw_data_path': self.raw_data_root + data,
                          'gt_data_path': self.gt_data_root + data})

        if self.data_type == 'train':
            self.data_infos = t[self.train_data_idx[0]:self.train_data_idx[1]]
        elif self.data_type == 'val':
            self.data_infos = t[self.val_data_idx[0]:self.val_data_idx[1]]
        elif self.data_type == 'test':
            self.data_infos = t[self.test_data_idx[0]:self.test_data_idx[1]]
                    
    def __len__(self):
        """Length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.data_infos)

    def __getitem__(self, idx):
        """Get item at each call.

        Args:
            idx (int): Index for getting each item.
        """
        
        results = copy.deepcopy(self.data_infos[idx])
        
        raw_data = Image.open(results['raw_data_path']).convert('RGB')
        gt_data = Image.open(results['gt_data_path']).convert('RGB')
        
        assert raw_data.size == gt_data.size
        
        if self.data_type == 'train':
            
            f1 = transforms.ToTensor()           
            raw_data = f1(raw_data)
            gt_data = f1(gt_data)         
            
            data = torch.cat((raw_data, gt_data), 0)
            
            f2 = transforms.RandomHorizontalFlip(p=0.5)
            data = f2(data)
            
            f3 = transforms.Resize([320, 320])
            data = f3(data)
            
            raw_data = data[0:3, :, :]
            gt_data = data[3:6, :, :]
            
            results['raw_data'] = raw_data
            results['gt_data'] = gt_data
    
        elif self.data_type == 'val':

            f1 = transforms.ToTensor()           
            raw_data = f1(raw_data)
            gt_data = f1(gt_data)         
            
            data = torch.cat((raw_data, gt_data), 0)
            
            f2 = transforms.RandomHorizontalFlip(p=0.5)
            data = f2(data)
            
            f3 = transforms.Resize([320, 320])
            data = f3(data)
            
            raw_data = data[0:3, :, :]
            gt_data = data[3:6, :, :]
            
            results['raw_data'] = raw_data
            results['gt_data'] = gt_data
            
        elif self.data_type == 'test':
                        
            f1 = transforms.ToTensor()           
            raw_data = f1(raw_data)
            gt_data = f1(gt_data)         
            
            data = torch.cat((raw_data, gt_data), 0)
            
            # f2 = transforms.RandomHorizontalFlip(p=0.5)
            # data = f2(data)
            
            f3 = transforms.Resize([320, 320])
            data = f3(data)
            
            raw_data = data[0:3, :, :]
            gt_data = data[3:6, :, :]
            
            results['raw_data'] = raw_data
            results['gt_data'] = gt_data
                
        return results
    
    
    
    
