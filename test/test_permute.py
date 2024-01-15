import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import unittest
import numpy as np
import sys
import os 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(2, 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(2, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x
    
    
    
class TestPermute(unittest.TestCase):
    def setUp(self):
        self.model_builder = SimpleModel
        self.model1 = self.model_builder()
        self.model2 = self.model_builder()
        self.model2.load_state_dict(self.model1.state_dict())
        from permute import model_permutation
        self.permutation_indices = [
           np.array([1, 0])
        ]
        permuted_state_dict = model_permutation(self.model2, 
                                           self.permutation_indices)
        self.model2.load_state_dict(permuted_state_dict)
        x = torch.randn(20, 2)
        y = torch.randint(0, 2, (20,))
        dummy_dataset = torch.utils.data.TensorDataset(x, y)
        self.val_loader = torch.utils.data.DataLoader(
            dummy_dataset, batch_size=5, shuffle=False
        )
        
        
    def test_functionality(self):
        from permute import match_and_permute
        returned_state_dict = match_and_permute(
            self.model_builder,
            self.model1.state_dict(),
            self.model2.state_dict(),
            self.val_loader,
            device="cpu",
        )
        for name in returned_state_dict:
            self.assertTrue(torch.allclose(returned_state_dict[name], self.model1.state_dict()[name]))        
if __name__ == "__main__":
    unittest.main()
   
        