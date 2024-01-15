import sys
import os 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import torch 
from interpolate import interpolate_weights

import unittest
import torch

# Assuming interpolate_weights function is defined here

class TestInterpolateWeights(unittest.TestCase):

    def setUp(self):
        # Sample state dicts for testing
        self.model1_state_dict = {
            'layer1.weight': torch.rand(10, 10),
            'layer1.bias': torch.rand(10),
        }
        self.model2_state_dict = {
            'layer1.weight': torch.rand(10, 10),
            'layer1.bias': torch.rand(10),
        }

    def test_basic_functionality(self):
        alpha = 0.5
        interpolated = interpolate_weights(self.model1_state_dict, self.model2_state_dict, alpha)
        for param in interpolated:
            expected = alpha * self.model1_state_dict[param] + (1 - alpha) * self.model2_state_dict[param]
            self.assertTrue(torch.allclose(interpolated[param], expected))

    def test_edge_cases(self):
        for alpha in [0.0, 1.0]:
            interpolated = interpolate_weights(self.model1_state_dict, self.model2_state_dict, alpha)
            expected_dict = self.model1_state_dict if alpha == 1 else self.model2_state_dict
            for param in interpolated:
                # print(interpolated[param] - expected_dict[param])
                self.assertTrue(torch.allclose(interpolated[param], expected_dict[param]))

    def test_structure_mismatch(self):
        mismatched_dict = {
            'layer1.weight': torch.rand(10, 10),
            # Missing 'layer1.bias'
        }
        with self.assertRaises(AssertionError):
            interpolate_weights(self.model1_state_dict, mismatched_dict, 0.5)

    def test_type_and_shape_consistency(self):
        alpha = 0.3
        interpolated = interpolate_weights(self.model1_state_dict, self.model2_state_dict, alpha)
        for param in interpolated:
            self.assertEqual(interpolated[param].shape, self.model1_state_dict[param].shape)
            self.assertEqual(interpolated[param].type(), self.model1_state_dict[param].type())

if __name__ == '__main__':
    unittest.main()
    # passed 
