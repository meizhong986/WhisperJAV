#!/usr/bin/env python3
"""Test CUDA version comparison fix in preflight checks."""

import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from whisperjav.utils.preflight_check import PreflightChecker, CheckStatus


class TestCudaVersionComparison(unittest.TestCase):
    """Test the CUDA version comparison logic."""
    
    def test_cuda_version_match_integer_to_string(self):
        """Test that integer compiled CUDA version matches string runtime version."""
        with patch('whisperjav.utils.preflight_check.sys') as mock_sys:
            mock_sys.version_info = (3, 10, 0)
            
            # Create mock torch module
            mock_torch = MagicMock()
            mock_torch.cuda.is_available.return_value = True
            mock_torch.cuda.device_count.return_value = 1
            mock_torch.cuda.get_device_name.return_value = "Test GPU"
            
            # This is the key part: compiled version is int 12090, runtime is string "12.9"
            mock_torch._C._cuda_getCompiledVersion.return_value = 12090
            mock_torch.version.cuda = "12.9"
            
            # Mock CUDA operations
            mock_tensor = MagicMock()
            mock_torch.zeros.return_value = mock_tensor
            mock_tensor.cuda.return_value = mock_tensor
            
            with patch.dict('sys.modules', {'torch': mock_torch}):
                checker = PreflightChecker(verbose=False)
                checker._check_pytorch_cuda()
                
                # Find the PyTorch CUDA Build result
                cuda_build_result = None
                for result in checker.results:
                    if result.name == "PyTorch CUDA Build":
                        cuda_build_result = result
                        break
                
                self.assertIsNotNone(cuda_build_result, "PyTorch CUDA Build check should exist")
                self.assertEqual(cuda_build_result.status, CheckStatus.PASS, 
                               "CUDA version 12090 (int) should match '12.9' (string)")
                self.assertIn("12.9", cuda_build_result.message)
    
    def test_cuda_version_mismatch(self):
        """Test that actual CUDA version mismatches are still detected."""
        with patch('whisperjav.utils.preflight_check.sys') as mock_sys:
            mock_sys.version_info = (3, 10, 0)
            
            # Create mock torch module
            mock_torch = MagicMock()
            mock_torch.cuda.is_available.return_value = True
            mock_torch.cuda.device_count.return_value = 1
            mock_torch.cuda.get_device_name.return_value = "Test GPU"
            
            # Different versions: 11080 -> "11.8", but runtime is "12.1"
            mock_torch._C._cuda_getCompiledVersion.return_value = 11080
            mock_torch.version.cuda = "12.1"
            
            # Mock CUDA operations
            mock_tensor = MagicMock()
            mock_torch.zeros.return_value = mock_tensor
            mock_tensor.cuda.return_value = mock_tensor
            
            with patch.dict('sys.modules', {'torch': mock_torch}):
                checker = PreflightChecker(verbose=False)
                checker._check_pytorch_cuda()
                
                # Find the PyTorch CUDA Build result
                cuda_build_result = None
                for result in checker.results:
                    if result.name == "PyTorch CUDA Build":
                        cuda_build_result = result
                        break
                
                self.assertIsNotNone(cuda_build_result, "PyTorch CUDA Build check should exist")
                self.assertEqual(cuda_build_result.status, CheckStatus.WARN, 
                               "Different CUDA versions should produce a warning")
                self.assertIn("11.8", str(cuda_build_result.details))
                self.assertIn("12.1", str(cuda_build_result.details))
    
    def test_cuda_version_conversion_edge_cases(self):
        """Test various CUDA version integer conversions."""
        test_cases = [
            (12090, "12.9"),  # From the issue screenshot
            (12010, "12.1"),
            (11080, "11.8"),
            (12000, "12.0"),
            (13020, "13.2"),
        ]
        
        for compiled_int, expected_str in test_cases:
            with self.subTest(compiled=compiled_int, expected=expected_str):
                major = compiled_int // 1000
                minor = (compiled_int % 1000) // 10
                result = f"{major}.{minor}"
                self.assertEqual(result, expected_str, 
                               f"Integer {compiled_int} should convert to '{expected_str}'")


if __name__ == '__main__':
    unittest.main()
