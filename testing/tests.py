import unittest

import sys
import os
sys.path.insert(0, os.path.abspath('..'))
from src import *


class TestDistributions(unittest.TestCase):
    """Unit tests for distribution estimation with large n."""
    
    def test_normal_distribution_mean(self):
        """Test that Normal distribution mean estimate is close to analytical truth."""
        mu = 5.0
        sigma = 2.0
        n = 10000  
        
        # Create Normal with known parameters
        normal = Normal(mu, False, sigma, False)  # parameters not marked as known to test estimation
        prog = Dist(normal, n)
        
        # Estimate mean
        estimated_mean = prog.estimate()
        
        # Analytical truth: mean of Normal(mu, sigma) is mu
        analytical_mean = mu
        
        # Check that estimate is close to analytical truth (within 3 standard errors)
        # Standard error of mean = sigma / sqrt(n)
        std_error = sigma / (n ** 0.5)
        tolerance = 3 * std_error
        
        self.assertAlmostEqual(
            estimated_mean, 
            analytical_mean, 
            delta=tolerance,
            msg=f"Estimated mean {estimated_mean} should be close to analytical mean {analytical_mean}"
        )
    
    def test_normal_distribution_variance(self):
        """Test that Normal distribution variance estimate is close to analytical truth."""
        mu = 3.0
        sigma = 1.5
        n = 10000  # large n for better accuracy
        
        # Create Normal with known parameters
        normal = Normal(mu, False, sigma, False)
        prog = normal
        
        # Estimate variance
        estimated_variance = Dist(prog.variance(), n).estimate()
        
        # Analytical truth: variance of Normal(mu, sigma) is sigma^2
        analytical_variance = sigma ** 2
        
        # For variance estimation, we need a larger tolerance
        # The variance of sample variance is approximately 2*sigma^4/(n-1) for large n
        # Using a more conservative tolerance
        tolerance = 0.5  # Allow some tolerance for variance estimation
        
        self.assertAlmostEqual(
            estimated_variance,
            analytical_variance,
            delta=tolerance,
            msg=f"Estimated variance {estimated_variance} should be close to analytical variance {analytical_variance}"
        )
    
    def test_uniform_distribution_mean(self):
        """Test that Uniform distribution mean estimate is close to analytical truth."""
        a = 0.0
        b = 10.0
        n = 10000  # large n for better accuracy
        
        # Create Uniform with known parameters
        uniform = Uniform(a, False, b, False)
        prog = Dist(uniform, n)
        
        # Estimate mean
        estimated_mean = prog.estimate()
        
        # Analytical truth: mean of Uniform(a, b) is (a + b) / 2
        analytical_mean = (a + b) / 2
        
        # Standard error for uniform: std_dev / sqrt(n) where std_dev = (b-a)/sqrt(12)
        std_dev = (b - a) / (12 ** 0.5)
        std_error = std_dev / (n ** 0.5)
        tolerance = 3 * std_error
        
        self.assertAlmostEqual(
            estimated_mean,
            analytical_mean,
            delta=tolerance,
            msg=f"Estimated mean {estimated_mean} should be close to analytical mean {analytical_mean}"
        )
    
    def test_uniform_distribution_variance(self):
        """Test that Uniform distribution variance estimate is close to analytical truth."""
        a = 0.0
        b = 10.0
        n = 10000  # large n for better accuracy
        
        # Create Uniform with known parameters
        uniform = Uniform(a, False, b, False)
        prog = uniform
        
        # Estimate variance
        estimated_variance = Dist(prog.variance(), n).estimate()
        
        # Analytical truth: variance of Uniform(a, b) is (b - a)^2 / 12
        analytical_variance = ((b - a) ** 2) / 12
        
        # Tolerance for variance estimation
        tolerance = 0.5
        
        self.assertAlmostEqual(
            estimated_variance,
            analytical_variance,
            delta=tolerance,
            msg=f"Estimated variance {estimated_variance} should be close to analytical variance {analytical_variance}"
        )
    
    def test_normal_with_known_parameters(self):
        """Test Normal distribution with known mean and variance parameters."""
        mu = 7.0
        sigma = 3.0
        n = 1000  # n can be smaller since we're using known parameters
        
        # Create Normal with known mean and variance
        normal = Normal(mu, True, sigma, True)
        prog = normal
        
        # Estimate mean (should be exact since it's known)
        estimated_mean = Dist(prog, n).estimate()
        analytical_mean = mu
        
        # Estimate variance (should be exact since it's known)
        estimated_variance = Dist(prog.variance(), n).estimate()
        analytical_variance = sigma ** 2
        
        # With known parameters, these should be exact
        self.assertEqual(estimated_mean, analytical_mean)
        self.assertEqual(estimated_variance, analytical_variance)
    
    def test_uniform_with_known_parameters(self):
        """Test Uniform distribution with known parameters."""
        a = 2.0
        b = 8.0
        n = 1000
        
        # Create Uniform with known parameters
        uniform = Uniform(a, True, b, True)
        prog = uniform
        
        # Estimate mean (should be exact)
        estimated_mean = Dist(prog, n).estimate()
        analytical_mean = (a + b) / 2
        
        # Estimate variance (should be exact)
        estimated_variance = Dist(prog.variance(), n).estimate()
        analytical_variance = ((b - a) ** 2) / 12
        
        # With known parameters, these should be exact
        self.assertEqual(estimated_mean, analytical_mean)
        self.assertEqual(estimated_variance, analytical_variance)

if __name__ == '__main__':
    unittest.main()
