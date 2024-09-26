#!/usr/bin/env python3
"""Test suite for the Neuron class"""

import unittest
import numpy as np
fro import Neuron  # Replace 'neuron' with the filename of your Neuron class script

class TestNeuron(unittest.TestCase):
    """Unit tests for the Neuron class"""

    def test_init_valid(self):
        """Test Neuron initialization with valid nx"""
        nx = 5
        neuron = Neuron(nx)
        self.assertEqual(neuron.W.shape, (1, nx))
        self.assertEqual(neuron.b, 0)
        self.assertEqual(neuron.A, 0)

    def test_init_invalid_nx_type(self):
        """Test Neuron initialization with invalid nx type"""
        with self.assertRaises(TypeError):
            Neuron("5")

    def test_init_invalid_nx_value(self):
        """Test Neuron initialization with nx less than 1"""
        with self.assertRaises(ValueError):
            Neuron(0)

    def test_W_property(self):
        """Test the W property getter"""
        nx = 3
        neuron = Neuron(nx)
        self.assertTrue(np.array_equal(neuron.W, neuron._Neuron__W))

    def test_b_property(self):
        """Test the b property getter"""
        neuron = Neuron(3)
        self.assertEqual(neuron.b, neuron._Neuron__b)

    def test_A_property(self):
        """Test the A property getter"""
        neuron = Neuron(3)
        self.assertEqual(neuron.A, neuron._Neuron__A)

    def test_forward_prop(self):
        """Test forward propagation"""
        np.random.seed(0)
        nx, m = 3, 2
        neuron = Neuron(nx)
        X = np.random.randn(nx, m)
        A = neuron.forward_prop(X)
        self.assertEqual(A.shape, (1, m))
        self.assertTrue((A >= 0).all() and (A <= 1).all())

    def test_cost(self):
        """Test cost calculation"""
        neuron = Neuron(3)
        Y = np.array([[1, 0]])
        A = np.array([[0.8, 0.2]])
        cost = neuron.cost(Y, A)
        expected_cost = -np.mean(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        self.assertAlmostEqual(cost, expected_cost, places=7)

    def test_evaluate(self):
        """Test the evaluate method"""
        np.random.seed(0)
        nx, m = 3, 5
        neuron = Neuron(nx)
        X = np.random.randn(nx, m)
        Y = np.random.randint(0, 2, (1, m))
        A, cost = neuron.evaluate(X, Y)
        self.assertEqual(A.shape, Y.shape)
        self.assertIsInstance(cost, float)

    def test_gradient_descent(self):
        """Test one pass of gradient descent"""
        neuron = Neuron(3)
        X = np.array([[1, 2, -1], [3, -2, 0], [2, 1, -3]])
        Y = np.array([[1, 0, 1]])
        A = neuron.forward_prop(X)
        old_W = neuron.W.copy()
        old_b = neuron.b
        neuron.gradient_descent(X, Y, A, alpha=0.05)
        self.assertFalse(np.array_equal(neuron.W, old_W))
        self.assertNotEqual(neuron.b, old_b)

    def test_train(self):
        """Test the train method"""
        np.random.seed(0)
        nx, m = 2, 10
        neuron = Neuron(nx)
        X = np.random.randn(nx, m)
        Y = np.random.randint(0, 2, (1, m))
        A, cost = neuron.train(X, Y, iterations=100, alpha=0.05, verbose=False, graph=False)
        self.assertEqual(A.shape, Y.shape)
        self.assertIsInstance(cost, float)

    def test_train_invalid_iterations_type(self):
        """Test train method with invalid iterations type"""
        neuron = Neuron(3)
        X = np.random.randn(3, 5)
        Y = np.random.randint(0, 2, (1, 5))
        with self.assertRaises(TypeError):
            neuron.train(X, Y, iterations="1000")

    def test_train_invalid_iterations_value(self):
        """Test train method with invalid iterations value"""
        neuron = Neuron(3)
        X = np.random.randn(3, 5)
        Y = np.random.randint(0, 2, (1, 5))
        with self.assertRaises(ValueError):
            neuron.train(X, Y, iterations=0)

    def test_train_invalid_alpha_type(self):
        """Test train method with invalid alpha type"""
        neuron = Neuron(3)
        X = np.random.randn(3, 5)
        Y = np.random.randint(0, 2, (1, 5))
        with self.assertRaises(TypeError):
            neuron.train(X, Y, alpha="0.05")

    def test_train_invalid_alpha_value(self):
        """Test train method with invalid alpha value"""
        neuron = Neuron(3)
        X = np.random.randn(3, 5)
        Y = np.random.randint(0, 2, (1, 5))
        with self.assertRaises(ValueError):
            neuron.train(X, Y, alpha=0)

    def test_train_invalid_step_type(self):
        """Test train method with invalid step type"""
        neuron = Neuron(3)
        X = np.random.randn(3, 5)
        Y = np.random.randint(0, 2, (1, 5))
        with self.assertRaises(TypeError):
            neuron.train(X, Y, step="10")

    def test_train_invalid_step_value(self):
        """Test train method with invalid step value"""
        neuron = Neuron(3)
        X = np.random.randn(3, 5)
        Y = np.random.randint(0, 2, (1, 5))
        with self.assertRaises(ValueError):
            neuron.train(X, Y, iterations=100, step=0)

    def test_train_step_greater_than_iterations(self):
        """Test train method with step greater than iterations"""
        neuron = Neuron(3)
        X = np.random.randn(3, 5)
        Y = np.random.randint(0, 2, (1, 5))
        with self.assertRaises(ValueError):
            neuron.train(X, Y, iterations=100, step=200)

    def test_verbose_output(self):
        """Test verbose output during training"""
        import io
        import sys
        neuron = Neuron(3)
        X = np.random.randn(3, 5)
        Y = np.random.randint(0, 2, (1, 5))
        captured_output = io.StringIO()
        sys.stdout = captured_output
        neuron.train(X, Y, iterations=10, alpha=0.05, verbose=True, graph=False, step=2)
        sys.stdout = sys.__stdout__
        output = captured_output.getvalue()
        self.assertIn("Cost after 0 iterations:", output)
        self.assertIn("Cost after 10 iterations:", output)

    def test_graph_output(self):
        """Test that graphing does not raise errors"""
        import matplotlib
        matplotlib.use('Agg')  # Use a non-interactive backend
        neuron = Neuron(3)
        X = np.random.randn(3, 5)
        Y = np.random.randint(0, 2, (1, 5))
        try:
            neuron.train(X, Y, iterations=10, alpha=0.05, verbose=False, graph=True, step=2)
        except Exception as e:
            self.fail(f"train method with graph=True raised an exception: {e}")

if __name__ == '__main__':
    unittest.main()
