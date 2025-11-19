<!--- SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved. --->

<!--- SPDX-License-Identifier: MIT --->

# TileGym Testing Framework

This documentation provides guidelines for running tests and contributing new tests to the TileGym framework.

## Running Tests

### Functional Tests
To run functionality tests:
```bash
pytest your_test_name -k test_op -v --log-cli-level=INFO
```

## Contributing Tests

To ensure compatibility with our testing framework, please follow these guidelines:

### Naming Conventions
- Test classes should start with `Test_`, e.g., `Test_RoPE`, `Test_Matmul`
- Implement test methods with appropriate names:
  - `test_op` - For functional correctness tests

### Class Structure
- Inherit from `common.PyTestCase`
- Implement a `reference` method for PyTorch baseline implementation
- Use `@pytest.mark.parametrize` for test case variations
- If you specify the value of parameter in `@pytest.mark.parametrize`, function should not contain default value


### Example Test Structure

```python
class Test_YourFeature(common.PyTestCase):
    @staticmethod
    def reference(x, ...):
        # Reference implementation using PyTorch
        return torch_result

    @pytest.mark.parametrize("param1,param2,...", [
        (value1, value2, ...),
        (value3, value4, ...),
    ])
    def test_op(self, param1, param2, ...):
        # Test for functional correctness
        self.assertCorrectness(
            tilegym.ops.your_op,
            self.reference,
            {'arg1': value1, 'arg2': value2, ...},
            rtol=1e-2,
            atol=1e-2,
        )

```

## Advanced Usage

- Use `self.assertCorrectness()` to compare TileGym implementation with reference
- For gradient checking, provide the `gradient` parameter to `assertCorrectness`
- Set appropriate tolerance values (`rtol`, `atol`) based on numeric precision needs
