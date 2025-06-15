# Test Suite for Chatbot Boilerplate

This directory contains comprehensive test cases for the chatbot boilerplate project. The test suite is designed to ensure reliability, maintainability, and correct functionality across all components.

## 📁 Test Structure

```
tests/
├── conftest.py              # Pytest fixtures and configuration
├── test_chat_utils.py       # Tests for utility functions
├── test_chat_state.py       # Tests for state classes and data models
├── test_chat_nodes.py       # Tests for processing nodes
├── test_chat_graph.py       # Tests for graph builder and workflow
├── test_chat_agent.py       # Tests for main ChatAgent class
├── test_edge_cases.py       # Edge cases and error condition tests
└── README.md               # This file
```

## 🧪 Test Categories

### Unit Tests
- **`test_chat_utils.py`**: Tests utility functions like system prompt generation
- **`test_chat_state.py`**: Tests state classes (ChatState, SessionInfo)
- **`test_chat_nodes.py`**: Tests individual processing nodes

### Integration Tests
- **`test_chat_agent.py`**: Tests the main ChatAgent orchestrator
- **`test_chat_graph.py`**: Tests graph workflow integration

### Edge Case Tests
- **`test_edge_cases.py`**: Tests boundary conditions, error handling, and performance

## 🚀 Running Tests

### Prerequisites

Install test dependencies:
```bash
pip install pytest pytest-asyncio pytest-cov
```

### Basic Usage

```bash
# Run all tests
python -m pytest tests/

# Run with coverage
python -m pytest tests/ --cov=app --cov-report=term-missing

# Run specific test file
python -m pytest tests/test_chat_agent.py

# Run with verbose output
python -m pytest tests/ -v
```

### Using the Test Runner Script

```bash
# Run all tests
python run_tests.py

# Run only unit tests
python run_tests.py --unit

# Run only integration tests
python run_tests.py --integration

# Run with coverage report
python run_tests.py --coverage

# Run specific test file
python run_tests.py --file tests/test_chat_agent.py
```

## 📊 Test Coverage

The test suite aims for comprehensive coverage of:

- ✅ **Core Functionality**: All main features and workflows
- ✅ **Error Handling**: Exception handling and graceful degradation
- ✅ **Edge Cases**: Boundary conditions and unusual inputs
- ✅ **Integration**: Component interaction and data flow
- ✅ **Session Management**: Multi-session handling and isolation
- ✅ **User Types**: Different user type behaviors
- ✅ **API Fallbacks**: Behavior without API keys

## 🧰 Test Fixtures

The `conftest.py` file provides shared fixtures:

- **`mock_openai_llm`**: Mocked OpenAI LLM for testing
- **`chat_nodes`**: Configured ChatNodes instance
- **`initialized_chat_agent`**: Ready-to-use ChatAgent
- **`sample_chat_state`**: Sample state for testing
- **`memory_saver`**: MemorySaver instance

## ⚡ Performance Tests

Performance-related tests include:

- **Memory Usage**: Tests with multiple sessions
- **Concurrent Operations**: Parallel session handling
- **Message Count Accuracy**: Proper session tracking
- **State Consistency**: Data integrity across operations

## 🐛 Error Condition Tests

Error handling tests cover:

- **API Failures**: OpenAI API timeouts and errors
- **Invalid Inputs**: Malformed messages and session IDs
- **Memory Corruption**: Handling of corrupted state
- **Concurrent Modifications**: Race condition handling

## 📝 Writing New Tests

### Test Naming Convention

- Test files: `test_<component>.py`
- Test classes: `Test<ComponentName>`
- Test methods: `test_<specific_behavior>`

### Example Test Structure

```python
class TestMyComponent:
    """Test cases for MyComponent."""
    
    def test_basic_functionality(self, fixture_name):
        """Test basic functionality."""
        # Arrange
        input_data = "test input"
        
        # Act
        result = my_component.process(input_data)
        
        # Assert
        assert result is not None
        assert isinstance(result, str)
        
    @pytest.mark.asyncio
    async def test_async_functionality(self, async_fixture):
        """Test async functionality."""
        result = await my_component.async_process()
        assert result["status"] == "success"
```

### Async Test Guidelines

- Use `@pytest.mark.asyncio` for async tests
- Use `AsyncMock` for mocking async functions
- Test both success and failure scenarios

### Mock Usage

```python
from unittest.mock import Mock, AsyncMock

# Mock synchronous calls
mock_obj = Mock()
mock_obj.method.return_value = "expected_result"

# Mock asynchronous calls
mock_obj = Mock()
mock_obj.async_method = AsyncMock(return_value="async_result")
```

## 🔧 Configuration

### Pytest Configuration

Test configuration is defined in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = [
    "-ra",
    "--strict-markers",
    "--strict-config",
    "--cov=app",
    "--cov-report=term-missing",
]
```

### Environment Variables

Tests automatically handle environment variables:

- `OPENAI_API_KEY`: Mocked for testing to avoid API calls
- Tests run with and without API keys to test fallback behavior

## 📈 Continuous Integration

For CI/CD integration, use:

```yaml
# Example GitHub Actions workflow
- name: Run tests
  run: |
    pip install -r requirements.txt
    pip install pytest pytest-asyncio pytest-cov
    python run_tests.py --coverage
```

## 🤝 Contributing

When adding new features:

1. **Write tests first** (TDD approach)
2. **Test both success and failure paths**
3. **Include edge cases**
4. **Add integration tests** for component interactions
5. **Update this README** if adding new test categories

### Test Checklist

- [ ] Unit tests for new functions/methods
- [ ] Integration tests for component interactions
- [ ] Error handling tests
- [ ] Edge case tests
- [ ] Async functionality tests (if applicable)
- [ ] Mock external dependencies
- [ ] Clear test names and documentation

## 🐛 Debugging Tests

### Common Issues

1. **Import Errors**: Ensure proper Python path setup
2. **Async Test Failures**: Use `@pytest.mark.asyncio`
3. **Mock Issues**: Verify mock setup and return values
4. **Fixture Problems**: Check fixture scope and dependencies

### Debugging Commands

```bash
# Run single test with full output
python -m pytest tests/test_chat_agent.py::TestChatAgent::test_basic_functionality -v -s

# Run with pdb debugger
python -m pytest tests/test_chat_agent.py --pdb

# Show local variables on failure
python -m pytest tests/test_chat_agent.py --tb=long
```

## 📊 Coverage Reports

After running tests with coverage:

```bash
# View coverage in terminal
python run_tests.py --coverage

# Generate HTML coverage report
python -m pytest tests/ --cov=app --cov-report=html

# Open HTML report
open htmlcov/index.html  # macOS
# or
xdg-open htmlcov/index.html  # Linux
```

## 🎯 Test Goals

- **90%+ Code Coverage**: Comprehensive test coverage
- **Fast Execution**: Tests complete quickly for rapid feedback
- **Reliable**: Tests are stable and not flaky
- **Maintainable**: Tests are easy to understand and modify
- **Comprehensive**: Cover all user scenarios and edge cases 