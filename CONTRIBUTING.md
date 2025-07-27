# Contributing to HTML RAG Pipeline

Thank you for your interest in contributing to the HTML RAG Pipeline! This document provides guidelines and instructions for contributing to the project.

## 🚀 Quick Start

1. **Fork and Clone**
   ```bash
   git clone https://github.com/yourusername/html-rag-pipeline.git
   cd html-rag-pipeline
   ```

2. **Set up Development Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -e ".[dev]"
   ```

3. **Set up Pre-commit Hooks**
   ```bash
   pre-commit install
   ```

## 📋 Development Workflow

### Branch Strategy

- `main` - Production-ready code
- `develop` - Integration branch for features
- `feature/feature-name` - Feature development
- `bugfix/bug-description` - Bug fixes
- `hotfix/critical-fix` - Critical production fixes

### Making Changes

1. **Create a Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Your Changes**
   - Write clear, concise code
   - Follow existing code style
   - Add tests for new functionality
   - Update documentation as needed

3. **Test Your Changes**
   ```bash
   # Run unit tests
   pytest tests/unit/ -v
   
   # Run integration tests
   pytest tests/integration/ -v
   
   # Run all tests with coverage
   pytest --cov=html_rag --cov-report=html
   
   # Run specific test types
   pytest -m "ukrainian"  # Ukrainian content tests
   pytest -m "wayback"    # Wayback Machine tests
   pytest -m "slow"       # Performance tests
   ```

4. **Check Code Quality**
   ```bash
   # Format code
   black src/ tests/ examples/
   
   # Sort imports
   isort src/ tests/ examples/
   
   # Lint code
   flake8 src/ tests/ examples/
   
   # Type checking
   mypy src/html_rag/
   
   # Security check
   bandit -r src/
   
   # Run all pre-commit hooks
   pre-commit run --all-files
   ```

5. **Commit Your Changes**
   ```bash
   git add .
   git commit -m "feat: add new feature description"
   ```

   Use conventional commit messages:
   - `feat:` - New features
   - `fix:` - Bug fixes
   - `docs:` - Documentation changes
   - `style:` - Code style changes
   - `refactor:` - Code refactoring
   - `test:` - Test additions/modifications
   - `perf:` - Performance improvements

6. **Push and Create Pull Request**
   ```bash
   git push origin feature/your-feature-name
   ```

## 🧪 Testing Guidelines

### Test Structure

```
tests/
├── conftest.py              # Shared fixtures and configuration
├── unit/                    # Unit tests
│   ├── test_pipeline.py
│   ├── test_processors.py
│   └── ...
├── integration/             # Integration tests
│   ├── test_full_pipeline.py
│   └── ...
└── fixtures/                # Test data and fixtures
    └── sample_data/
```

### Test Categories

- **Unit Tests** (`-m unit`) - Fast, isolated component tests
- **Integration Tests** (`-m integration`) - Full workflow tests
- **Ukrainian Content Tests** (`-m ukrainian`) - Ukrainian language processing
- **Wayback Tests** (`-m wayback`) - Wayback Machine functionality
- **Slow Tests** (`-m slow`) - Performance and large-scale tests

### Writing Tests

1. **Use Descriptive Names**
   ```python
   def test_ukrainian_content_preserves_cyrillic_characters():
       # Test implementation
   ```

2. **Use Fixtures**
   ```python
   def test_pipeline_processing(pipeline, sample_html):
       result = pipeline.process_html(sample_html)
       assert result['success'] is True
   ```

3. **Test Edge Cases**
   - Empty inputs
   - Malformed HTML
   - Large documents
   - Error conditions

4. **Add Markers**
   ```python
   @pytest.mark.ukrainian
   @pytest.mark.slow
   def test_large_ukrainian_document():
       # Test implementation
   ```

## 📝 Code Style Guidelines

### Python Style

- Follow PEP 8 with 100 character line limit
- Use type hints for all function signatures
- Write comprehensive docstrings
- Prefer composition over inheritance
- Use descriptive variable names

### Code Organization

```python
"""
Module docstring describing the purpose.
"""

from typing import List, Dict, Any, Optional
import logging

from .exceptions import PipelineError
from .utils import validate_input

logger = logging.getLogger(__name__)


class ProcessorClass:
    """Class docstring with purpose and usage."""
    
    def __init__(self, config: ProcessorConfig) -> None:
        """Initialize with configuration."""
        self.config = config
    
    def process_data(self, data: str) -> Dict[str, Any]:
        """
        Process input data and return results.
        
        Args:
            data: Input data to process
            
        Returns:
            Dictionary with processing results
            
        Raises:
            PipelineError: If processing fails
        """
        try:
            # Implementation
            return {"success": True}
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            raise PipelineError(f"Processing failed: {e}") from e
```

### Documentation Style

- Use Google-style docstrings
- Include type information
- Provide examples for complex functions
- Document exceptions that can be raised

## 🐛 Bug Reports

When reporting bugs, please include:

1. **Environment Information**
   - Python version
   - Operating system
   - Package versions (`pip freeze`)

2. **Reproduction Steps**
   - Minimal code example
   - Input data (if applicable)
   - Expected vs actual behavior

3. **Error Messages**
   - Full traceback
   - Log outputs (if available)

## 💡 Feature Requests

For new features:

1. **Check Existing Issues** - Avoid duplicates
2. **Describe the Use Case** - Why is this needed?
3. **Propose Implementation** - How should it work?
4. **Consider Alternatives** - Are there other solutions?

## 🏗️ Architecture Guidelines

### Core Principles

1. **Modularity** - Each component has a single responsibility
2. **Extensibility** - Easy to add new processors or features
3. **Testability** - All components are easily testable
4. **Configuration** - Behavior is configurable without code changes
5. **Error Handling** - Graceful failure with meaningful messages

### Adding New Processors

1. **Create Processor Class**
   ```python
   class NewProcessor:
       def __init__(self, config: ProcessorConfig):
           self.config = config
       
       def process(self, input_data: Any) -> Any:
           # Implementation
           pass
   ```

2. **Add Configuration**
   ```python
   @dataclass
   class ProcessorConfig:
       new_processor_setting: str = "default"
   ```

3. **Integrate with Pipeline**
   ```python
   class RAGPipeline:
       def __init__(self):
           self.new_processor = NewProcessor(config.new_processor)
   ```

4. **Add Tests**
   ```python
   class TestNewProcessor:
       def test_basic_processing(self):
           # Test implementation
   ```

### Adding Ukrainian Language Features

When adding Ukrainian language support:

1. **Use Basic Cleaning** - Prefer basic HTML cleaning over AI models
2. **Test with Real Content** - Use actual Ukrainian text in tests
3. **Handle Encoding** - Ensure proper UTF-8 handling
4. **Consider Semantics** - Ukrainian-specific semantic features

## 🚢 Release Process

### Version Numbering

We follow [Semantic Versioning](https://semver.org/):
- `MAJOR.MINOR.PATCH`
- `MAJOR` - Breaking changes
- `MINOR` - New features (backward compatible)
- `PATCH` - Bug fixes (backward compatible)

### Release Checklist

1. **Update Version Numbers**
   - `pyproject.toml`
   - `src/html_rag/__init__.py`

2. **Update Changelog**
   - Add new features
   - List bug fixes
   - Note breaking changes

3. **Run Full Test Suite**
   ```bash
   pytest tests/ -v
   pytest -m "slow"
   pytest -m "ukrainian"
   pytest -m "wayback"
   ```

4. **Build and Test Package**
   ```bash
   python -m build
   twine check dist/*
   ```

5. **Create Release**
   - Tag commit: `git tag v1.2.3`
   - Push tag: `git push origin v1.2.3`
   - GitHub Actions will handle publishing

## 🔧 Development Tools

### Docker Development

```bash
# Build development image
docker build --target development -t html-rag-dev .

# Run with mounted code
docker run -v $(pwd):/app -it html-rag-dev bash

# Run tests in container
docker run html-rag-dev pytest tests/
```

### Performance Profiling

```bash
# Run performance monitoring
python examples/performance_monitoring.py

# Profile specific functions
python -m cProfile -o profile.stats script.py
```

### Database Management

```bash
# Reset development database
rm -rf ./chroma_db

# Export data for testing
python -c "
from html_rag import create_pipeline
p = create_pipeline()
p.export_documents('test_data.json')
"
```

## 📞 Getting Help

- **Discussions** - Use GitHub Discussions for questions
- **Issues** - Report bugs or request features
- **Documentation** - Check the docs at https://html-rag-pipeline.readthedocs.io/
- **Examples** - See `examples/` directory for usage patterns

## 🙏 Recognition

Contributors are recognized in:
- README.md contributors section
- Release notes
- Documentation acknowledgments

Thank you for contributing to HTML RAG Pipeline! 🎉