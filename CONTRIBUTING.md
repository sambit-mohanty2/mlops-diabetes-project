# Diabetic Risk Prediction - MLOps Project

## Contributing Guidelines

Thank you for your interest in contributing to the Diabetic Risk Prediction project! We welcome contributions from everyone.

### Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/mlops-diabetes-project.git`
3. Create a virtual environment: `python -m venv venv`
4. Activate the environment: `source venv/bin/activate`
5. Install dependencies: `pip install -r requirements-dev.txt`

### Development Workflow

1. Create a feature branch: `git checkout -b feature/your-feature`
2. Make your changes
3. Write tests for new functionality
4. Run tests: `pytest tests/ -v --cov=src`
5. Format code: `black src/ tests/`
6. Run linting: `flake8 src/ tests/`
7. Commit changes: `git commit -m "Add your feature"`
8. Push to your fork: `git push origin feature/your-feature`
9. Create a Pull Request

### Code Standards

- Use PEP 8 style guide
- Add docstrings to all functions and classes
- Write type hints where applicable
- Keep functions small and focused
- Add unit tests for new code

### Testing

- All new features must include tests
- Run tests before submitting PR: `pytest`
- Maintain code coverage above 80%

### Commit Messages

- Use clear, descriptive commit messages
- Start with a verb: "Add", "Fix", "Update", etc.
- Reference issues when applicable: "Fixes #123"

### Pull Request Process

1. Ensure all tests pass
2. Update documentation if needed
3. Add description of changes
4. Link related issues
5. Wait for review and address feedback

### Issues

- Use GitHub Issues for bug reports and feature requests
- Provide clear description and reproducible steps
- Include environment details

### License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Questions? Open an issue or contact the maintainers!
