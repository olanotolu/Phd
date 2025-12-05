# Contributing to TLM / Babel Engine

Thank you for your interest in contributing! This is early-stage research, and we welcome contributions.

## Areas for Contribution

We're especially interested in:

- **New factor types**: trigrams, long-range interactions, hierarchical factors
- **Better samplers**: improved Gibbs sampling, parallel sampling, hardware optimization
- **Visualization tools**: energy landscapes, pattern emergence, interactive demos
- **Real-data experiments**: training on Wikipedia, code, domain-specific corpora
- **Training improvements**: contrastive divergence, score matching, better optimization
- **Documentation**: tutorials, examples, architecture explanations

## Development Setup

1. Fork the repository
2. Clone your fork: `git clone https://github.com/yourname/tlm-babel.git`
3. Install dependencies: `pip install -r requirements.txt`
4. Create a branch: `git checkout -b feature/your-feature-name`
5. Make your changes
6. Test: `python example.py`
7. Submit a pull request

## Code Style

- Follow PEP 8 for Python code
- Use type hints where helpful
- Add docstrings to functions and classes
- Keep functions focused and modular

## Testing

Before submitting:

- Run `python example.py` to ensure basic functionality
- Test your changes with the notebook `babel_exploration.ipynb`
- Check that visualizations still work

## Pull Request Process

1. Update README.md if needed
2. Add tests/examples if applicable
3. Update documentation
4. Ensure code runs without errors
5. Submit PR with clear description

## Questions?

Open an issue with the `question` label, or reach out directly.

Thank you for contributing to thermodynamic language models!

