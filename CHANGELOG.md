# Changelog

All notable changes to TLM / Babel Engine will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-01-XX

### Added
- Initial release of Babel Engine v0.1
- Character-level Thermodynamic Language Model (TLM)
- Bigram factor interactions
- Block Gibbs sampling implementation
- Training via pseudo-likelihood and KL-gradient estimation
- Pattern analysis and visualization tools
- Interactive Jupyter notebook
- Complete documentation:
  - Architecture specification
  - Branding guide
  - Citation information
  - Getting started guide
  - Contributing guidelines

### Features
- Generate random "Library of Babel" sequences
- Train EBM on symbolic sequences
- Sample structured patterns from trained model
- Analyze emergent patterns (bigrams, motifs, character clusters)
- Visualize energy landscapes and pattern emergence
- Compare random data vs model samples

### Technical Details
- Alphabet: 29 characters (a-z, space, comma, period)
- Energy function: bigram interactions
- Sampling: block Gibbs with temperature control
- Training: pseudo-likelihood (fast) and KL-gradient (accurate)

### Documentation
- README with project overview
- ARCHITECTURE.md with technical specification
- BRANDING.md with naming and messaging
- CITATION.md with citation formats
- CONTRIBUTING.md with contribution guidelines
- docs/GETTING_STARTED.md with user guide

## [Unreleased]

### Planned for v0.2
- Training on real text corpora (Wikipedia, domain-specific)
- Improved visualization tools
- Better documentation and examples

### Planned for v0.3
- Multi-scale factors (trigrams, span-level)
- Long-range interactions
- Hierarchical structure

### Planned for v0.4
- Contrastive divergence training
- Score matching
- Better optimization methods

### Planned for v0.5
- Anomaly detection on LLM outputs
- Conditional generation
- Real-world applications

---

[0.1.0]: https://github.com/yourname/tlm-babel/releases/tag/v0.1.0

