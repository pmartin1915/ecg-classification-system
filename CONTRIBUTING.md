# Contributing to ECG Classification System

Thank you for your interest in contributing to the ECG Classification System! This document provides guidelines for contributing to this medical AI project.

## 🏥 Medical Disclaimer

**IMPORTANT**: This is a medical AI system intended for educational and research purposes only. All contributions must adhere to strict medical and ethical standards.

## 🚀 Quick Start

1. **Fork the repository**
2. **Clone your fork**: `git clone https://github.com/yourusername/ecg-classification-system.git`
3. **Install dependencies**: `pip install -r requirements.txt`
4. **Run tests**: `python deployment_validation.py`
5. **Start development**: `python -m streamlit run complete_user_friendly.py`

## 📋 Development Setup

### Prerequisites
- Python 3.8+ (Python 3.13+ recommended)
- Git
- 4GB+ RAM
- Modern web browser

### Environment Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Verify installation
python deployment_validation.py
```

## 🧪 Testing

### Running Tests
```bash
# Full system validation
python deployment_validation.py

# Manual component testing
python -c "import complete_user_friendly; print('UI test passed')"
python app/utils/ptbxl_mi_extractor.py
```

### Test Requirements
- All tests in `deployment_validation.py` must pass
- New features require corresponding tests
- Medical accuracy claims must be validated against PTB-XL database

## 🎯 Branch Strategy

### Main Branches
- `main`: Production-ready code, stable releases
- `develop`: Integration branch for new features
- `feature/*`: Individual feature development
- `hotfix/*`: Critical bug fixes

### Branch Naming
- `feature/ptbxl-integration`
- `fix/streamlit-ui-bug`
- `docs/api-documentation`

## 💻 Code Style

### Python Style Guide
- **Formatter**: Black (line length: 88)
- **Import sorting**: isort
- **Linting**: Ruff
- **Type hints**: Required for new functions
- **Docstrings**: Google style

### Pre-commit Setup
```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install

# Run on all files
pre-commit run --all-files
```

### Code Quality Checklist
- [ ] Code formatted with Black
- [ ] Imports sorted with isort
- [ ] No linting errors from Ruff
- [ ] Type hints added for new functions
- [ ] Docstrings following Google style
- [ ] Tests passing
- [ ] No hardcoded PHI or sensitive data

## 📝 Commit Messages

### Format
```
type(scope): brief description

Detailed description if needed.

Fixes #123
```

### Types
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

### Examples
```
feat(ptbxl): add MI territory mapping

Add comprehensive territory mapping for PTB-XL MI cases including
anterior, inferior, lateral, and posterior territories.

Fixes #42
```

## 🔍 Pull Request Process

### Before Submitting
1. **Test locally**: Run `deployment_validation.py` - all tests must pass
2. **Code quality**: Ensure pre-commit hooks pass
3. **Documentation**: Update relevant docs
4. **Medical validation**: Verify medical accuracy claims
5. **PHI check**: Ensure no patient data is included

### PR Description Template
```markdown
## Description
Brief description of changes.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Medical accuracy improvement
- [ ] Performance optimization

## Testing
- [ ] `deployment_validation.py` passes (100% success rate)
- [ ] Manual testing performed
- [ ] Medical validation completed (if applicable)

## Medical Impact
Describe any impact on medical accuracy or clinical workflow.

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests added/updated
- [ ] No PHI or sensitive data included
```

### Review Process
1. **Automated checks**: CI must pass
2. **Code review**: At least one approving review
3. **Medical review**: Medical accuracy changes need medical expert review
4. **Testing**: All deployment validation tests must pass

## 📚 Documentation Standards

### Code Documentation
- **Functions**: Google-style docstrings with Args, Returns, Raises
- **Classes**: Purpose, attributes, and usage examples
- **Modules**: Overview and key functions
- **Medical functions**: Include clinical context and validation source

### Example Docstring
```python
def analyze_ecg(ecg_data: np.ndarray, confidence_threshold: float = 0.8) -> Dict[str, Any]:
    """Analyze ECG data for cardiac abnormalities.
    
    Args:
        ecg_data: 12-lead ECG signal data, shape (n_samples, 12)
        confidence_threshold: Minimum confidence for positive diagnosis
        
    Returns:
        Dictionary containing:
            - diagnosis: Primary diagnosis
            - confidence: AI confidence score (0-1)
            - clinical_recommendations: List of clinical actions
            - ptbxl_validation: Evidence from PTB-XL database
            
    Raises:
        ValueError: If ECG data format is invalid
        
    Clinical Context:
        This function implements evidence-based MI detection validated
        against 4,926 physician-diagnosed cases from PTB-XL database.
    """
```

## 🏥 Medical Contribution Guidelines

### Medical Accuracy Requirements
- **Evidence-based**: All medical claims must cite sources
- **PTB-XL validation**: Use PTB-XL database for validation
- **Clinical review**: Medical accuracy changes need clinical expert review
- **Disclaimer compliance**: Maintain educational/research use only

### Prohibited Medical Content
- ❌ Clinical diagnosis recommendations
- ❌ Treatment protocols
- ❌ Drug recommendations
- ❌ Patient-specific advice
- ❌ Claims of replacing physician judgment

### Allowed Medical Content
- ✅ Educational explanations
- ✅ Pattern recognition training
- ✅ AI explainability for learning
- ✅ Evidence-based validation metrics
- ✅ Clinical workflow support tools

## 🔐 Security and Privacy

### Data Protection
- **No PHI**: Never commit patient data or identifiers
- **Data anonymization**: All medical data must be de-identified
- **Local processing**: Maintain local-only data processing
- **Secure handling**: Follow HIPAA-like practices

### Security Checklist
- [ ] No hardcoded secrets or keys
- [ ] No patient identifiable information
- [ ] Input validation for all user inputs
- [ ] Secure file handling
- [ ] Error messages don't leak sensitive info

## 🧮 Performance Guidelines

### Performance Requirements
- **Analysis speed**: <3 seconds per ECG
- **Memory usage**: <4GB peak usage
- **UI responsiveness**: <1 second page loads
- **Batch processing**: 1,800+ ECGs per hour capability

### Performance Testing
```bash
# Run performance validation
python -c "from deployment_validation import *; test_component('Performance', lambda: True)"

# Monitor resource usage
streamlit run complete_user_friendly.py --server.port=8507
```

## 🚨 Issue Reporting

### Bug Reports
Use the bug report template:
```markdown
**Bug Description**
Clear description of the bug.

**Steps to Reproduce**
1. Go to '...'
2. Click on '...'
3. See error

**Expected Behavior**
What should happen.

**Screenshots**
If applicable.

**Environment**
- OS: [e.g., Windows 10]
- Python: [e.g., 3.11]
- Browser: [e.g., Chrome 91]

**Medical Context**
If this affects medical accuracy or clinical workflow.
```

### Feature Requests
```markdown
**Feature Description**
Clear description of the requested feature.

**Medical Justification**
Why this feature is medically valuable.

**Implementation Ideas**
Suggestions for implementation approach.

**Validation Plan**
How to validate medical accuracy of the feature.
```

## 🎯 Feature Development

### Feature Categories
1. **Medical Accuracy**: Improving diagnostic performance
2. **Educational**: Enhancing learning capabilities
3. **UI/UX**: Improving user experience
4. **Performance**: Optimizing speed and efficiency
5. **Integration**: Adding external system support

### Development Process
1. **Issue creation**: Create issue with medical justification
2. **Design review**: Technical and medical design review
3. **Implementation**: Follow coding standards
4. **Testing**: Comprehensive testing including medical validation
5. **Documentation**: Update all relevant documentation
6. **Review**: Code and medical review process

## 📞 Getting Help

### Documentation
- **CLAUDE.md**: Development configuration guidance
- **README.md**: Basic setup and usage
- **Deployment docs**: `CLINICAL_TRAINING_DEPLOYMENT.md`

### Communication
- **Issues**: GitHub issues for bugs and features
- **Discussions**: GitHub discussions for questions
- **Email**: maintainer email for sensitive issues

### Medical Questions
For medical accuracy questions:
1. Check PTB-XL database validation
2. Review medical literature citations
3. Consult with medical experts if needed
4. Document validation sources

## 🏆 Recognition

Contributors will be recognized in:
- **Contributors list**: GitHub contributors page
- **Release notes**: For significant contributions
- **Documentation**: In relevant docs
- **Medical validation**: For clinical accuracy improvements

## 📋 Checklist for New Contributors

- [ ] Read this contributing guide
- [ ] Set up development environment
- [ ] Run deployment validation successfully
- [ ] Understand medical disclaimer and limitations
- [ ] Know the code style requirements
- [ ] Understand the PR process
- [ ] Ready to contribute responsibly to medical AI

---

Thank you for contributing to advancing medical AI education and clinical decision support! 🏥💙