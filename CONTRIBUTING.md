# Contributing to MFSU Comet Analysis Framework

**Welcome to the astronomical community's first fractal analysis framework!**

We're excited that you want to contribute to this groundbreaking project. This guide will help you get started with contributing code, documentation, or scientific insights.

## 🎯 **Types of Contributions**

### **1. Code Contributions**
- 🐛 **Bug fixes** - Help improve stability and reliability
- ✨ **New features** - Extend functionality for broader applications
- 🔧 **Performance improvements** - Optimize analysis speed and memory usage
- 📊 **Additional analysis methods** - Implement new fractal or morphological techniques

### **2. Scientific Contributions**
- 📚 **Documentation improvements** - Better explanations of methodology
- 🔬 **Algorithm validation** - Test with known standards or synthetic data
- 🌌 **New applications** - Extend to asteroids, interstellar objects, etc.
- 📖 **Literature integration** - Connect with established astronomical methods

### **3. Community Contributions**
- 📝 **Tutorial creation** - Help new users get started
- 🎓 **Educational materials** - Workshops, presentations, courses
- 🐛 **Issue reporting** - Help identify problems and improvements
- 💬 **User support** - Answer questions in discussions

## 🚀 **Getting Started**

### **Development Setup**

1. **Fork and clone the repository**
```bash
# Fork on GitHub, then clone your fork
git clone https://github.com/YOUR-USERNAME/mfsu-comet-analysis.git
cd mfsu-comet-analysis

# Add upstream remote
git remote add upstream https://github.com/original-username/mfsu-comet-analysis.git
```

2. **Set up development environment**
```bash
# Create virtual environment
python -m venv mfsu-dev
source mfsu-dev/bin/activate  # Windows: mfsu-dev\Scripts\activate

# Install development dependencies
pip install -r requirements-dev.txt
pip install -e .  # Install in development mode

# Install pre-commit hooks
pre-commit install
```

3. **Run tests to verify setup**
```bash
pytest tests/
python -c "from mfsu_comet_analysis import run_complete_analysis; print('✅ Dev setup complete!')"
```

### **Development Workflow**

1. **Create a feature branch**
```bash
git checkout -b feature/your-feature-name
# or
git checkout -b bugfix/issue-description
```

2. **Make your changes**
- Write clean, well-documented code
- Follow existing code style and conventions
- Add tests for new functionality
- Update documentation as needed

3. **Test thoroughly**
```bash
# Run all tests
pytest tests/

# Run specific test category
pytest tests/test_fractal_analysis.py

# Check code style
flake8 src/
black --check src/
```

4. **Commit with clear messages**
```bash
git add .
git commit -m "Add: Brief description of what you added

More detailed explanation if needed.
- Specific change 1
- Specific change 2

Fixes #issue-number (if applicable)"
```

5. **Push and create pull request**
```bash
git push origin feature/your-feature-name
# Then create pull request on GitHub
```

## 📝 **Code Standards**

### **Python Style Guidelines**

We follow **PEP 8** with some specific conventions:

```python
# Good: Clear function names and docstrings
def calculate_fractal_dimension(scales, counts, method='linear'):
    """
    Calculate fractal dimension from box-counting data.
    
    Parameters
    ----------
    scales : numpy.ndarray
        Array of scales in physical units (arcsec)
    counts : numpy.ndarray  
        Number of occupied boxes at each scale
    method : str, default 'linear'
        Fitting method ('linear', 'robust', 'weighted')
        
    Returns
    -------
    df : float
        Fractal dimension
    df_error : float
        Statistical uncertainty
    r_squared : float
        Goodness of fit
        
    Notes
    -----
    Uses log-log linear regression: log(N) ~ -df * log(ε)
    """
    # Implementation here
    pass

# Good: Clear variable names
astronomical_pixel_scale = 0.080  # arcsec/pixel
detection_threshold_sigma = 3.0
box_counting_r_squared = 0.9982

# Bad: Unclear names
x = 0.080
thresh = 3.0
r2 = 0.9982
```

### **Scientific Code Standards**

```python
# Always include units in comments/docstrings
radius_arcsec = pixel_radius * pixel_scale  # Convert to arcseconds

# Use physical constants with clear names
JWST_PIXEL_SCALE = 0.080  # arcsec/pixel for IFU
DETECTION_THRESHOLD_SIGMA = 3.0  # Standard astronomical threshold

# Validate input parameters
if len(scales) < 5:
    raise ValueError(f"Insufficient scales for analysis: {len(scales)} < 5")

# Propagate uncertainties properly
df_error = np.sqrt(covariance_matrix[0, 0])

# Include quality metrics
fit_quality = {
    'r_squared': r_squared,
    'residual_std': np.std(residuals),
    'n_points': len(scales)
}
```

### **Documentation Standards**

- **All functions must have docstrings** following NumPy style
- **Include parameter types and units**
- **Provide usage examples for complex functions**
- **Reference scientific papers** where appropriate

## 🧪 **Testing Guidelines**

### **Test Categories**

1. **Unit tests** - Test individual functions
```python
def test_fractal_dimension_calculation():
    """Test fractal dimension calculation with known data."""
    # Synthetic power law data
    scales = np.array([1.0, 0.5, 0.25, 0.125])
    counts = np.array([1, 4, 16, 64])  # Perfect df = 2.0
    
    df, df_error, r_squared = calculate_fractal_dimension(scales, counts)
    
    assert abs(df - 2.0) < 0.01, f"Expected df=2.0, got {df}"
    assert r_squared > 0.999, f"Expected perfect fit, got R²={r_squared}"
```

2. **Integration tests** - Test complete workflows
```python
def test_complete_analysis_pipeline():
    """Test full analysis pipeline with synthetic comet."""
    analyzer = MFSUCometReal()
    
    # Create synthetic comet image
    synthetic_image = create_test_comet_image()
    
    # Run complete analysis
    processed_image, preprocessing_data = analyzer.preprocess_image(synthetic_image)
    box_data = analyzer.advanced_box_counting(processed_image, preprocessing_data)
    
    # Verify reasonable results
    df_measured = box_data[0]
    assert 1.0 < df_measured < 3.0, f"Unrealistic df: {df_measured}"
```

3. **Performance tests** - Ensure reasonable speed
```python
def test_analysis_performance():
    """Ensure analysis completes within reasonable time."""
    import time
    
    start_time = time.time()
    analyzer, image, results = run_complete_analysis()
    analysis_time = time.time() - start_time
    
    assert analysis_time < 120, f"Analysis too slow: {analysis_time:.1f}s > 120s"
```

### **Adding Tests**

When adding new functionality:

1. **Create test file** in `tests/` directory
2. **Name test functions** starting with `test_`
3. **Include both positive and negative test cases**
4. **Test edge cases** (empty data, extreme values, etc.)
5. **Document test purpose** clearly

## 📊 **Scientific Validation**

### **Reference Data**

For testing and validation, we maintain:

- **Synthetic test cases** with known fractal dimensions
- **Reference measurements** from established methods  
- **Benchmark datasets** for performance comparison

### **Adding New Analysis Methods**

When implementing new fractal or morphological analysis methods:

1. **Literature review** - Reference established papers
2. **Theoretical validation** - Ensure mathematical correctness
3. **Synthetic testing** - Validate with known test cases
4. **Comparison studies** - Compare with existing methods
5. **Documentation** - Explain when to use each method

Example template for new methods:
```python
def new_fractal_method(image, parameters):
    """
    New fractal analysis method based on [Reference].
    
    This method implements the technique described in:
    Author et al. (Year). "Title". Journal, Volume, Pages.
    
    Parameters
    ----------
    image : numpy.ndarray
        Preprocessed astronomical image
    parameters : dict
        Method-specific parameters
        
    Returns
    -------
    fractal_params : dict
        Dictionary containing:
        - 'dimension': Fractal dimension
        - 'error': Statistical uncertainty  
        - 'quality': Fit quality metrics
        
    Notes
    -----
    This method is particularly useful for [specific cases].
    Limitations: [known limitations]
    
    References
    ----------
    .. [1] Author et al. (Year). Journal reference.
    """
    # Implementation
    pass
```

## 🐛 **Reporting Issues**

### **Bug Reports**

When reporting bugs, include:

```markdown
**Bug Description**
Clear description of what's wrong

**To Reproduce**
1. Steps to reproduce the behavior
2. Include minimal code example
3. Specify input data characteristics

**Expected Behavior**
What should happen instead

**Environment**
- OS: [e.g., Windows 10, macOS 12, Ubuntu 20.04]
- Python version: [e.g., 3.9.7]
- Package versions: numpy, matplotlib, scipy versions
- Hardware: [RAM, CPU if relevant]

**Additional Context**
- Error messages (full traceback)
- Sample data (if possible to share)
- Screenshots (if relevant)
```

### **Feature Requests**

For new features:

```markdown
**Feature Description**
Clear description of the proposed feature

**Scientific Motivation**
Why is this feature needed? What scientific problem does it solve?

**Proposed Implementation**
Ideas for how it could be implemented

**References**
Relevant papers or methods

**Additional Context**
Examples, mockups, or related work
```

## 📚 **Documentation Contributions**

### **Types of Documentation**

- **API documentation** - Function and class documentation
- **User tutorials** - Step-by-step guides for common tasks
- **Theory explanations** - Mathematical background and interpretation
- **Best practices** - Guidelines for high-quality analysis

### **Documentation Style**

- **Clear and concise** - Avoid unnecessary jargon
- **Include examples** - Show practical usage
- **Scientific accuracy** - Verify all technical content
- **Accessibility** - Make content useful for different experience levels

## 🏆 **Recognition**

Contributors will be recognized through:

- **Contributor list** in README and documentation
- **Git commit history** preserving authorship
- **Acknowledgments** in scientific papers using the framework
- **Community recognition** in discussions and announcements

### **Significant Contributions**

Major contributors may be offered:
- **Co-authorship** on methodology papers
- **Collaboration opportunities** on related research
- **Conference presentation** opportunities
- **Mentorship** roles for new contributors

## 📋 **Pull Request Process**

### **Before Submitting**

- [ ] Code follows style guidelines
- [ ] Tests pass locally
- [ ] Documentation updated (if applicable)
- [ ] Commits have clear messages
- [ ] Branch is up-to-date with main

### **Pull Request Template**

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature  
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Scientific method addition

## Testing
- [ ] Added new tests
- [ ] All tests pass
- [ ] Manual testing completed

## Scientific Validation
- [ ] Method validated against known cases
- [ ] Literature references included
- [ ] Physical interpretation documented

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Changes ready for production use
```

### **Review Process**

1. **Automated checks** run (tests, style, etc.)
2. **Scientific review** for accuracy and methodology
3. **Code review** for quality and maintainability  
4. **Community feedback** if significant changes
5. **Final approval** and merge

## 🌟 **Community Guidelines**

### **Code of Conduct**

We are committed to providing a welcoming and inclusive environment:

- **Be respectful** in all interactions
- **Focus on scientific merit** rather than personal attributes
- **Provide constructive feedback** with specific suggestions
- **Help newcomers** learn and contribute effectively
- **Acknowledge others' contributions** appropriately

### **Communication Channels**

- **GitHub Issues** - Bug reports and feature requests
- **GitHub Discussions** - Questions, ideas, and general discussion
- **Email** - research@mfsu.org for direct contact
- **Scientific conferences** - Present work at relevant meetings

## 📈 **Roadmap & Priorities**

### **High Priority**
- Multi-wavelength support (optical, UV, radio)
- Performance optimization for large surveys
- Additional fractal analysis methods
- Automated quality assessment

### **Medium Priority**  
- Machine learning integration
- Interactive visualization tools
- Database integration
- Web-based interface

### **Future Vision**
- Real-time survey processing
- Multi-messenger astronomy integration
- Citizen science applications
- Educational platform development

---

**🌌 Thank you for contributing to the future of astronomical analysis!**

**Together, we're building tools that will revolutionize how we study small solar system bodies and beyond.**

**Questions? Contact: research@mfsu.org or open a GitHub Discussion**
