#!/usr/bin/env python3
"""
MFSU Comet Analysis - Package Setup Configuration
================================================

Professional Python package setup for the Unified Fractal-Stochastic Model (MFSU)
framework for astronomical object analysis.

Author: Miguel Ángel Franco León 
Date: September 2025
NASA Review: Production-ready package configuration

This setup.py provides:
- Complete package installation configuration
- Dependency management with version constraints
- Optional feature installations (FITS, Jupyter, development)
- NASA-level quality standards
- Cross-platform compatibility
- Professional metadata for scientific distribution
"""

import os
import sys
from pathlib import Path
from setuptools import setup, find_packages

# Ensure Python version compatibility
if sys.version_info < (3, 8):
    raise RuntimeError(
        "MFSU Comet Analysis requires Python 3.8 or higher. "
        f"Current Python version: {sys.version_info.major}.{sys.version_info.minor}"
    )

# Package directory
HERE = Path(__file__).parent.absolute()

# Read long description from README
def read_long_description():
    """Read long description from README.md with error handling."""
    readme_path = HERE / "README.md"
    try:
        with open(readme_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "MFSU Comet Analysis: Fractal analysis framework for astronomical objects"

# Read version from package
def get_version():
    """Extract version from package __init__.py"""
    version_file = HERE / "mfsu_comet_analysis" / "__init__.py"
    try:
        with open(version_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("__version__"):
                    # Extract version string: __version__ = "1.0.0"
                    return line.split("=")[1].strip().strip('"').strip("'")
    except FileNotFoundError:
        pass
    return "1.0.0"  # Fallback version

# Core dependencies with scientific computing focus
CORE_REQUIREMENTS = [
    # Numerical computing (essential)
    "numpy>=1.21.0,<2.0.0",              # Stable NumPy for numerical computations
    
    # Scientific computing
    "scipy>=1.7.0,<2.0.0",               # Statistical analysis and optimization
    
    # Visualization (essential for scientific plots)
    "matplotlib>=3.5.0,<4.0.0",          # Publication-quality plotting
    
    # Image processing
    "Pillow>=8.3.0,<11.0.0",             # Image loading and basic processing
    
    # Data handling
    "pandas>=1.3.0,<3.0.0",              # Data structures for analysis results
]

# Optional dependencies for enhanced functionality
EXTRAS_REQUIRE = {
    # FITS file support (recommended for astronomical data)
    "fits": [
        "astropy>=5.0.0,<7.0.0",          # Astronomical data formats and tools
    ],
    
    # Jupyter notebook support
    "jupyter": [
        "jupyter>=1.0.0,<2.0.0",          # Jupyter notebook environment
        "ipykernel>=6.0.0,<8.0.0",        # IPython kernel for notebooks
        "ipywidgets>=7.6.0,<9.0.0",       # Interactive widgets
    ],
    
    # Development tools
    "dev": [
        "pytest>=6.0.0,<8.0.0",           # Testing framework
        "pytest-cov>=2.12.0,<5.0.0",      # Coverage reporting
        "black>=22.0.0,<24.0.0",          # Code formatting
        "flake8>=4.0.0,<7.0.0",           # Linting
        "mypy>=0.910,<2.0.0",             # Type checking
        "isort>=5.9.0,<6.0.0",            # Import sorting
        "pre-commit>=2.15.0,<4.0.0",      # Git hooks
    ],
    
    # Enhanced plotting and visualization
    "plotting": [
        "seaborn>=0.11.0,<1.0.0",         # Statistical plotting
        "plotly>=5.0.0,<6.0.0",           # Interactive plots
    ],
    
    # Performance enhancements
    "performance": [
        "numba>=0.56.0,<1.0.0",           # JIT compilation for speed
        "joblib>=1.1.0,<2.0.0",           # Parallel processing
    ],
    
    # Additional scientific tools
    "science": [
        "scikit-image>=0.18.0,<1.0.0",    # Advanced image processing
        "scikit-learn>=1.0.0,<2.0.0",     # Machine learning tools
    ],
}

# All optional dependencies combined
EXTRAS_REQUIRE["all"] = [
    dep for deps in EXTRAS_REQUIRE.values() for dep in deps
]

# Complete package configuration
setup(
    # Basic package information
    name="mfsu-comet-analysis",
    version=get_version(),
    
    # Author and contact information
    author="Miguel Ángel Franco León",
    author_email="research@mfsu.org",
    maintainer="MFSU Research Group",
    maintainer_email="research@mfsu.org",
    
    # Package description
    description=(
        "MFSU framework for rigorous fractal analysis of astronomical objects "
        "using the Unified Fractal-Stochastic Model"
    ),
    long_description=read_long_description(),
    long_description_content_type="text/markdown",
    
    # URLs and links
    url="https://github.com/miguelfrancoleon/MFSU-CometAnalysis",
    download_url="https://github.com/miguelfrancoleon/MFSU-CometAnalysis/archive/v1.0.0.tar.gz",
    project_urls={
        "Documentation": "https://github.com/miguelfrancoleon/MFSU-CometAnalysis/tree/main/docs",
        "Source Code": "https://github.com/miguelfrancoleon/MFSU-CometAnalysis",
        "Bug Tracker": "https://github.com/miguelfrancoleon/MFSU-CometAnalysis/issues",
        "NASA Review": "https://github.com/miguelfrancoleon/MFSU-CometAnalysis/tree/main/docs",
        "Research Paper": "https://github.com/miguelfrancoleon/MFSU-CometAnalysis/tree/main/docs",
    },
    
    # Package structure
    packages=find_packages(
        include=["mfsu_comet_analysis", "mfsu_comet_analysis.*"]
    ),
    package_dir={"": "."},
    
    # Include additional files
    include_package_data=True,
    package_data={
        "mfsu_comet_analysis": [
            "data/*.json",
            "data/*.txt",
            "configs/*.yaml",
            "configs/*.json",
        ],
    },
    
    # Entry points for command-line interface (optional)
    entry_points={
        "console_scripts": [
            "mfsu-analyze=mfsu_comet_analysis.cli:main",
            "mfsu-validate=mfsu_comet_analysis.cli:validate",
        ],
    },
    
    # Dependencies
    install_requires=CORE_REQUIREMENTS,
    extras_require=EXTRAS_REQUIRE,
    
    # Python version requirements
    python_requires=">=3.8,<4.0",
    
    # Package classification for PyPI
    classifiers=[
        # Development status
        "Development Status :: 5 - Production/Stable",
        
        # Intended audience
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "Intended Audience :: Developers",
        
        # License
        "License :: OSI Approved :: MIT License",
        
        # Programming language
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3 :: Only",
        
        # Operating systems
        "Operating System :: OS Independent",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS",
        
        # Topics
        "Topic :: Scientific/Engineering :: Astronomy",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Visualization",
        "Topic :: Software Development :: Libraries :: Python Modules",
        
        # Natural language
        "Natural Language :: English",
        
        # Environment
        "Environment :: Console",
        "Environment :: Web Environment",
    ],
    
    # Keywords for search
    keywords=[
        "astronomy", "astrophysics", "fractals", "comet analysis", 
        "MFSU", "fractal dimension", "space objects", "JWST",
        "image analysis", "scientific computing", "NASA",
        "astronomical data analysis", "fractal geometry"
    ],
    
    # License
    license="MIT",
    
    # Zip safety
    zip_safe=False,
    
    # Additional metadata for scientific packages
    platforms=["any"],
    
    # Test suite
    test_suite="tests",
    tests_require=[
        "pytest>=6.0.0",
        "pytest-cov>=2.12.0",
    ],
    
    # Setuptools specific options
    options={
        "build_py": {
            "compile": True,
            "optimize": 2,
        },
        "bdist_wheel": {
            "universal": False,  # Not universal due to Python 3.8+ requirement
        },
    },
)

# Post-installation message for NASA review team
def print_installation_message():
    """Print helpful installation message."""
    print("\n" + "="*70)
    print("🌌 MFSU COMET ANALYSIS - INSTALLATION COMPLETE")
    print("="*70)
    print("✅ Core package installed successfully")
    print()
    print("📦 OPTIONAL INSTALLATIONS:")
    print("   pip install mfsu-comet-analysis[fits]       # FITS file support")
    print("   pip install mfsu-comet-analysis[jupyter]    # Jupyter notebooks")
    print("   pip install mfsu-comet-analysis[dev]        # Development tools")
    print("   pip install mfsu-comet-analysis[all]        # All features")
    print()
    print("🚀 QUICK START:")
    print("   python -c \"from mfsu_comet_analysis import run_complete_analysis; run_complete_analysis()\"")
    print()
    print("📚 DOCUMENTATION:")
    print("   docs/USER_MANUAL.md      - Complete usage guide")
    print("   docs/API_REFERENCE.md    - Technical documentation")
    print("   docs/INSTALLATION.md     - Detailed installation guide")
    print()
    print("🔬 NASA REVIEW:")
    print("   docs/development_timeline.md  - Complete development history")
    print("   examples/                      - Working code examples")
    print("   examples/colab_standalone.py   - Original research code")
    print()
    print("Ready for astronomical fractal analysis! 🌟")
    print("="*70)

# Run post-installation message if installing
if __name__ == "__main__" and "install" in sys.argv:
    import atexit
    atexit.register(print_installation_message)
