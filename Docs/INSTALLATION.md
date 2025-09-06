# MFSU Comet Analysis - Installation Guide

**Version:** 1.0.0  
**Author:** Miguel Ángel Franco León   
**Date:** September 2025

Complete installation guide for the MFSU (Unified Fractal-Stochastic Model) Comet Analysis package.

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Quick Installation](#quick-installation)
3. [Detailed Installation Steps](#detailed-installation-steps)
4. [Virtual Environment Setup](#virtual-environment-setup)
5. [Development Installation](#development-installation)
6. [Optional Dependencies](#optional-dependencies)
7. [Verification](#verification)
8. [Troubleshooting](#troubleshooting)
9. [Platform-Specific Notes](#platform-specific-notes)
10. [Updating](#updating)

---

## System Requirements

### Operating Systems
✅ **Windows** (10/11, 64-bit)  
✅ **macOS** (10.14+ or 11+, Intel/Apple Silicon)  
✅ **Linux** (Ubuntu 18.04+, CentOS 7+, or equivalent)

### Python Requirements
- **Python 3.8+** (recommended: Python 3.9-3.11)
- **64-bit Python installation** (required for large image processing)

### Hardware Requirements

| **Component** | **Minimum** | **Recommended** | **Optimal** |
|---------------|-------------|-----------------|-------------|
| **RAM** | 4 GB | 8 GB | 16+ GB |
| **CPU** | Dual-core | Quad-core | 8+ cores |
| **Storage** | 1 GB free | 5 GB free | 10+ GB free |
| **GPU** | Not required | Not required | Optional for future features |

### Dependencies Overview

**Core dependencies** (automatically installed):
- `numpy >= 1.21.0` - Numerical computations
- `matplotlib >= 3.5.0` - Scientific plotting  
- `scipy >= 1.7.0` - Statistical analysis
- `Pillow >= 8.3.0` - Image processing

**Optional dependencies**:
- `astropy >= 5.0` - FITS file support (recommended for astronomy)
- `jupyter >= 1.0.0` - Notebook support
- `pytest >= 6.0.0` - Testing framework

---

## Quick Installation

### Option 1: Standard Installation (Recommended)

```bash
# Install from source (when package is on PyPI)
pip install mfsu-comet-analysis

# Or install directly from GitHub
pip install git+https://github.com/yourusername/MFSU-CometAnalysis.git
```

### Option 2: Local Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/MFSU-CometAnalysis.git
cd MFSU-CometAnalysis

# Install in development mode
pip install -e .
```

### Option 3: Conda Installation (if available)

```bash
# When available on conda-forge
conda install -c conda-forge mfsu-comet-analysis
```

---

## Detailed Installation Steps

### Step 1: Check Python Installation

First, verify you have Python 3.8+ installed:

```bash
# Check Python version
python --version
# or
python3 --version

# Should output: Python 3.8.x or higher
```

If Python is not installed or version is too old:

**Windows:**
- Download from [python.org](https://www.python.org/downloads/)
- Choose "Add Python to PATH" during installation

**macOS:**
```bash
# Using Homebrew (recommended)
brew install python

# Or download from python.org
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv
```

**Linux (CentOS/RHEL):**
```bash
sudo yum install python3 python3-pip
# or on newer versions:
sudo dnf install python3 python3-pip
```

### Step 2: Update pip

Ensure you have the latest pip version:

```bash
python -m pip install --upgrade pip
```

### Step 3: Install MFSU Comet Analysis

#### Method A: From Source (Current Method)

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/MFSU-CometAnalysis.git

# 2. Navigate to the directory
cd MFSU-CometAnalysis

# 3. Install the package
pip install -e .

# 4. Verify installation
python -c "import mfsu_comet_analysis; print('Installation successful!')"
```

#### Method B: Direct GitHub Install

```bash
# Install directly from GitHub
pip install git+https://github.com/yourusername/MFSU-CometAnalysis.git

# For a specific branch or tag
pip install git+https://github.com/yourusername/MFSU-CometAnalysis.git@main
```

#### Method C: From Downloaded ZIP

```bash
# 1. Download ZIP from GitHub
# 2. Extract to a folder
# 3. Navigate to the folder
cd MFSU-CometAnalysis-main

# 4. Install
pip install .
```

---

## Virtual Environment Setup

**Highly recommended** to avoid conflicts with other packages:

### Using venv (Built-in)

```bash
# Create virtual environment
python -m venv mfsu_env

# Activate virtual environment
# Windows:
mfsu_env\Scripts\activate
# macOS/Linux:
source mfsu_env/bin/activate

# Install MFSU
pip install git+https://github.com/yourusername/MFSU-CometAnalysis.git

# Deactivate when done
deactivate
```

### Using conda

```bash
# Create conda environment
conda create -n mfsu_env python=3.10

# Activate environment
conda activate mfsu_env

# Install dependencies
conda install numpy matplotlib scipy pillow

# Install MFSU
pip install git+https://github.com/yourusername/MFSU-CometAnalysis.git

# Deactivate when done
conda deactivate
```

### Using virtualenv

```bash
# Install virtualenv if not available
pip install virtualenv

# Create virtual environment
virtualenv mfsu_env

# Activate (similar to venv)
# Windows:
mfsu_env\Scripts\activate
# macOS/Linux:
source mfsu_env/bin/activate

# Install MFSU
pip install git+https://github.com/yourusername/MFSU-CometAnalysis.git
```

---

## Development Installation

For developers who want to contribute or modify the code:

### Step 1: Clone and Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/MFSU-CometAnalysis.git
cd MFSU-CometAnalysis

# Create development environment
python -m venv dev_env
source dev_env/bin/activate  # Linux/macOS
# or
dev_env\Scripts\activate  # Windows
```

### Step 2: Install in Development Mode

```bash
# Install in editable mode with development dependencies
pip install -e ".[dev]"

# Or install dependencies manually
pip install -e .
pip install pytest pytest-cov black flake8 mypy jupyter
```

### Step 3: Verify Development Setup

```bash
# Run tests
pytest tests/

# Check code style
black --check mfsu_comet_analysis/
flake8 mfsu_comet_analysis/

# Type checking
mypy mfsu_comet_analysis/
```

---

## Optional Dependencies

### FITS File Support (Recommended for Astronomy)

```bash
# Install astropy for FITS file support
pip install astropy

# Verify FITS support
python -c "from mfsu_comet_analysis.preprocessing import DataLoader; print('FITS support available')"
```

### Jupyter Notebook Support

```bash
# Install Jupyter
pip install jupyter

# Or use JupyterLab
pip install jupyterlab

# Start Jupyter
jupyter notebook
# or
jupyter lab
```

### Enhanced Plotting

```bash
# For additional plotting features
pip install seaborn plotly

# For LaTeX rendering in plots (optional)
# Linux:
sudo apt install texlive-latex-base texlive-fonts-recommended
# macOS:
brew install --cask mactex
# Windows: Install MiKTeX from miktex.org
```

### Performance Enhancements

```bash
# For faster numerical computations
pip install numba

# For parallel processing
pip install joblib

# For memory-efficient image processing
pip install scikit-image
```

---

## Verification

### Basic Verification

```bash
# Test basic import
python -c "import mfsu_comet_analysis; print('✅ MFSU package imported successfully')"

# Test specific modules
python -c "from mfsu_comet_analysis.core import MFSUCometReal; print('✅ Core module working')"
python -c "from mfsu_comet_analysis.fractal_analysis import FractalAnalyzer; print('✅ Fractal analysis module working')"
python -c "from mfsu_comet_analysis.preprocessing import AstronomicalPreprocessor; print('✅ Preprocessing module working')"
python -c "from mfsu_comet_analysis.visualization import MFSUVisualizer; print('✅ Visualization module working')"
python -c "from mfsu_comet_analysis.utils import MFSU_CONSTANTS; print('✅ Utilities module working')"
```

### Complete Verification

```python
# Save this as test_installation.py and run: python test_installation.py

#!/usr/bin/env python3
"""
MFSU Installation Verification Script
Run this script to verify your MFSU installation is working correctly.
"""

def test_installation():
    """Comprehensive installation test."""
    print("🔧 MFSU COMET ANALYSIS - INSTALLATION VERIFICATION")
    print("=" * 60)
    
    # Test 1: Basic imports
    print("\n1️⃣ Testing basic imports...")
    try:
        import mfsu_comet_analysis
        print("   ✅ Main package imported")
        
        from mfsu_comet_analysis.core import MFSUCometReal
        from mfsu_comet_analysis.fractal_analysis import FractalAnalyzer
        from mfsu_comet_analysis.preprocessing import AstronomicalPreprocessor
        from mfsu_comet_analysis.visualization import MFSUVisualizer
        from mfsu_comet_analysis.utils import MFSU_CONSTANTS
        print("   ✅ All core modules imported successfully")
    except ImportError as e:
        print(f"   ❌ Import error: {e}")
        return False
    
    # Test 2: Core dependencies
    print("\n2️⃣ Testing core dependencies...")
    try:
        import numpy as np
        import matplotlib.pyplot as plt
        import scipy
        from PIL import Image
        print("   ✅ All core dependencies available")
        print(f"   📦 NumPy version: {np.__version__}")
        print(f"   📦 Matplotlib version: {plt.matplotlib.__version__}")
        print(f"   📦 SciPy version: {scipy.__version__}")
    except ImportError as e:
        print(f"   ❌ Dependency error: {e}")
        return False
    
    # Test 3: Optional dependencies
    print("\n3️⃣ Testing optional dependencies...")
    try:
        import astropy
        print(f"   ✅ Astropy available (version {astropy.__version__})")
        fits_support = True
    except ImportError:
        print("   ⚠️  Astropy not available (FITS support limited)")
        fits_support = False
    
    try:
        import jupyter
        print("   ✅ Jupyter available")
    except ImportError:
        print("   ⚠️  Jupyter not available")
    
    # Test 4: Basic functionality
    print("\n4️⃣ Testing basic functionality...")
    try:
        # Initialize components
        analyzer = MFSUCometReal()
        preprocessor = AstronomicalPreprocessor()
        fractal_analyzer = FractalAnalyzer()
        visualizer = MFSUVisualizer()
        
        print("   ✅ All classes can be instantiated")
        
        # Test constants
        print(f"   ✅ MFSU constants loaded (df = {MFSU_CONSTANTS.DF_THEORETICAL})")
        
    except Exception as e:
        print(f"   ❌ Functionality error: {e}")
        return False
    
    # Test 5: Simple analysis (with synthetic data)
    print("\n5️⃣ Testing analysis pipeline...")
    try:
        import numpy as np
        
        # Create synthetic test image
        size = 128
        x = np.linspace(-2, 2, size)
        y = np.linspace(-2, 2, size)
        X, Y = np.meshgrid(x, y)
        R = np.sqrt(X**2 + Y**2)
        
        # Simple synthetic comet-like structure
        test_image = 100 * np.exp(-R**2/0.5) + 50 * np.exp(-R/1.0) + np.random.normal(0, 2, (size, size))
        test_image = np.maximum(test_image, 0)
        
        # Basic preprocessing
        processed_image, prep_data = preprocessor.preprocess_image(test_image)
        print("   ✅ Preprocessing works")
        
        # Basic fractal analysis (minimal parameters for speed)
        box_data = fractal_analyzer.advanced_box_counting(
            processed_image, prep_data['detection_threshold'], 
            n_scales=4, verbose=False
        )
        print(f"   ✅ Box-counting analysis works (df = {box_data[0]:.3f})")
        
        radial_data = fractal_analyzer.advanced_radial_analysis(
            processed_image, prep_data['detection_threshold'],
            n_bins=10, verbose=False
        )
        print(f"   ✅ Radial analysis works (α = {radial_data[0]:.3f})")
        
    except Exception as e:
        print(f"   ❌ Analysis error: {e}")
        return False
    
    # Test 6: Visualization
    print("\n6️⃣ Testing visualization...")
    try:
        # Test plot creation (but don't show)
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        
        fig = visualizer.create_focused_box_counting_plot(
            box_data[2], box_data[3], box_data[0], box_data[1], box_data[4]
        )
        print("   ✅ Visualization works")
        
        # Clean up
        plt.close(fig)
        
    except Exception as e:
        print(f"   ❌ Visualization error: {e}")
        return False
    
    # Summary
    print("\n" + "=" * 60)
    print("🎉 INSTALLATION VERIFICATION COMPLETE")
    print("✅ All core functionality working correctly")
    
    if fits_support:
        print("✅ FITS support available")
    else:
        print("ℹ️  Install astropy for FITS support: pip install astropy")
    
    print("\n🚀 Ready to analyze astronomical objects with MFSU!")
    print("📚 See USER_MANUAL.md for usage examples")
    
    return True

if __name__ == "__main__":
    success = test_installation()
    exit(0 if success else 1)
```

Run the verification:

```bash
python test_installation.py
```

---

## Troubleshooting

### Common Installation Issues

#### Issue 1: `pip` command not found

**Problem:** `pip` is not recognized as a command

**Solutions:**
```bash
# Try python -m pip instead
python -m pip install mfsu-comet-analysis

# Or on some systems:
python3 -m pip install mfsu-comet-analysis

# Windows: Add Python to PATH
# Go to: Settings > Apps > Python > Advanced Options > Add to PATH
```

#### Issue 2: Permission errors

**Problem:** Permission denied when installing packages

**Solutions:**
```bash
# Option 1: Install for current user only
pip install --user mfsu-comet-analysis

# Option 2: Use virtual environment (recommended)
python -m venv mfsu_env
source mfsu_env/bin/activate  # or mfsu_env\Scripts\activate on Windows
pip install mfsu-comet-analysis

# Option 3: Use sudo (Linux/macOS, not recommended)
sudo pip install mfsu-comet-analysis
```

#### Issue 3: Dependency conflicts

**Problem:** Conflicting package versions

**Solutions:**
```bash
# Option 1: Use virtual environment (best solution)
python -m venv clean_env
source clean_env/bin/activate
pip install mfsu-comet-analysis

# Option 2: Force reinstall
pip install --force-reinstall mfsu-comet-analysis

# Option 3: Update conflicting packages
pip install --upgrade numpy matplotlib scipy
```

#### Issue 4: NumPy/SciPy installation fails

**Problem:** Compilation errors for NumPy/SciPy

**Solutions:**

**Windows:**
```bash
# Install pre-compiled wheels
pip install --only-binary=all numpy scipy matplotlib

# Or use conda
conda install numpy scipy matplotlib
```

**Linux:**
```bash
# Install system dependencies first
sudo apt install python3-dev build-essential gfortran libopenblas-dev liblapack-dev

# Then install packages
pip install numpy scipy matplotlib
```

**macOS:**
```bash
# Install Xcode command line tools
xcode-select --install

# Install packages
pip install numpy scipy matplotlib
```

#### Issue 5: MFSU import fails

**Problem:** `ModuleNotFoundError: No module named 'mfsu_comet_analysis'`

**Solutions:**
```bash
# Check if package is installed
pip list | grep mfsu

# If not installed, install it
pip install -e .  # if in source directory
# or
pip install git+https://github.com/yourusername/MFSU-CometAnalysis.git

# Check Python path
python -c "import sys; print('\n'.join(sys.path))"

# Reinstall in development mode
pip uninstall mfsu-comet-analysis
pip install -e .
```

#### Issue 6: Matplotlib backend issues

**Problem:** Plots don't display or crash

**Solutions:**
```python
# Set backend explicitly
import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg', 'Agg'
import matplotlib.pyplot as plt

# For headless systems
matplotlib.use('Agg')

# Check available backends
import matplotlib
print(matplotlib.rcsetup.all_backends)
```

### Platform-Specific Issues

#### Windows-Specific

**Issue:** Long path names causing problems
```bash
# Enable long paths in Windows 10/11
# Run as administrator in PowerShell:
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```

**Issue:** Antivirus blocking installation
- Add Python installation directory to antivirus exceptions
- Temporarily disable real-time protection during installation

#### macOS-Specific

**Issue:** SSL certificate errors
```bash
# Update certificates
/Applications/Python\ 3.x/Install\ Certificates.command

# Or install certificates manually
pip install --upgrade certifi
```

**Issue:** Homebrew Python vs. System Python conflicts
```bash
# Use Homebrew Python explicitly
/usr/local/bin/python3 -m pip install mfsu-comet-analysis

# Or create alias
echo 'alias python=/usr/local/bin/python3' >> ~/.zshrc
```

#### Linux-Specific

**Issue:** Missing system dependencies
```bash
# Ubuntu/Debian
sudo apt install python3-dev python3-pip python3-tk libfreetype6-dev pkg-config

# CentOS/RHEL
sudo yum install python3-devel python3-pip python3-tkinter freetype-devel pkgconfig
# or on newer versions:
sudo dnf install python3-devel python3-pip python3-tkinter freetype-devel pkgconfig
```

---

## Platform-Specific Notes

### Windows

**Recommended setup:**
1. Install Python from [python.org](https://www.python.org/downloads/)
2. Use Windows Terminal or PowerShell
3. Consider using Anaconda for easier package management

**Path issues:**
- Ensure Python is added to PATH during installation
- Use `py` launcher: `py -m pip install mfsu-comet-analysis`

### macOS

**Recommended setup:**
1. Install Homebrew: `/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"`
2. Install Python: `brew install python`
3. Use Terminal or iTerm2

**Apple Silicon (M1/M2) notes:**
- Most packages now support ARM64 natively
- If issues occur, try using Rosetta: `arch -x86_64 pip install mfsu-comet-analysis`

### Linux

**Ubuntu/Debian:**
```bash
# Update package list
sudo apt update

# Install Python and dependencies
sudo apt install python3 python3-pip python3-venv python3-dev

# Install MFSU
pip3 install mfsu-comet-analysis
```

**CentOS/RHEL:**
```bash
# Enable EPEL repository
sudo yum install epel-release

# Install Python
sudo yum install python3 python3-pip python3-devel

# Install MFSU
pip3 install mfsu-comet-analysis
```

---

## Updating

### Update to Latest Version

```bash
# Update from PyPI (when available)
pip install --upgrade mfsu-comet-analysis

# Update from GitHub
pip install --upgrade git+https://github.com/yourusername/MFSU-CometAnalysis.git

# Force reinstall if needed
pip install --force-reinstall mfsu-comet-analysis
```

### Update Dependencies

```bash
# Update all dependencies
pip install --upgrade numpy matplotlib scipy pillow

# Update optional dependencies
pip install --upgrade astropy jupyter

# Check for outdated packages
pip list --outdated
```

### Development Version Updates

```bash
# If installed in development mode, just pull updates
cd MFSU-CometAnalysis
git pull origin main

# Dependencies might need updating
pip install --upgrade -e ".[dev]"
```

---

## Uninstallation

### Remove MFSU Package

```bash
# Uninstall MFSU
pip uninstall mfsu-comet-analysis

# Remove virtual environment (if used)
rm -rf mfsu_env  # Linux/macOS
rmdir /s mfsu_env  # Windows

# Remove downloaded source (if applicable)
rm -rf MFSU-CometAnalysis
```

### Clean Uninstall

```bash
# Remove all MFSU-related packages
pip uninstall mfsu-comet-analysis numpy matplotlib scipy pillow astropy

# Clear pip cache
pip cache purge

# Remove virtual environment
deactivate  # if currently active
rm -rf mfsu_env
```

---

## Getting Help

### If Installation Fails

1. **Check the error message carefully** - it often contains the solution
2. **Try the troubleshooting section** above for your specific error
3. **Use a virtual environment** - this solves 90% of installation issues
4. **Update your system** - ensure you have the latest Python and pip
5. **Check platform-specific notes** for your operating system

### Resources

- **GitHub Issues**: Report installation problems
- **Documentation**: Complete user manual and API reference
- **Community**: Stack Overflow with tag `mfsu-comet-analysis`

### System Information Script

If you need help, run this script and include the output:

```python
#!/usr/bin/env python3
"""System information for debugging installation issues."""

import sys
import platform
import subprocess

def get_system_info():
    print("MFSU INSTALLATION DEBUG INFO")
    print("=" * 50)
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print(f"Platform: {platform.platform()}")
    print(f"Architecture: {platform.architecture()}")
    print(f"Machine: {platform.machine()}")
    
    # Check pip version
    try:
        result = subprocess.run([sys.executable, '-m', 'pip', '--version'], 
                              capture_output=True, text=True)
        print(f"Pip version: {result.stdout.strip()}")
    except:
        print("Pip version: Not available")
    
    # Check installed packages
    try:
        result = subprocess.run([sys.executable, '-m', 'pip', 'list'], 
                              capture_output=True, text=True)
        print("\nInstalled packages:")
        print(result.stdout)
    except:
        print("Cannot list installed packages")

if __name__ == "__main__":
    get_system_info()
```

---

## Next Steps

After successful installation:

1. **Read the User Manual** (`USER_MANUAL.md`) for usage examples
2. **Check the API Reference** (`API_REFERENCE.md`) for detailed documentation  
3. **Try the examples** in the `/examples/` directory
4. **Run the verification script** to ensure everything works
5. **Start analyzing your astronomical data!** 🌌

---

*Installation complete! Ready to explore the fractal universe! 🚀*
