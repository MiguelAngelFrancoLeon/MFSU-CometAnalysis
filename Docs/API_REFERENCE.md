# MFSU Comet Analysis - API Reference

**Version:** 1.0.0  
**Author:** Miguel Ángel Franco León 
**Date:** September 2025

This document provides comprehensive API documentation for the MFSU Comet Analysis package, including all classes, methods, functions, and their parameters.

## Table of Contents

1. [Core Module (`core.py`)](#core-module)
2. [Fractal Analysis Module (`fractal_analysis.py`)](#fractal-analysis-module)
3. [Preprocessing Module (`preprocessing.py`)](#preprocessing-module)
4. [Visualization Module (`visualization.py`)](#visualization-module)
5. [Utilities Module (`utils.py`)](#utilities-module)
6. [Examples](#examples)
7. [Error Handling](#error-handling)

---

## Core Module

### Class: `MFSUCometReal`

Main analysis class for MFSU fractal analysis of astronomical objects.

#### Constructor

```python
MFSUCometReal()
```

**Description:** Initialize the MFSU comet analyzer with theoretical parameters.

**Attributes:**
- `df_theoretical` (float): MFSU theoretical fractal dimension (2.079)
- `delta_theoretical` (float): MFSU theoretical correlation parameter (0.921)
- `arcsec_per_pixel` (float): Pixel scale in arcsec/pixel
- `wavelength_band` (str): Observation wavelength band
- `observation_date` (str): Date of observation

#### Methods

##### `load_jwst_image()`

```python
load_jwst_image(image_path=None, image_data=None) -> np.ndarray
```

**Description:** Load and preprocess JWST image data.

**Parameters:**
- `image_path` (str, optional): Path to image file
- `image_data` (np.ndarray, optional): Direct numpy array input

**Returns:**
- `np.ndarray`: 2D image array (float64)

**Example:**
```python
analyzer = MFSUCometReal()
comet_image = analyzer.load_jwst_image("comet_data.fits")
```

##### `run_complete_analysis()`

```python
run_complete_analysis() -> Tuple[MFSUCometReal, np.ndarray, Dict[str, Any]]
```

**Description:** Execute complete MFSU analysis pipeline.

**Returns:**
- `analyzer` (MFSUCometReal): Configured analyzer instance
- `image` (np.ndarray): Loaded comet image
- `results` (dict): Complete analysis results

**Example:**
```python
analyzer, image, results = run_complete_analysis()
print(f"Fractal dimension: {results['df_measured']:.3f}")
```

---

## Fractal Analysis Module

### Class: `FractalAnalyzer`

Advanced fractal analysis algorithms for astronomical objects.

#### Constructor

```python
FractalAnalyzer(pixel_scale=0.12)
```

**Parameters:**
- `pixel_scale` (float): Pixel scale in arcsec/pixel (default: 0.12 for JWST IFU)

**Attributes:**
- `pixel_scale` (float): Pixel scale in arcsec/pixel
- `df_theoretical` (float): MFSU theoretical fractal dimension
- `delta_theoretical` (float): MFSU theoretical correlation parameter

#### Methods

##### `advanced_box_counting()`

```python
advanced_box_counting(image, detection_threshold, n_scales=8, verbose=True) -> Tuple[float, float, np.ndarray, np.ndarray, float, np.ndarray]
```

**Description:** Perform advanced box-counting analysis with astronomical considerations.

**Parameters:**
- `image` (np.ndarray): 2D processed image (background-subtracted)
- `detection_threshold` (float): 3-sigma detection threshold for binary mask
- `n_scales` (int): Maximum number of scales to analyze (default: 8)
- `verbose` (bool): Print progress information (default: True)

**Returns:**
- `df_measured` (float): Measured fractal dimension
- `df_error` (float): Uncertainty in fractal dimension
- `scales` (np.ndarray): Physical scales used (arcsec)
- `counts` (np.ndarray): Occupied box counts at each scale
- `r_squared` (float): Quality of linear fit (R²)
- `box_sizes` (np.ndarray): Box sizes in pixels

**Example:**
```python
analyzer = FractalAnalyzer(pixel_scale=0.12)
df, df_err, scales, counts, r2, boxes = analyzer.advanced_box_counting(
    processed_image, threshold=3.5, n_scales=10
)
print(f"Fractal dimension: {df:.3f} ± {df_err:.3f} (R² = {r2:.4f})")
```

##### `advanced_radial_analysis()`

```python
advanced_radial_analysis(image, detection_threshold, center=None, n_bins=20, verbose=True) -> Tuple[float, float, np.ndarray, np.ndarray, float, np.ndarray]
```

**Description:** Perform advanced radial profile analysis with astronomical techniques.

**Parameters:**
- `image` (np.ndarray): 2D processed image
- `detection_threshold` (float): Detection threshold for centroid calculation
- `center` (tuple, optional): Manual center coordinates (x, y). If None, calculates photometric center
- `n_bins` (int): Number of radial bins (default: 20)
- `verbose` (bool): Print progress information (default: True)

**Returns:**
- `alpha` (float): Radial slope (I(r) ~ r^(-alpha))
- `alpha_error` (float): Uncertainty in radial slope
- `radii` (np.ndarray): Radial coordinates (arcsec)
- `intensities` (np.ndarray): Mean intensities at each radius
- `r_squared` (float): Quality of power-law fit
- `intensity_errors` (np.ndarray): Uncertainties in mean intensities

**Example:**
```python
alpha, alpha_err, radii, intensities, r2, errors = analyzer.advanced_radial_analysis(
    processed_image, threshold=3.0, n_bins=25
)
print(f"Radial slope: α = {alpha:.3f} ± {alpha_err:.3f}")
```

##### `scientific_interpretation()`

```python
scientific_interpretation(df_measured, df_error, alpha, alpha_error, verbose=True) -> Dict[str, Any]
```

**Description:** Comprehensive scientific interpretation of fractal analysis results.

**Parameters:**
- `df_measured` (float): Measured fractal dimension
- `df_error` (float): Uncertainty in fractal dimension
- `alpha` (float): Radial profile exponent
- `alpha_error` (float): Uncertainty in radial exponent
- `verbose` (bool): Print detailed interpretation (default: True)

**Returns:**
- `results` (dict): Comprehensive analysis results and interpretation

**Dictionary Keys:**
- `df_measured`, `df_error`: Measured fractal dimension and uncertainty
- `alpha`, `alpha_error`: Radial slope and uncertainty
- `delta_derived`: Derived δp parameter (3 - df)
- `df_sigma`: Statistical significance of MFSU agreement
- `mfsu_agreement`: Agreement assessment with MFSU
- `df_class`, `alpha_class`: Classification of measured parameters

### Class: `MFSUComparator`

Specialized class for comparing results with MFSU theoretical framework.

#### Constructor

```python
MFSUComparator()
```

#### Methods

##### `validate_parameters()`

```python
validate_parameters(df_measured, df_error) -> Dict[str, Any]
```

**Description:** Validate measured parameters against MFSU predictions.

**Parameters:**
- `df_measured` (float): Measured fractal dimension
- `df_error` (float): Uncertainty in measurement

**Returns:**
- `validation` (dict): Validation results and statistics

**Dictionary Keys:**
- `sigma_deviation` (float): Statistical significance of deviation
- `relative_error` (float): Relative error percentage
- `status` (str): Validation status ("VALIDATED", "SUPPORTED", etc.)
- `confidence` (str): Confidence level ("High", "Good", etc.)

##### `generate_predictions()`

```python
generate_predictions(df_measured) -> Dict[str, float]
```

**Description:** Generate MFSU-based predictions from measured fractal dimension.

**Parameters:**
- `df_measured` (float): Measured fractal dimension

**Returns:**
- `predictions` (dict): MFSU predictions based on measured df

---

## Preprocessing Module

### Class: `AstronomicalPreprocessor`

Scientific preprocessing for astronomical images.

#### Constructor

```python
AstronomicalPreprocessor(default_pixel_scale=0.12, detection_sigma=3.0)
```

**Parameters:**
- `default_pixel_scale` (float): Default pixel scale in arcsec/pixel
- `detection_sigma` (float): Detection threshold in units of noise sigma

#### Methods

##### `load_image()`

```python
load_image(image_path=None, image_data=None, verbose=True) -> Tuple[np.ndarray, Dict[str, Any]]
```

**Description:** Load astronomical image with metadata extraction.

**Parameters:**
- `image_path` (str, optional): Path to image file
- `image_data` (np.ndarray, optional): Direct numpy array input
- `verbose` (bool): Print loading information

**Returns:**
- `image` (np.ndarray): 2D image array (float64)
- `metadata` (dict): Image metadata and properties

##### `preprocess_image()`

```python
preprocess_image(image, background_method='corners', noise_method='mad', verbose=True) -> Tuple[np.ndarray, Dict[str, Any]]
```

**Description:** Scientific preprocessing of astronomical image.

**Parameters:**
- `image` (np.ndarray): Raw 2D image
- `background_method` (str): Background estimation method ('corners', 'sigma_clip', 'percentile')
- `noise_method` (str): Noise estimation method ('mad', 'std', 'robust')
- `verbose` (bool): Print processing information

**Returns:**
- `processed_image` (np.ndarray): Background-subtracted image
- `preprocessing_data` (dict): Processing metadata and derived products

**Preprocessing Data Keys:**
- `background` (float): Estimated background level
- `noise_std` (float): Estimated noise standard deviation
- `detection_threshold` (float): 3-sigma detection threshold
- `snr_map` (np.ndarray): Signal-to-noise ratio map
- `detection_mask` (np.ndarray): Binary detection mask
- `processing_quality` (dict): Quality assessment

##### `create_quality_report()`

```python
create_quality_report(image, processed_image, preprocessing_data, verbose=True) -> Dict[str, Any]
```

**Description:** Generate comprehensive quality report for preprocessing.

**Parameters:**
- `image` (np.ndarray): Original image
- `processed_image` (np.ndarray): Processed image
- `preprocessing_data` (dict): Preprocessing metadata
- `verbose` (bool): Print detailed report

**Returns:**
- `report` (dict): Comprehensive quality report

### Class: `DataLoader`

Specialized loader for various astronomical data formats.

#### Static Methods

##### `load_fits()`

```python
@staticmethod
load_fits(filepath, hdu=0) -> Tuple[np.ndarray, Dict[str, Any]]
```

**Description:** Load FITS file (requires astropy).

**Parameters:**
- `filepath` (str): Path to FITS file
- `hdu` (int): HDU number to load (default: 0)

**Returns:**
- `image` (np.ndarray): 2D image data
- `header` (dict): FITS header metadata

##### `load_jwst_pipeline()`

```python
@staticmethod
load_jwst_pipeline(filepath) -> Tuple[np.ndarray, Dict[str, Any]]
```

**Description:** Load JWST pipeline products with full metadata.

**Parameters:**
- `filepath` (str): Path to JWST file

**Returns:**
- `image` (np.ndarray): Science image
- `metadata` (dict): Complete JWST metadata

---

## Visualization Module

### Class: `MFSUVisualizer`

Publication-quality visualization for MFSU fractal analysis.

#### Constructor

```python
MFSUVisualizer(style='publication', dpi=300, figsize_large=(20, 12), figsize_medium=(16, 10))
```

**Parameters:**
- `style` (str): Plot style ('publication', 'presentation', 'paper')
- `dpi` (int): Resolution for saved figures
- `figsize_large` (tuple): Figure size for comprehensive plots
- `figsize_medium` (tuple): Figure size for focused plots

#### Methods

##### `create_comprehensive_analysis_plot()`

```python
create_comprehensive_analysis_plot(original_image, processed_image, preprocessing_data, 
                                 box_data, radial_data, interpretation_results, 
                                 pixel_scale, save_path=None, show=True) -> plt.Figure
```

**Description:** Create comprehensive analysis plot with all results.

**Parameters:**
- `original_image` (np.ndarray): Original astronomical image
- `processed_image` (np.ndarray): Background-subtracted image
- `preprocessing_data` (dict): Preprocessing metadata
- `box_data` (tuple): Box-counting analysis results
- `radial_data` (tuple): Radial profile analysis results
- `interpretation_results` (dict): Scientific interpretation results
- `pixel_scale` (float): Pixel scale in arcsec/pixel
- `save_path` (str, optional): Path to save figure
- `show` (bool): Whether to display figure

**Returns:**
- `fig` (matplotlib.Figure): Complete analysis figure

##### `create_focused_box_counting_plot()`

```python
create_focused_box_counting_plot(scales, counts, df_measured, df_error, r2_box, save_path=None) -> plt.Figure
```

**Description:** Create focused box-counting analysis plot.

**Parameters:**
- `scales` (np.ndarray): Physical scales (arcsec)
- `counts` (np.ndarray): Box counts
- `df_measured` (float): Measured fractal dimension
- `df_error` (float): Uncertainty in fractal dimension
- `r2_box` (float): R-squared value
- `save_path` (str, optional): Path to save figure

**Returns:**
- `fig` (matplotlib.Figure): Box-counting analysis figure

##### `create_focused_radial_plot()`

```python
create_focused_radial_plot(radii, intensities, intensity_errors, alpha, alpha_error, 
                         r2_radial, save_path=None) -> plt.Figure
```

**Description:** Create focused radial profile analysis plot.

**Parameters:**
- `radii` (np.ndarray): Radial coordinates
- `intensities` (np.ndarray): Intensity values
- `intensity_errors` (np.ndarray): Intensity uncertainties
- `alpha` (float): Radial slope
- `alpha_error` (float): Uncertainty in radial slope
- `r2_radial` (float): R-squared value
- `save_path` (str, optional): Path to save figure

**Returns:**
- `fig` (matplotlib.Figure): Radial profile analysis figure

##### `create_image_comparison_plot()`

```python
create_image_comparison_plot(original_image, processed_image, preprocessing_data, 
                           pixel_scale, save_path=None) -> plt.Figure
```

**Description:** Create image comparison plot showing preprocessing steps.

**Parameters:**
- `original_image` (np.ndarray): Original image
- `processed_image` (np.ndarray): Processed image
- `preprocessing_data` (dict): Preprocessing metadata
- `pixel_scale` (float): Pixel scale
- `save_path` (str, optional): Path to save figure

**Returns:**
- `fig` (matplotlib.Figure): Image comparison figure

### Function: `create_publication_figure()`

```python
create_publication_figure(analysis_results, save_path='mfsu_comet_analysis.pdf') -> plt.Figure
```

**Description:** Create final publication-ready figure with all analysis results.

**Parameters:**
- `analysis_results` (dict): Complete analysis results from MFSU pipeline
- `save_path` (str): Path to save publication figure

**Returns:**
- `fig` (matplotlib.Figure): Publication-ready figure

---

## Utilities Module

### Data Classes

#### `MFSUConstants`

```python
@dataclass
class MFSUConstants
```

**Attributes:**
- `DF_THEORETICAL` (float): Theoretical fractal dimension (2.079)
- `DELTA_THEORETICAL` (float): δp = 3 - df (0.921)
- `UNCERTAINTY_DF` (float): Theoretical uncertainty (0.003)
- `UNCERTAINTY_DELTA` (float): Theoretical uncertainty (0.003)
- `ARCSEC_PER_RADIAN` (float): Conversion factor (206265.0)
- `JWST_IFU_PIXEL_SCALE` (float): JWST IFU pixel scale (0.12)
- `JWST_IMAGER_PIXEL_SCALE` (float): JWST imager pixel scale (0.032)

#### `AnalysisConfig`

```python
@dataclass
class AnalysisConfig
```

**Attributes:**
- `min_box_size` (int): Minimum box size for counting (4)
- `max_scales` (int): Maximum number of scales (8)
- `min_pixels_per_box` (float): Minimum fraction for occupied box (0.25)
- `detection_sigma` (float): Detection threshold in sigma (3.0)
- `min_r_squared` (float): Minimum R² for good fit (0.8)

### Validation Functions

#### `validate_image_data()`

```python
validate_image_data(image, min_size=32, max_size=8192) -> None
```

**Description:** Validate image data for MFSU analysis.

**Parameters:**
- `image` (np.ndarray): 2D image array
- `min_size` (int): Minimum acceptable image dimension
- `max_size` (int): Maximum acceptable image dimension

**Raises:**
- `ValidationError`: If image data is invalid

#### `validate_pixel_scale()`

```python
validate_pixel_scale(pixel_scale, min_scale=0.001, max_scale=10.0) -> None
```

**Description:** Validate pixel scale parameter.

**Parameters:**
- `pixel_scale` (float): Pixel scale in arcsec/pixel
- `min_scale` (float): Minimum acceptable scale
- `max_scale` (float): Maximum acceptable scale

**Raises:**
- `ValidationError`: If pixel scale is invalid

#### `validate_analysis_results()`

```python
validate_analysis_results(box_data, radial_data) -> Dict[str, Any]
```

**Description:** Validate analysis results for consistency and quality.

**Parameters:**
- `box_data` (tuple): Box-counting analysis results
- `radial_data` (tuple): Radial profile analysis results

**Returns:**
- `validation_report` (dict): Comprehensive validation report

### Statistical Utilities

#### `robust_statistics()`

```python
robust_statistics(data, method='mad') -> Dict[str, float]
```

**Description:** Calculate robust statistical measures.

**Parameters:**
- `data` (np.ndarray): Input data array
- `method` (str): Method for robust estimation ('mad', 'percentile', 'iqr')

**Returns:**
- `stats` (dict): Dictionary of robust statistics

#### `calculate_uncertainties()`

```python
calculate_uncertainties(x, y, fit_params, method='bootstrap', n_bootstrap=1000) -> Tuple[float, float]
```

**Description:** Calculate parameter uncertainties using various methods.

**Parameters:**
- `x` (np.ndarray): Independent variable data
- `y` (np.ndarray): Dependent variable data
- `fit_params` (np.ndarray): Fitted parameters [slope, intercept]
- `method` (str): Uncertainty estimation method ('bootstrap', 'analytical', 'monte_carlo')
- `n_bootstrap` (int): Number of bootstrap samples

**Returns:**
- `slope_error` (float): Uncertainty in slope parameter
- `intercept_error` (float): Uncertainty in intercept parameter

#### `power_law_fit()`

```python
power_law_fit(x, y, weights=None) -> Dict[str, Any]
```

**Description:** Robust power law fitting with comprehensive diagnostics.

**Parameters:**
- `x` (np.ndarray): Independent variable (must be positive)
- `y` (np.ndarray): Dependent variable (must be positive)
- `weights` (np.ndarray, optional): Weights for fitting

**Returns:**
- `fit_results` (dict): Comprehensive fitting results

### Coordinate Utilities

#### `pixel_to_physical()`

```python
pixel_to_physical(pixel_coords, pixel_scale, center=None) -> np.ndarray
```

**Description:** Convert pixel coordinates to physical coordinates.

**Parameters:**
- `pixel_coords` (np.ndarray): Pixel coordinates (N x 2 array)
- `pixel_scale` (float): Pixel scale in arcsec/pixel
- `center` (tuple, optional): Center pixel coordinates (x, y)

**Returns:**
- `physical_coords` (np.ndarray): Physical coordinates in arcsec (N x 2 array)

#### `arcsec_to_au()`

```python
arcsec_to_au(arcsec, distance_pc) -> float
```

**Description:** Convert angular size to physical size.

**Parameters:**
- `arcsec` (float): Angular size in arcseconds
- `distance_pc` (float): Distance in parsecs

**Returns:**
- `au` (float): Physical size in astronomical units

### File I/O

#### `save_analysis_results()`

```python
save_analysis_results(results, filepath, include_images=False) -> None
```

**Description:** Save analysis results to JSON file.

**Parameters:**
- `results` (dict): Complete analysis results
- `filepath` (str or Path): Output file path
- `include_images` (bool): Whether to include image arrays

#### `load_analysis_results()`

```python
load_analysis_results(filepath) -> Dict[str, Any]
```

**Description:** Load analysis results from JSON file.

**Parameters:**
- `filepath` (str or Path): Input file path

**Returns:**
- `results` (dict): Loaded analysis results

### Logging

#### Class: `AnalysisLogger`

```python
AnalysisLogger(log_level='INFO')
```

**Methods:**
- `log(message, level='INFO')`: Log a message with timestamp
- `info(message)`: Log info message
- `warning(message)`: Log warning message
- `error(message)`: Log error message
- `get_summary()`: Get analysis summary

### Helper Functions

#### `format_scientific_notation()`

```python
format_scientific_notation(value, uncertainty, precision=3) -> str
```

**Description:** Format number with uncertainty in scientific notation.

#### `create_summary_table()`

```python
create_summary_table(results) -> str
```

**Description:** Create formatted summary table of analysis results.

#### `check_dependencies()`

```python
check_dependencies() -> Dict[str, bool]
```

**Description:** Check availability of optional dependencies.

#### `print_system_info()`

```python
print_system_info() -> None
```

**Description:** Print system and package information.

#### `run_full_validation()`

```python
run_full_validation(image, pixel_scale, analysis_results) -> Dict[str, Any]
```

**Description:** Run comprehensive validation of image and analysis results.

---

## Examples

### Basic Usage

```python
from mfsu_comet_analysis import MFSUCometReal
from mfsu_comet_analysis.fractal_analysis import FractalAnalyzer
from mfsu_comet_analysis.preprocessing import AstronomicalPreprocessor
from mfsu_comet_analysis.visualization import MFSUVisualizer

# Initialize components
analyzer = MFSUCometReal()
preprocessor = AstronomicalPreprocessor()
fractal_analyzer = FractalAnalyzer(pixel_scale=0.12)
visualizer = MFSUVisualizer()

# Load and preprocess image
image, metadata = preprocessor.load_image("comet_data.fits")
processed_image, preprocessing_data = preprocessor.preprocess_image(image)

# Perform fractal analysis
box_data = fractal_analyzer.advanced_box_counting(
    processed_image, preprocessing_data['detection_threshold']
)
radial_data = fractal_analyzer.advanced_radial_analysis(
    processed_image, preprocessing_data['detection_threshold']
)

# Scientific interpretation
df, df_err = box_data[0], box_data[1]
alpha, alpha_err = radial_data[0], radial_data[1]
results = fractal_analyzer.scientific_interpretation(df, df_err, alpha, alpha_err)

# Create publication plot
fig = visualizer.create_comprehensive_analysis_plot(
    image, processed_image, preprocessing_data,
    box_data, radial_data, results, 0.12
)
```

### Advanced Usage with Custom Configuration

```python
from mfsu_comet_analysis.utils import AnalysisConfig, run_full_validation
from mfsu_comet_analysis.fractal_analysis import FractalAnalyzer

# Custom configuration
config = AnalysisConfig(
    min_box_size=8,
    max_scales=12,
    detection_sigma=2.5,
    min_r_squared=0.85
)

# Initialize with custom settings
analyzer = FractalAnalyzer(pixel_scale=0.08)

# Perform analysis with custom parameters
box_data = analyzer.advanced_box_counting(
    processed_image, 
    threshold * 2.5/3.0,  # Adjust for custom sigma
    n_scales=12
)

# Validate results
validation = run_full_validation(image, 0.08, {
    'box_data': box_data,
    'radial_data': radial_data
})
print(f"Validation status: {validation['overall_status']}")
```

### FITS File Processing

```python
from mfsu_comet_analysis.preprocessing import DataLoader

# Load JWST FITS file
image, metadata = DataLoader.load_jwst_pipeline("jwst_data.fits")
print(f"Instrument: {metadata['instrument']}")
print(f"Filter: {metadata['filter']}")
print(f"Pixel scale: {metadata['pixel_scale_linear']:.4f} arcsec/pixel")

# Use extracted pixel scale
analyzer = FractalAnalyzer(pixel_scale=metadata['pixel_scale_linear'])
```

---

## Error Handling

### Exception Classes

#### `ValidationError`

Raised when input validation fails.

```python
from mfsu_comet_analysis.utils import ValidationError

try:
    validate_image_data(invalid_image)
except ValidationError as e:
    print(f"Validation failed: {e}")
```

### Common Error Scenarios

1. **Invalid Image Data**
   - Non-2D arrays
   - Images too small/large
   - Constant images
   - Non-finite values (NaN, Inf)

2. **Insufficient Data**
   - Too few scales for box-counting
   - Too few radial bins
   - Low detection fraction

3. **Poor Fit Quality**
   - R² below threshold
   - Large residuals
   - Unstable parameter estimates

4. **Missing Dependencies**
   - astropy for FITS support
   - scipy for advanced statistics

### Error Recovery

```python
from mfsu_comet_analysis.utils import validate_analysis_results

# Validate results and get recommendations
validation = validate_analysis_results(box_data, radial_data)

if not validation['valid']:
    print("Analysis failed:")
    for error in validation['errors']:
        print(f"  - {error}")
    
    print("Warnings:")
    for warning in validation['warnings']:
        print(f"  - {warning}")
else:
    print(f"Analysis successful (quality: {validation['quality_score']}/10)")
```

---

## Performance Considerations

### Memory Usage

- Images are processed as float64 arrays
- Large images (>2048×2048) may require substantial memory
- Set `include_images=False` when saving results to reduce file size

### Computational Complexity

- Box-counting: O(n_scales × n_pixels)
- Radial analysis: O(n_bins × n_pixels)
- Typical analysis time: 10-60 seconds for 512×512 images

### Optimization Tips

1. **Use appropriate pixel scales** to avoid unnecessarily high resolution
2. **Limit number of scales** for faster box-counting
3. **Reduce radial bins** for faster profile analysis
4. **Enable verbose=False** for batch processing

---

## Version History

### Version 1.0.0 (September 2025)
- Initial release
- Complete MFSU framework implementation
- Support for JWST data
- Publication-quality visualization
- Comprehensive validation and error handling

---

## See Also

- [User Manual](USER_MANUAL.md) - Complete usage guide
- [Installation Guide](INSTALLATION.md) - Setup instructions
- [Methodology Documentation](methodology.md) - Theoretical background
- [Examples](../examples/) - Additional usage examples

---

*For questions or issues, please refer to the troubleshooting section in the User Manual or contact the development team.*
