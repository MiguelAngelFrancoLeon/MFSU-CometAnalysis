#!/usr/bin/env python3
"""
MFSU Comet Analysis - Utilities Module
======================================

Utility functions, constants, validators, and helper tools
for the MFSU comet analysis framework.

Author: Miguel Ángel Franco León & Claude
Date: September 2025
"""

import numpy as np
from typing import Tuple, Dict, Optional, Any, List, Union
import warnings
from pathlib import Path
import json
import time
from dataclasses import dataclass
from datetime import datetime

# ============================================================================
# CONSTANTS AND CONFIGURATION
# ============================================================================

@dataclass
class MFSUConstants:
    """MFSU theoretical constants and predictions."""
    DF_THEORETICAL: float = 2.079  # Theoretical fractal dimension
    DELTA_THEORETICAL: float = 0.921  # δp = 3 - df
    UNCERTAINTY_DF: float = 0.003  # Theoretical uncertainty
    UNCERTAINTY_DELTA: float = 0.003  # Theoretical uncertainty
    
    # Physical constants
    ARCSEC_PER_RADIAN: float = 206265.0
    PLANCK_H: float = 6.62607015e-34  # J⋅s
    SPEED_OF_LIGHT: float = 299792458.0  # m/s
    
    # JWST instrument specifications
    JWST_IFU_PIXEL_SCALE: float = 0.12  # arcsec/pixel (typical)
    JWST_IMAGER_PIXEL_SCALE: float = 0.032  # arcsec/pixel (typical)
    JWST_WAVELENGTH_RANGE: Tuple[float, float] = (0.6, 28.0)  # μm

@dataclass
class AnalysisConfig:
    """Configuration parameters for MFSU analysis."""
    # Box-counting parameters
    min_box_size: int = 4
    max_scales: int = 8
    min_pixels_per_box: float = 0.25
    min_scales_required: int = 4
    
    # Radial analysis parameters
    min_radial_bins: int = 5
    max_radial_bins: int = 25
    radial_extent_fraction: float = 0.8
    min_pixels_per_annulus: int = 3
    
    # Detection parameters
    detection_sigma: float = 3.0
    background_sigma_clip: int = 3
    noise_estimation_method: str = 'mad'
    
    # Quality thresholds
    min_r_squared: float = 0.8
    min_detection_fraction: float = 0.005
    max_detection_fraction: float = 0.7
    min_snr: float = 5.0

# Global instances
MFSU_CONSTANTS = MFSUConstants()
DEFAULT_CONFIG = AnalysisConfig()

# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

class ValidationError(Exception):
    """Custom exception for validation failures."""
    pass

def validate_image_data(image: np.ndarray, 
                       min_size: int = 32,
                       max_size: int = 8192) -> None:
    """
    Validate image data for MFSU analysis.
    
    Parameters:
    -----------
    image : np.ndarray
        2D image array
    min_size : int
        Minimum acceptable image dimension
    max_size : int
        Maximum acceptable image dimension
        
    Raises:
    -------
    ValidationError
        If image data is invalid
    """
    if not isinstance(image, np.ndarray):
        raise ValidationError("Image must be a numpy array")
    
    if image.ndim != 2:
        raise ValidationError(f"Image must be 2D, got {image.ndim}D")
    
    h, w = image.shape
    if h < min_size or w < min_size:
        raise ValidationError(f"Image too small: {h}x{w}, minimum {min_size}x{min_size}")
    
    if h > max_size or w > max_size:
        raise ValidationError(f"Image too large: {h}x{w}, maximum {max_size}x{max_size}")
    
    if not np.isfinite(image).all():
        raise ValidationError("Image contains non-finite values (NaN or Inf)")
    
    if np.all(image == image.flat[0]):
        raise ValidationError("Image is constant (no variation)")

def validate_pixel_scale(pixel_scale: float,
                        min_scale: float = 0.001,
                        max_scale: float = 10.0) -> None:
    """
    Validate pixel scale parameter.
    
    Parameters:
    -----------
    pixel_scale : float
        Pixel scale in arcsec/pixel
    min_scale : float
        Minimum acceptable scale
    max_scale : float
        Maximum acceptable scale
        
    Raises:
    -------
    ValidationError
        If pixel scale is invalid
    """
    if not isinstance(pixel_scale, (int, float)):
        raise ValidationError("Pixel scale must be numeric")
    
    if pixel_scale <= 0:
        raise ValidationError("Pixel scale must be positive")
    
    if pixel_scale < min_scale:
        raise ValidationError(f"Pixel scale too small: {pixel_scale}, minimum {min_scale}")
    
    if pixel_scale > max_scale:
        raise ValidationError(f"Pixel scale too large: {pixel_scale}, maximum {max_scale}")

def validate_analysis_results(box_data: Tuple, radial_data: Tuple) -> Dict[str, Any]:
    """
    Validate analysis results for consistency and quality.
    
    Parameters:
    -----------
    box_data : tuple
        Box-counting analysis results
    radial_data : tuple
        Radial profile analysis results
        
    Returns:
    --------
    validation_report : dict
        Comprehensive validation report
    """
    report = {
        'valid': True,
        'warnings': [],
        'errors': [],
        'quality_score': 0
    }
    
    # Unpack data
    try:
        df, df_err, scales, counts, r2_box, box_sizes = box_data
        alpha, alpha_err, radii, intensities, r2_radial, intensity_errors = radial_data
    except (ValueError, TypeError) as e:
        report['errors'].append(f"Failed to unpack analysis data: {e}")
        report['valid'] = False
        return report
    
    # Validate box-counting results
    if not (0.5 <= df <= 3.0):
        report['warnings'].append(f"Unusual fractal dimension: {df:.3f}")
    
    if df_err <= 0 or df_err > 0.5:
        report['warnings'].append(f"Unusual fractal dimension error: {df_err:.3f}")
    
    if r2_box < DEFAULT_CONFIG.min_r_squared:
        report['warnings'].append(f"Low box-counting R²: {r2_box:.3f}")
    
    if len(scales) < DEFAULT_CONFIG.min_scales_required:
        report['errors'].append(f"Insufficient scales: {len(scales)}")
        report['valid'] = False
    
    # Validate radial results
    if not (0.1 <= alpha <= 5.0):
        report['warnings'].append(f"Unusual radial slope: {alpha:.3f}")
    
    if alpha_err <= 0 or alpha_err > 1.0:
        report['warnings'].append(f"Unusual radial slope error: {alpha_err:.3f}")
    
    if r2_radial < DEFAULT_CONFIG.min_r_squared:
        report['warnings'].append(f"Low radial profile R²: {r2_radial:.3f}")
    
    if len(radii) < DEFAULT_CONFIG.min_radial_bins:
        report['errors'].append(f"Insufficient radial bins: {len(radii)}")
        report['valid'] = False
    
    # Calculate quality score (0-10)
    quality_score = 10
    quality_score -= len(report['warnings']) * 1  # -1 per warning
    quality_score -= len(report['errors']) * 3    # -3 per error
    quality_score += min(2, (r2_box - 0.8) * 10)  # Bonus for good fits
    quality_score += min(2, (r2_radial - 0.8) * 10)
    
    report['quality_score'] = max(0, min(10, quality_score))
    
    return report

# ============================================================================
# STATISTICAL UTILITIES
# ============================================================================

def robust_statistics(data: np.ndarray, 
                     method: str = 'mad') -> Dict[str, float]:
    """
    Calculate robust statistical measures.
    
    Parameters:
    -----------
    data : np.ndarray
        Input data array
    method : str
        Method for robust estimation ('mad', 'percentile', 'iqr')
        
    Returns:
    --------
    stats : dict
        Dictionary of robust statistics
    """
    data_clean = data[np.isfinite(data)]
    
    if len(data_clean) == 0:
        return {'median': np.nan, 'robust_std': np.nan, 'iqr': np.nan}
    
    median = np.median(data_clean)
    
    if method == 'mad':
        # Median Absolute Deviation
        mad = np.median(np.abs(data_clean - median))
        robust_std = 1.4826 * mad  # Convert to std estimate
    elif method == 'percentile':
        # Inter-percentile range (P10-P90)
        p10, p90 = np.percentile(data_clean, [10, 90])
        robust_std = (p90 - p10) / 2.56  # Convert to std estimate
    elif method == 'iqr':
        # Interquartile range
        q1, q3 = np.percentile(data_clean, [25, 75])
        robust_std = (q3 - q1) / 1.349  # Convert to std estimate
    else:
        raise ValueError(f"Unknown robust method: {method}")
    
    # Additional statistics
    iqr = np.percentile(data_clean, 75) - np.percentile(data_clean, 25)
    
    return {
        'median': median,
        'robust_std': robust_std,
        'iqr': iqr,
        'mad': np.median(np.abs(data_clean - median)),
        'n_valid': len(data_clean),
        'outlier_fraction': 1 - len(data_clean) / len(data)
    }

def calculate_uncertainties(x: np.ndarray, y: np.ndarray, 
                          fit_params: np.ndarray,
                          method: str = 'bootstrap',
                          n_bootstrap: int = 1000) -> Tuple[float, float]:
    """
    Calculate parameter uncertainties using various methods.
    
    Parameters:
    -----------
    x : np.ndarray
        Independent variable data
    y : np.ndarray
        Dependent variable data
    fit_params : np.ndarray
        Fitted parameters [slope, intercept]
    method : str
        Uncertainty estimation method ('bootstrap', 'analytical', 'monte_carlo')
    n_bootstrap : int
        Number of bootstrap samples
        
    Returns:
    --------
    slope_error : float
        Uncertainty in slope parameter
    intercept_error : float
        Uncertainty in intercept parameter
    """
    if method == 'bootstrap':
        slopes = []
        intercepts = []
        
        for _ in range(n_bootstrap):
            # Resample with replacement
            indices = np.random.choice(len(x), len(x), replace=True)
            x_boot = x[indices]
            y_boot = y[indices]
            
            # Fit line
            try:
                params = np.polyfit(x_boot, y_boot, 1)
                slopes.append(params[0])
                intercepts.append(params[1])
            except np.linalg.LinAlgError:
                continue  # Skip failed fits
        
        slope_error = np.std(slopes)
        intercept_error = np.std(intercepts)
        
    elif method == 'analytical':
        # Analytical uncertainties from linear regression
        n = len(x)
        if n < 3:
            return np.nan, np.nan
        
        # Calculate residuals
        y_pred = np.polyval(fit_params, x)
        residuals = y - y_pred
        mse = np.sum(residuals**2) / (n - 2)
        
        # Standard errors
        sx2 = np.sum((x - np.mean(x))**2)
        slope_error = np.sqrt(mse / sx2)
        intercept_error = np.sqrt(mse * (1/n + np.mean(x)**2/sx2))
        
    else:
        raise ValueError(f"Unknown uncertainty method: {method}")
    
    return slope_error, intercept_error

def power_law_fit(x: np.ndarray, y: np.ndarray,
                 weights: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """
    Robust power law fitting with comprehensive diagnostics.
    
    Parameters:
    -----------
    x : np.ndarray
        Independent variable (must be positive)
    y : np.ndarray
        Dependent variable (must be positive)
    weights : np.ndarray, optional
        Weights for fitting
        
    Returns:
    --------
    fit_results : dict
        Comprehensive fitting results
    """
    # Validate inputs
    if np.any(x <= 0) or np.any(y <= 0):
        raise ValueError("Power law fitting requires positive values")
    
    # Log-transform
    log_x = np.log10(x)
    log_y = np.log10(y)
    
    # Weighted fitting if weights provided
    if weights is not None:
        fit_params, cov = np.polyfit(log_x, log_y, 1, w=weights, cov=True)
    else:
        fit_params, cov = np.polyfit(log_x, log_y, 1, cov=True)
    
    slope, intercept = fit_params
    slope_err = np.sqrt(cov[0, 0])
    intercept_err = np.sqrt(cov[1, 1])
    
    # Calculate R-squared
    y_pred = np.polyval(fit_params, log_x)
    ss_res = np.sum((log_y - y_pred)**2)
    ss_tot = np.sum((log_y - np.mean(log_y))**2)
    r_squared = 1 - (ss_res / ss_tot)
    
    # Calculate residuals and diagnostics
    residuals = log_y - y_pred
    rmse = np.sqrt(np.mean(residuals**2))
    max_residual = np.max(np.abs(residuals))
    
    # Convert back to linear space parameters
    # y = A * x^B => log(y) = log(A) + B*log(x)
    # slope = B, intercept = log(A)
    power_index = slope
    amplitude = 10**intercept
    
    return {
        'power_index': power_index,
        'power_index_error': slope_err,
        'amplitude': amplitude,
        'amplitude_error': amplitude * np.log(10) * intercept_err,
        'r_squared': r_squared,
        'rmse': rmse,
        'max_residual': max_residual,
        'residuals': residuals,
        'fit_params_log': fit_params,
        'covariance_matrix': cov,
        'n_points': len(x)
    }

# ============================================================================
# COORDINATE AND UNIT UTILITIES
# ============================================================================

def pixel_to_physical(pixel_coords: np.ndarray, 
                     pixel_scale: float,
                     center: Optional[Tuple[float, float]] = None) -> np.ndarray:
    """
    Convert pixel coordinates to physical coordinates.
    
    Parameters:
    -----------
    pixel_coords : np.ndarray
        Pixel coordinates (N x 2 array)
    pixel_scale : float
        Pixel scale in arcsec/pixel
    center : tuple, optional
        Center pixel coordinates (x, y). If None, uses image center
        
    Returns:
    --------
    physical_coords : np.ndarray
        Physical coordinates in arcsec (N x 2 array)
    """
    if center is None:
        center = (0, 0)  # Assume origin at (0,0)
    
    # Offset from center
    dx = pixel_coords[:, 0] - center[0]
    dy = pixel_coords[:, 1] - center[1]
    
    # Convert to physical units
    physical_coords = np.column_stack([dx * pixel_scale, dy * pixel_scale])
    
    return physical_coords

def calculate_radial_distance(coordinates: np.ndarray,
                            center: Tuple[float, float]) -> np.ndarray:
    """
    Calculate radial distances from center.
    
    Parameters:
    -----------
    coordinates : np.ndarray
        Coordinate array (N x 2)
    center : tuple
        Center coordinates (x, y)
        
    Returns:
    --------
    distances : np.ndarray
        Radial distances from center
    """
    dx = coordinates[:, 0] - center[0]
    dy = coordinates[:, 1] - center[1]
    return np.sqrt(dx**2 + dy**2)

def arcsec_to_au(arcsec: float, distance_pc: float) -> float:
    """
    Convert angular size to physical size.
    
    Parameters:
    -----------
    arcsec : float
        Angular size in arcseconds
    distance_pc : float
        Distance in parsecs
        
    Returns:
    --------
    au : float
        Physical size in astronomical units
    """
    # 1 arcsec at 1 pc = 1 AU
    return arcsec * distance_pc

def estimate_comet_distance(apparent_size_arcsec: float,
                          typical_size_km: float = 10.0) -> float:
    """
    Estimate comet distance from apparent size.
    
    Parameters:
    -----------
    apparent_size_arcsec : float
        Apparent angular size in arcseconds
    typical_size_km : float
        Typical physical size in kilometers
        
    Returns:
    --------
    distance_au : float
        Estimated distance in AU
    """
    # Convert km to AU
    km_per_au = 149597870.7
    size_au = typical_size_km / km_per_au
    
    # Angular size formula: θ = size / distance
    # θ in radians = size_au / distance_au
    # θ in arcsec = θ_rad * 206265
    distance_au = size_au * MFSU_CONSTANTS.ARCSEC_PER_RADIAN / apparent_size_arcsec
    
    return distance_au

# ============================================================================
# FILE I/O AND SERIALIZATION
# ============================================================================

def save_analysis_results(results: Dict[str, Any], 
                         filepath: Union[str, Path],
                         include_images: bool = False) -> None:
    """
    Save analysis results to JSON file.
    
    Parameters:
    -----------
    results : dict
        Complete analysis results
    filepath : str or Path
        Output file path
    include_images : bool
        Whether to include image arrays (makes file very large)
    """
    filepath = Path(filepath)
    
    # Create serializable copy
    serializable_results = {}
    
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            if include_images or value.ndim <= 1:  # Include 1D arrays always
                serializable_results[key] = {
                    'type': 'numpy_array',
                    'data': value.tolist(),
                    'shape': value.shape,
                    'dtype': str(value.dtype)
                }
            else:
                serializable_results[key] = {
                    'type': 'numpy_array_excluded',
                    'shape': value.shape,
                    'dtype': str(value.dtype)
                }
        elif isinstance(value, dict):
            serializable_results[key] = value
        else:
            serializable_results[key] = value
    
    # Add metadata
    serializable_results['_metadata'] = {
        'saved_at': datetime.now().isoformat(),
        'mfsu_version': '1.0.0',
        'include_images': include_images
    }
    
    # Save to JSON
    with open(filepath, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"✅ Analysis results saved: {filepath}")

def load_analysis_results(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Load analysis results from JSON file.
    
    Parameters:
    -----------
    filepath : str or Path
        Input file path
        
    Returns:
    --------
    results : dict
        Loaded analysis results
    """
    filepath = Path(filepath)
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Reconstruct numpy arrays
    results = {}
    for key, value in data.items():
        if isinstance(value, dict) and value.get('type') == 'numpy_array':
            results[key] = np.array(value['data'], dtype=value['dtype']).reshape(value['shape'])
        elif isinstance(value, dict) and value.get('type') == 'numpy_array_excluded':
            print(f"Warning: Array {key} was excluded from saved file")
            results[key] = None
        else:
            results[key] = value
    
    print(f"✅ Analysis results loaded: {filepath}")
    return results

# ============================================================================
# LOGGING AND PROGRESS TRACKING
# ============================================================================

class AnalysisLogger:
    """Simple logger for analysis progress and results."""
    
    def __init__(self, log_level: str = 'INFO'):
        self.log_level = log_level
        self.start_time = time.time()
        self.steps = []
        
    def log(self, message: str, level: str = 'INFO'):
        """Log a message with timestamp."""
        timestamp = time.time() - self.start_time
        log_entry = f"[{timestamp:7.2f}s] {level}: {message}"
        print(log_entry)
        
        self.steps.append({
            'timestamp': timestamp,
            'level': level,
            'message': message
        })
    
    def info(self, message: str):
        """Log info message."""
        self.log(message, 'INFO')
    
    def warning(self, message: str):
        """Log warning message."""
        self.log(message, 'WARNING')
    
    def error(self, message: str):
        """Log error message."""
        self.log(message, 'ERROR')
    
    def get_summary(self) -> Dict[str, Any]:
        """Get analysis summary."""
        total_time = time.time() - self.start_time
        
        return {
            'total_time': total_time,
            'n_steps': len(self.steps),
            'n_warnings': len([s for s in self.steps if s['level'] == 'WARNING']),
            'n_errors': len([s for s in self.steps if s['level'] == 'ERROR']),
            'steps': self.steps
        }

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def format_scientific_notation(value: float, 
                              uncertainty: float,
                              precision: int = 3) -> str:
    """
    Format number with uncertainty in scientific notation.
    
    Parameters:
    -----------
    value : float
        Central value
    uncertainty : float
        Uncertainty
    precision : int
        Number of significant digits
        
    Returns:
    --------
    formatted : str
        Formatted string like "2.079 ± 0.003"
    """
    if uncertainty <= 0:
        return f"{value:.{precision}f}"
    
    # Find the order of magnitude of uncertainty
    if uncertainty > 0:
        unc_order = int(np.floor(np.log10(uncertainty)))
        
        # Round uncertainty to 1-2 significant figures
        unc_rounded = np.round(uncertainty, -unc_order + 1)
        
        # Round value to same decimal place
        value_rounded = np.round(value, -unc_order + 1)
        
        # Format with appropriate decimal places
        decimal_places = max(0, -unc_order + 1)
        
        return f"{value_rounded:.{decimal_places}f} ± {unc_rounded:.{decimal_places}f}"
    else:
        return f"{value:.{precision}f}"

def create_summary_table(results: Dict[str, Any]) -> str:
    """
    Create formatted summary table of analysis results.
    
    Parameters:
    -----------
    results : dict
        Analysis results dictionary
        
    Returns:
    --------
    table : str
        Formatted summary table
    """
    table_lines = [
        "MFSU COMET ANALYSIS SUMMARY",
        "=" * 50
    ]
    
    # Extract key results
    df_measured = results.get('df_measured', np.nan)
    df_error = results.get('df_error', np.nan)
    alpha = results.get('alpha', np.nan)
    alpha_error = results.get('alpha_error', np.nan)
    
    # Format parameters
    df_str = format_scientific_notation(df_measured, df_error)
    alpha_str = format_scientific_notation(alpha, alpha_error)
    delta_str = format_scientific_notation(3 - df_measured, df_error)
    
    table_lines.extend([
        f"Fractal dimension:     df = {df_str}",
        f"Radial slope:          α  = {alpha_str}",
        f"Derived parameter:     δp = {delta_str}",
        "",
        "MFSU COMPARISON:",
        f"Theoretical df:        {MFSU_CONSTANTS.DF_THEORETICAL:.3f}",
        f"Theoretical δp:        {MFSU_CONSTANTS.DELTA_THEORETICAL:.3f}",
        f"Agreement:             {results.get('mfsu_agreement', 'Unknown')}",
        "",
        "QUALITY METRICS:",
        f"Box-counting R²:       {results.get('r2_box', np.nan):.4f}",
        f"Radial profile R²:     {results.get('r2_radial', np.nan):.4f}",
        f"Classification:        {results.get('df_class', 'Unknown')}"
    ])
    
    return "\n".join(table_lines)

def check_dependencies() -> Dict[str, bool]:
    """
    Check availability of optional dependencies.
    
    Returns:
    --------
    dependencies : dict
        Dictionary of dependency availability
    """
    dependencies = {}
    
    # Check astropy for FITS support
    try:
        import astropy
        dependencies['astropy'] = True
    except ImportError:
        dependencies['astropy'] = False
    
    # Check scipy for advanced statistics
    try:
        import scipy
        dependencies['scipy'] = True
    except ImportError:
        dependencies['scipy'] = False
    
    # Check sklearn for advanced analysis
    try:
        import sklearn
        dependencies['sklearn'] = True
    except ImportError:
        dependencies['sklearn'] = False
    
    return dependencies

def print_system_info():
    """Print system and package information."""
    import sys
    import platform
    
    print("MFSU COMET ANALYSIS - SYSTEM INFORMATION")
    print("=" * 50)
    print(f"Python version:        {sys.version}")
    print(f"Platform:              {platform.platform()}")
    print(f"NumPy version:         {np.__version__}")
    
    try:
        import matplotlib
        print(f"Matplotlib version:    {matplotlib.__version__}")
    except ImportError:
        print("Matplotlib:            Not available")
    
    deps = check_dependencies()
    print("\nOptional dependencies:")
    for name, available in deps.items():
        status = "✅ Available" if available else "❌ Not available"
        print(f"  {name:15}: {status}")
    
    print(f"\nMFSU Constants:")
    print(f"  df theoretical:      {MFSU_CONSTANTS.DF_THEORETICAL}")
    print(f"  δp theoretical:      {MFSU_CONSTANTS.DELTA_THEORETICAL}")

# ============================================================================
# MAIN UTILITY FUNCTION
# ============================================================================

def run_full_validation(image: np.ndarray,
                       pixel_scale: float,
                       analysis_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run comprehensive validation of image and analysis results.
    
    Parameters:
    -----------
    image : np.ndarray
        Input image
    pixel_scale : float
        Pixel scale in arcsec/pixel
    analysis_results : dict
        Complete analysis results
        
    Returns:
    --------
    validation_report : dict
        Comprehensive validation report
    """
    logger = AnalysisLogger()
    logger.info("Starting comprehensive validation")
    
    validation_report = {
        'image_validation': {},
        'scale_validation': {},
        'results_validation': {},
        'overall_status': 'UNKNOWN',
        'recommendations': []
    }
    
    # Validate image
    try:
        validate_image_data(image)
        validation_report['image_validation'] = {'status': 'PASSED', 'errors': []}
        logger.info("Image validation passed")
    except ValidationError as e:
        validation_report['image_validation'] = {'status': 'FAILED', 'errors': [str(e)]}
        logger.error(f"Image validation failed: {e}")
    
    # Validate pixel scale
    try:
        validate_pixel_scale(pixel_scale)
        validation_report['scale_validation'] = {'status': 'PASSED', 'errors': []}
        logger.info("Pixel scale validation passed")
    except ValidationError as e:
        validation_report['scale_validation'] = {'status': 'FAILED', 'errors': [str(e)]}
        logger.error(f"Pixel scale validation failed: {e}")
    
    # Validate analysis results
    if 'box_data' in analysis_results and 'radial_data' in analysis_results:
        results_validation = validate_analysis_results(
            analysis_results['box_data'], 
            analysis_results['radial_data']
        )
        validation_report['results_validation'] = results_validation
        
        if results_validation['valid']:
            logger.info(f"Results validation passed (quality score: {results_validation['quality_score']}/10)")
        else:
            logger.error("Results validation failed")
    
    # Overall status
    all_passed = (
        validation_report['image_validation'].get('status') == 'PASSED' and
        validation_report['scale_validation'].get('status') == 'PASSED' and
        validation_report['results_validation'].get('valid', False)
    )
    
    validation_report['overall_status'] = 'PASSED' if all_passed else 'FAILED'
    validation_report['logger_summary'] = logger.get_summary()
    
    logger.info(f"Validation complete: {validation_report['overall_status']}")
    
    return validation_report
