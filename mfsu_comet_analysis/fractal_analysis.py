#!/usr/bin/env python3
"""
MFSU Comet Analysis - Fractal Analysis Module
===========================================

Advanced fractal analysis algorithms for astronomical objects using MFSU framework.
Implements box-counting, radial profile analysis, and scientific interpretation.

Author: Miguel Ángel Franco León & Claude
Date: September 2025
"""

import numpy as np
from scipy import stats
from typing import Tuple, Dict, Optional, Any
import warnings

class FractalAnalyzer:
    """
    Advanced fractal analysis for astronomical objects using MFSU framework.
    
    This class implements rigorous fractal analysis methods specifically
    designed for astronomical data, with proper error propagation and
    statistical validation.
    """
    
    def __init__(self, pixel_scale: float = 0.12):
        """
        Initialize fractal analyzer.
        
        Parameters:
        -----------
        pixel_scale : float
            Pixel scale in arcsec/pixel (default: 0.12 for JWST IFU)
        """
        self.pixel_scale = pixel_scale
        self.df_theoretical = 2.079  # MFSU prediction
        self.delta_theoretical = 0.921  # δ = 3 - df
        
        # Analysis parameters
        self.min_box_size = 4  # Minimum box size in pixels
        self.min_pixels_per_box = 0.25  # Minimum fraction for occupied box
        self.min_scales = 4  # Minimum number of scales for valid analysis
        
    def advanced_box_counting(self, 
                            image: np.ndarray, 
                            detection_threshold: float,
                            n_scales: int = 8,
                            verbose: bool = True) -> Tuple[float, float, np.ndarray, np.ndarray, float, np.ndarray]:
        """
        Advanced box-counting analysis with astronomical considerations.
        
        Implements logarithmic scale progression, physical units,
        and robust statistical analysis.
        
        Parameters:
        -----------
        image : np.ndarray
            2D processed image (background-subtracted)
        detection_threshold : float
            3-sigma detection threshold for binary mask
        n_scales : int
            Maximum number of scales to analyze
        verbose : bool
            Print progress information
            
        Returns:
        --------
        df_measured : float
            Measured fractal dimension
        df_error : float
            Uncertainty in fractal dimension
        scales : np.ndarray
            Physical scales used (arcsec)
        counts : np.ndarray
            Occupied box counts at each scale
        r_squared : float
            Quality of linear fit (R²)
        box_sizes : np.ndarray
            Box sizes in pixels
        """
        if verbose:
            print("\n📊 Advanced astronomical box-counting analysis...")
        
        # Create binary detection mask
        binary = (image > detection_threshold).astype(float)
        total_signal_pixels = np.sum(binary)
        
        if verbose:
            print(f"   Detection threshold: {detection_threshold:.3f}")
            print(f"   Detected pixels: {total_signal_pixels} / {binary.size}")
            print(f"   Detection fraction: {total_signal_pixels/binary.size*100:.2f}%")
        
        if total_signal_pixels < 100:
            warnings.warn("Low detection count - results may be uncertain")
            
        # Initialize scale arrays
        scales = []
        counts = []
        box_sizes = []
        
        h, w = image.shape
        min_dim = min(h, w)
        
        # Logarithmic scale progression
        for i in range(n_scales):
            # Geometric progression of box sizes
            box_size = max(self.min_box_size, int(min_dim * (0.5 ** i)))
            
            if box_size < self.min_box_size:
                break
                
            occupied_boxes = self._count_occupied_boxes(binary, box_size)
            
            # Scale in physical units (arcsec)
            scale_arcsec = box_size * self.pixel_scale
            
            scales.append(scale_arcsec)
            counts.append(occupied_boxes)
            box_sizes.append(box_size)
            
            if verbose:
                n_boxes_total = (h // box_size) * (w // box_size)
                print(f"   Scale {len(scales)}: box={box_size}px "
                      f"({scale_arcsec:.3f}arcsec), occupied={occupied_boxes}/{n_boxes_total}")
        
        # Convert to arrays
        scales = np.array(scales)
        counts = np.array(counts)
        box_sizes = np.array(box_sizes)
        
        # Validate sufficient scales
        if len(scales) < self.min_scales:
            raise ValueError(f"Insufficient scales for analysis (only {len(scales)})")
        
        # Calculate fractal dimension
        df_measured, df_error, r_squared = self._calculate_fractal_dimension(scales, counts)
        
        if verbose:
            print(f"   ✅ Box-counting results:")
            print(f"      df = {df_measured:.3f} ± {df_error:.3f}")
            print(f"      R² = {r_squared:.4f}")
            print(f"      Scales used: {len(scales)}")
        
        return df_measured, df_error, scales, counts, r_squared, box_sizes
    
    def _count_occupied_boxes(self, binary_image: np.ndarray, box_size: int) -> int:
        """
        Count boxes containing significant signal.
        
        Parameters:
        -----------
        binary_image : np.ndarray
            Binary detection mask
        box_size : int
            Size of counting boxes in pixels
            
        Returns:
        --------
        occupied_boxes : int
            Number of boxes with significant signal
        """
        h, w = binary_image.shape
        n_boxes_y = h // box_size
        n_boxes_x = w // box_size
        
        occupied_boxes = 0
        
        for i_box in range(n_boxes_y):
            for j_box in range(n_boxes_x):
                y_start = i_box * box_size
                y_end = min((i_box + 1) * box_size, h)
                x_start = j_box * box_size  
                x_end = min((j_box + 1) * box_size, w)
                
                box_data = binary_image[y_start:y_end, x_start:x_end]
                
                # Box is "occupied" if it contains significant signal
                if np.sum(box_data) > self.min_pixels_per_box:
                    occupied_boxes += 1
                    
        return occupied_boxes
    
    def _calculate_fractal_dimension(self, scales: np.ndarray, counts: np.ndarray) -> Tuple[float, float, float]:
        """
        Calculate fractal dimension from box-counting data.
        
        Uses robust linear regression with proper error propagation.
        
        Parameters:
        -----------
        scales : np.ndarray
            Physical scales (arcsec)
        counts : np.ndarray
            Box counts at each scale
            
        Returns:
        --------
        df : float
            Fractal dimension
        df_error : float
            Uncertainty in fractal dimension
        r_squared : float
            Quality of linear fit
        """
        # Fractal scaling: N(ε) ~ ε^(-df) => log(N) ~ -df * log(ε)
        log_scales = np.log10(scales)
        log_counts = np.log10(counts)
        
        # Linear regression with covariance
        coeffs, cov = np.polyfit(log_scales, log_counts, 1, cov=True)
        df_measured = -coeffs[0]  # Negative of slope
        df_error = np.sqrt(cov[0, 0])
        
        # Calculate R-squared
        predicted_log_counts = np.polyval(coeffs, log_scales)
        ss_res = np.sum((log_counts - predicted_log_counts)**2)
        ss_tot = np.sum((log_counts - np.mean(log_counts))**2)
        r_squared = 1 - (ss_res / ss_tot)
        
        return df_measured, df_error, r_squared
    
    def advanced_radial_analysis(self, 
                                image: np.ndarray,
                                detection_threshold: float,
                                center: Optional[Tuple[float, float]] = None,
                                n_bins: int = 20,
                                verbose: bool = True) -> Tuple[float, float, np.ndarray, np.ndarray, float, np.ndarray]:
        """
        Advanced radial profile analysis with astronomical techniques.
        
        Implements photometric centroiding, logarithmic binning,
        and robust power-law fitting.
        
        Parameters:
        -----------
        image : np.ndarray
            2D processed image
        detection_threshold : float
            Detection threshold for centroid calculation
        center : tuple, optional
            Manual center coordinates (x, y). If None, calculates photometric center
        n_bins : int
            Number of radial bins
        verbose : bool
            Print progress information
            
        Returns:
        --------
        alpha : float
            Radial slope (I(r) ~ r^(-alpha))
        alpha_error : float
            Uncertainty in radial slope
        radii : np.ndarray
            Radial coordinates (arcsec)
        intensities : np.ndarray
            Mean intensities at each radius
        r_squared : float
            Quality of power-law fit
        intensity_errors : np.ndarray
            Uncertainties in mean intensities
        """
        if verbose:
            print("\n🎯 Advanced radial profile analysis...")
        
        # Calculate photometric center
        if center is None:
            center_x, center_y = self._calculate_photometric_center(image, detection_threshold)
        else:
            center_x, center_y = center
            
        if verbose:
            print(f"   Photometric center: ({center_x:.1f}, {center_y:.1f})")
        
        # Create radius grid
        h, w = image.shape
        y_coords, x_coords = np.ogrid[:h, :w]
        radius_pix = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
        radius_arcsec = radius_pix * self.pixel_scale
        
        # Determine radial range
        max_radius_pix = min(center_x, center_y, w - center_x, h - center_y)
        max_radius_arcsec = max_radius_pix * self.pixel_scale
        
        # Logarithmic radial binning
        r_min = 2 * self.pixel_scale  # Minimum radius (2 pixels)
        r_max = max_radius_arcsec * 0.8  # Use 80% of maximum radius
        
        if r_max <= r_min:
            raise ValueError("Image too small for radial analysis")
        
        radius_bins = np.logspace(np.log10(r_min), np.log10(r_max), n_bins + 1)
        
        # Calculate radial profile
        radii, intensities, intensity_errors = self._calculate_radial_profile(
            image, radius_arcsec, radius_bins
        )
        
        if verbose:
            print(f"   Radial bins: {len(radii)}")
            print(f"   Radial range: {radii[0]:.3f} - {radii[-1]:.3f} arcsec")
        
        if len(radii) < 5:
            raise ValueError(f"Insufficient radial bins for analysis (only {len(radii)})")
        
        # Power law fit: I(r) ~ r^(-α)
        alpha, alpha_error, r_squared = self._fit_power_law(radii, intensities)
        
        if verbose:
            print(f"   ✅ Radial profile results:")
            print(f"      I(r) ~ r^(-{alpha:.3f} ± {alpha_error:.3f})")
            print(f"      R² = {r_squared:.4f}")
        
        return alpha, alpha_error, radii, intensities, r_squared, intensity_errors
    
    def _calculate_photometric_center(self, image: np.ndarray, threshold: float) -> Tuple[float, float]:
        """
        Calculate intensity-weighted centroid for accurate center determination.
        
        Parameters:
        -----------
        image : np.ndarray
            2D image
        threshold : float
            Detection threshold for signal mask
            
        Returns:
        --------
        center_x, center_y : float, float
            Photometric center coordinates
        """
        signal_mask = image > threshold
        
        if not np.any(signal_mask):
            raise ValueError("No significant signal detected for centroid calculation")
        
        h, w = image.shape
        y_coords, x_coords = np.ogrid[:h, :w]
        
        signal_data = np.where(signal_mask, image, 0)
        total_signal = np.sum(signal_data)
        
        center_x = np.sum(x_coords * signal_data) / total_signal
        center_y = np.sum(y_coords * signal_data) / total_signal
        
        return center_x, center_y
    
    def _calculate_radial_profile(self, 
                                image: np.ndarray, 
                                radius_arcsec: np.ndarray, 
                                radius_bins: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate radial intensity profile with error propagation.
        
        Parameters:
        -----------
        image : np.ndarray
            2D image
        radius_arcsec : np.ndarray
            Radius grid in arcsec
        radius_bins : np.ndarray
            Radial bin edges
            
        Returns:
        --------
        radii : np.ndarray
            Bin center radii
        intensities : np.ndarray
            Mean intensities
        intensity_errors : np.ndarray
            Standard errors
        """
        radii = []
        intensities = []
        intensity_errors = []
        
        for i in range(len(radius_bins) - 1):
            r_inner, r_outer = radius_bins[i], radius_bins[i + 1]
            r_center = np.sqrt(r_inner * r_outer)  # Geometric mean for log spacing
            
            # Annulus mask
            annulus_mask = (radius_arcsec >= r_inner) & (radius_arcsec < r_outer)
            
            if np.sum(annulus_mask) > 3:  # Minimum pixels for statistics
                annulus_data = image[annulus_mask]
                
                # Robust statistics
                mean_intensity = np.mean(annulus_data)
                std_intensity = np.std(annulus_data) / np.sqrt(len(annulus_data))
                
                radii.append(r_center)
                intensities.append(mean_intensity)
                intensity_errors.append(std_intensity)
        
        return np.array(radii), np.array(intensities), np.array(intensity_errors)
    
    def _fit_power_law(self, radii: np.ndarray, intensities: np.ndarray) -> Tuple[float, float, float]:
        """
        Fit power law to radial profile: I(r) ~ r^(-α)
        
        Parameters:
        -----------
        radii : np.ndarray
            Radial coordinates
        intensities : np.ndarray
            Intensity values
            
        Returns:
        --------
        alpha : float
            Power law exponent
        alpha_error : float
            Uncertainty in exponent
        r_squared : float
            Quality of fit
        """
        # Use only positive intensities and radii
        valid = (intensities > 0) & (radii > 0)
        
        if np.sum(valid) < 4:
            raise ValueError("Insufficient positive intensities for power law fit")
        
        r_fit = radii[valid]
        I_fit = intensities[valid]
        
        # Linear regression in log-log space: log(I) = log(I0) - α * log(r)
        log_r = np.log10(r_fit)
        log_I = np.log10(I_fit)
        
        coeffs, cov = np.polyfit(log_r, log_I, 1, cov=True)
        alpha = -coeffs[0]  # Negative of slope
        alpha_error = np.sqrt(cov[0, 0])
        
        # Calculate R-squared
        predicted_log_I = np.polyval(coeffs, log_r)
        ss_res = np.sum((log_I - predicted_log_I)**2)
        ss_tot = np.sum((log_I - np.mean(log_I))**2)
        r_squared = 1 - (ss_res / ss_tot)
        
        return alpha, alpha_error, r_squared
    
    def scientific_interpretation(self, 
                                df_measured: float, 
                                df_error: float, 
                                alpha: float, 
                                alpha_error: float,
                                verbose: bool = True) -> Dict[str, Any]:
        """
        Comprehensive scientific interpretation of fractal analysis results.
        
        Parameters:
        -----------
        df_measured : float
            Measured fractal dimension
        df_error : float
            Uncertainty in fractal dimension
        alpha : float
            Radial profile exponent
        alpha_error : float
            Uncertainty in radial exponent
        verbose : bool
            Print detailed interpretation
            
        Returns:
        --------
        results : dict
            Comprehensive analysis results and interpretation
        """
        if verbose:
            print("\n" + "="*70)
            print("🔬 ADVANCED SCIENTIFIC INTERPRETATION")
            print("="*70)
        
        # Calculate derived parameters
        delta_derived = 3 - df_measured
        df_error_pct = abs(df_measured - self.df_theoretical) / self.df_theoretical * 100
        delta_error_pct = abs(delta_derived - self.delta_theoretical) / self.delta_theoretical * 100
        
        if verbose:
            print("📊 MEASURED PARAMETERS:")
            print(f"   Fractal dimension: df = {df_measured:.3f} ± {df_error:.3f}")
            print(f"   Radial slope: α = {alpha:.3f} ± {alpha_error:.3f}")
            print(f"   Derived δp = 3 - df = {delta_derived:.3f}")
            
            print("\n🎯 MFSU THEORETICAL COMPARISON:")
            print(f"   MFSU predicted df: {self.df_theoretical:.3f}")
            print(f"   MFSU predicted δp: {self.delta_theoretical:.3f}")
            print(f"   Relative error df: {df_error_pct:.1f}%")
            print(f"   Relative error δp: {delta_error_pct:.1f}%")
        
        # Classify fractal dimension
        df_class, df_desc = self._classify_fractal_dimension(df_measured)
        
        # Classify radial profile
        alpha_class, alpha_desc = self._classify_radial_profile(alpha)
        
        # MFSU agreement assessment
        df_sigma = abs(df_measured - self.df_theoretical) / df_error if df_error > 0 else np.inf
        mfsu_agreement, mfsu_status = self._assess_mfsu_agreement(df_sigma)
        
        if verbose:
            print("\n🌌 ASTRONOMICAL INTERPRETATION:")
            print(f"   df = {df_measured:.3f}: {df_class}")
            print(f"   → {df_desc}")
            print(f"   α = {alpha:.3f}: {alpha_class}")
            print(f"   → {alpha_desc}")
            
            print("\n🔍 MFSU FRAMEWORK ASSESSMENT:")
            print(f"   Statistical significance: {df_sigma:.1f}σ deviation")
            print(f"   → {mfsu_agreement}")
            print(f"   → {mfsu_status}")
        
        # Compile results
        results = {
            'df_measured': df_measured,
            'df_error': df_error,
            'alpha': alpha,
            'alpha_error': alpha_error,
            'delta_derived': delta_derived,
            'df_error_pct': df_error_pct,
            'delta_error_pct': delta_error_pct,
            'df_sigma': df_sigma,
            'mfsu_agreement': mfsu_agreement,
            'mfsu_status': mfsu_status,
            'df_class': df_class,
            'df_description': df_desc,
            'alpha_class': alpha_class,
            'alpha_description': alpha_desc
        }
        
        return results
    
    def _classify_fractal_dimension(self, df: float) -> Tuple[str, str]:
        """Classify fractal dimension for astronomical interpretation."""
        if df < 1.4:
            return "Simple structure", "Smooth, regular morphology - possibly inactive nucleus"
        elif 1.4 <= df < 1.7:
            return "Moderate complexity", "Typical active comet with basic coma structure"
        elif 1.7 <= df < 2.1:
            return "Complex structure", "Multi-component activity with jets/fans"
        else:
            return "Highly complex", "Unusual morphology requiring detailed investigation"
    
    def _classify_radial_profile(self, alpha: float) -> Tuple[str, str]:
        """Classify radial profile for astronomical interpretation."""
        if alpha < 1.0:
            return "Shallow profile", "Extended, diffuse coma - high gas/dust ratio"
        elif 1.0 <= alpha < 1.5:
            return "Moderate profile", "Typical comet brightness distribution"
        elif 1.5 <= alpha < 2.5:
            return "Steep profile", "Concentrated activity - dusty coma"
        else:
            return "Very steep", "Highly concentrated - possible point source dominance"
    
    def _assess_mfsu_agreement(self, sigma: float) -> Tuple[str, str]:
        """Assess agreement with MFSU theoretical predictions."""
        if sigma < 1:
            return "Excellent agreement", "MFSU prediction strongly validated"
        elif sigma < 2:
            return "Good agreement", "MFSU prediction supported"
        elif sigma < 3:
            return "Moderate agreement", "MFSU prediction partially supported"
        else:
            return "Poor agreement", "MFSU prediction not supported by this object"

class MFSUComparator:
    """
    Specialized class for comparing results with MFSU theoretical framework.
    """
    
    def __init__(self):
        self.df_mfsu = 2.079
        self.delta_mfsu = 0.921
        
    def validate_parameters(self, df_measured: float, df_error: float) -> Dict[str, Any]:
        """
        Validate measured parameters against MFSU predictions.
        
        Parameters:
        -----------
        df_measured : float
            Measured fractal dimension
        df_error : float
            Uncertainty in measurement
            
        Returns:
        --------
        validation : dict
            Validation results and statistics
        """
        # Statistical significance
        sigma_deviation = abs(df_measured - self.df_mfsu) / df_error if df_error > 0 else np.inf
        
        # Relative error
        relative_error = abs(df_measured - self.df_mfsu) / self.df_mfsu * 100
        
        # Validation status
        if sigma_deviation < 1:
            status = "VALIDATED"
            confidence = "High"
        elif sigma_deviation < 2:
            status = "SUPPORTED"
            confidence = "Good"
        elif sigma_deviation < 3:
            status = "PARTIAL"
            confidence = "Moderate"
        else:
            status = "INCONSISTENT"
            confidence = "Low"
        
        return {
            'sigma_deviation': sigma_deviation,
            'relative_error': relative_error,
            'status': status,
            'confidence': confidence,
            'df_theoretical': self.df_mfsu,
            'delta_theoretical': self.delta_mfsu
        }
    
    def generate_predictions(self, df_measured: float) -> Dict[str, float]:
        """
        Generate MFSU-based predictions from measured fractal dimension.
        
        Parameters:
        -----------
        df_measured : float
            Measured fractal dimension
            
        Returns:
        --------
        predictions : dict
            MFSU predictions based on measured df
        """
        delta_predicted = 3 - df_measured
        
        # Power spectrum predictions
        power_spectrum_exponent = -(3 - df_measured)
        
        # Correlation function predictions
        correlation_exponent = -df_measured
        
        return {
            'delta_predicted': delta_predicted,
            'power_spectrum_exponent': power_spectrum_exponent,
            'correlation_exponent': correlation_exponent,
            'scaling_relation': f"δ = 3 - df = {delta_predicted:.3f}"
        }
