#!/usr/bin/env python3
"""
MFSU Comet Analysis - Preprocessing Module
==========================================

Scientific preprocessing of astronomical images for fractal analysis.
Implements background subtraction, noise estimation, and detection thresholding.

Author: Miguel Ángel Franco León & Claude
Date: September 2025
"""

import numpy as np
from PIL import Image
from typing import Tuple, Dict, Optional, Union, Any
import warnings

class AstronomicalPreprocessor:
    """
    Scientific preprocessing for astronomical images.
    
    Implements standard astronomical data reduction techniques
    optimized for fractal analysis of extended objects like comets.
    """
    
    def __init__(self, 
                 default_pixel_scale: float = 0.12,
                 detection_sigma: float = 3.0):
        """
        Initialize astronomical preprocessor.
        
        Parameters:
        -----------
        default_pixel_scale : float
            Default pixel scale in arcsec/pixel for scale estimation
        detection_sigma : float
            Detection threshold in units of noise sigma
        """
        self.default_pixel_scale = default_pixel_scale
        self.detection_sigma = detection_sigma
        
        # JWST-specific parameters
        self.jwst_bands = {
            'F070W': {'wavelength': 0.7, 'typical_scale': 0.032},
            'F090W': {'wavelength': 0.9, 'typical_scale': 0.032}, 
            'F115W': {'wavelength': 1.15, 'typical_scale': 0.032},
            'F150W': {'wavelength': 1.5, 'typical_scale': 0.032},
            'F200W': {'wavelength': 2.0, 'typical_scale': 0.032},
            'IFU': {'wavelength': 1.5, 'typical_scale': 0.12}  # Default for IFU
        }
    
    def load_image(self, 
                   image_path: Optional[str] = None,
                   image_data: Optional[np.ndarray] = None,
                   verbose: bool = True) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Load astronomical image with metadata extraction.
        
        Parameters:
        -----------
        image_path : str, optional
            Path to image file
        image_data : np.ndarray, optional
            Direct numpy array input
        verbose : bool
            Print loading information
            
        Returns:
        --------
        image : np.ndarray
            2D image array (float64)
        metadata : dict
            Image metadata and properties
        """
        if verbose:
            print("\n🔭 Loading astronomical image...")
        
        if image_data is not None:
            # Direct numpy array input
            image = image_data.astype(np.float64)
            source = "numpy array"
        elif image_path is not None:
            # Load from file
            try:
                img = Image.open(image_path)
                image = np.array(img).astype(np.float64)
                source = image_path
            except Exception as e:
                raise ValueError(f"Failed to load image from {image_path}: {e}")
        else:
            # Create high-fidelity JWST representation
            if verbose:
                print("   📊 Creating high-fidelity JWST representation...")
            image = self._create_jwst_representation()
            source = "synthetic JWST representation"
        
        # Handle color images
        if image.ndim == 3:
            if verbose:
                print("   Converting RGB to grayscale...")
            image = np.mean(image, axis=2)
        
        # Extract metadata
        metadata = self._extract_metadata(image, source)
        
        if verbose:
            print(f"   ✅ Image loaded from: {source}")
            print(f"   Dimensions: {metadata['shape']}")
            print(f"   Intensity range: [{metadata['min_value']:.1f}, {metadata['max_value']:.1f}]")
            print(f"   Estimated scale: {metadata['estimated_pixel_scale']:.3f} arcsec/pixel")
        
        return image, metadata
    
    def _create_jwst_representation(self) -> np.ndarray:
        """
        Create high-fidelity representation matching JWST characteristics.
        
        Returns:
        --------
        image : np.ndarray
            Synthetic JWST-like comet image
        """
        size = 512  # High resolution for detailed analysis
        
        # Physical coordinate system (RA/Dec offsets in arcsec)
        x = np.linspace(-3, 3, size)
        y = np.linspace(-2, 2, size)
        X, Y = np.meshgrid(x, y)
        R = np.sqrt(X**2 + Y**2)
        
        # Multi-component comet model
        
        # 1. Central nucleus (point source with seeing)
        nucleus = 1000 * np.exp(-R**2 / (0.08**2))
        
        # 2. Inner coma with observed asymmetry
        angle_offset = np.arctan2(Y, X)
        asymmetry = 1.0 + 0.4 * np.cos(angle_offset - np.pi/3)
        inner_coma = 200 * np.exp(-R**2 / (0.4**2)) * asymmetry
        
        # 3. Extended envelope
        outer_coma = 80 * np.exp(-R / 1.2)
        
        # 4. Directional jets/fans
        jet1_angle = np.pi/4
        jet1_width = 0.25
        jet1_mask = (np.abs(angle_offset - jet1_angle) < jet1_width) & (R > 0.15) & (R < 2.0)
        jet1 = 100 * np.exp(-R / 1.0) * jet1_mask.astype(float)
        
        jet2_angle = -2*np.pi/3
        jet2_width = 0.3
        jet2_mask = (np.abs(angle_offset - jet2_angle) < jet2_width) & (R > 0.2) & (R < 1.8)
        jet2 = 60 * np.exp(-R / 0.8) * jet2_mask.astype(float)
        
        # 5. Extended tail structure
        tail_mask = (X < -0.5) & (np.abs(Y) < 0.8)
        tail = 40 * np.exp(-np.abs(X + 1.0) / 0.6) * tail_mask.astype(float)
        
        # Combine components
        comet_total = nucleus + inner_coma + outer_coma + jet1 + jet2 + tail
        
        # Add realistic JWST noise
        # Shot noise (Poisson statistics)
        shot_noise = np.random.poisson(np.maximum(comet_total/10, 1)) * 10 - comet_total
        
        # Read noise (Gaussian)
        read_noise = np.random.normal(0, 3, comet_total.shape)
        
        # Background level
        background = 15
        
        # Final image with noise floor
        image_final = np.maximum(comet_total + shot_noise + read_noise + background, 
                               background * 0.1)
        
        return image_final
    
    def _extract_metadata(self, image: np.ndarray, source: str) -> Dict[str, Any]:
        """
        Extract comprehensive metadata from image.
        
        Parameters:
        -----------
        image : np.ndarray
            2D image array
        source : str
            Source description
            
        Returns:
        --------
        metadata : dict
            Comprehensive image metadata
        """
        # Basic properties
        metadata = {
            'source': source,
            'shape': image.shape,
            'min_value': np.min(image),
            'max_value': np.max(image),
            'mean_value': np.mean(image),
            'std_value': np.std(image),
            'total_flux': np.sum(image),
            'estimated_pixel_scale': self._estimate_pixel_scale(image)
        }
        
        # Find peak and center of mass
        peak_idx = np.unravel_index(np.argmax(image), image.shape)
        metadata['peak_position'] = peak_idx
        metadata['peak_intensity'] = image[peak_idx]
        
        # Center of mass calculation
        h, w = image.shape
        y_coords, x_coords = np.ogrid[:h, :w]
        
        # Use positive values only for center of mass
        positive_image = np.maximum(image - np.percentile(image, 10), 0)
        total_positive = np.sum(positive_image)
        
        if total_positive > 0:
            center_x = np.sum(x_coords * positive_image) / total_positive
            center_y = np.sum(y_coords * positive_image) / total_positive
            metadata['center_of_mass'] = (center_x, center_y)
        else:
            metadata['center_of_mass'] = (w/2, h/2)
        
        # Dynamic range and quality metrics
        metadata['dynamic_range'] = metadata['max_value'] / metadata['mean_value']
        metadata['snr_estimate'] = metadata['peak_intensity'] / metadata['std_value']
        
        return metadata
    
    def _estimate_pixel_scale(self, image: np.ndarray) -> float:
        """
        Estimate pixel scale from image characteristics.
        
        Parameters:
        -----------
        image : np.ndarray
            2D image array
            
        Returns:
        --------
        pixel_scale : float
            Estimated pixel scale in arcsec/pixel
        """
        # Find the brightest source (presumably the nucleus)
        peak_idx = np.unravel_index(np.argmax(image), image.shape)
        center_y, center_x = peak_idx
        
        # Extract region around peak
        region_size = 40
        y_start = max(0, center_y - region_size//2)
        y_end = min(image.shape[0], center_y + region_size//2)
        x_start = max(0, center_x - region_size//2)  
        x_end = min(image.shape[1], center_x + region_size//2)
        
        center_region = image[y_start:y_end, x_start:x_end]
        
        if center_region.size == 0:
            return self.default_pixel_scale
        
        # Find FWHM of central source
        peak = np.max(center_region)
        background = np.median(center_region)
        half_max = background + (peak - background) / 2
        
        # Count pixels above half maximum
        above_half = np.sum(center_region > half_max)
        
        if above_half > 0:
            fwhm_pixels = np.sqrt(above_half / np.pi) * 2
            
            # Typical comet nucleus angular size ~0.15 arcsec
            # For point sources, use seeing-limited size ~0.1-0.2 arcsec
            estimated_scale = 0.15 / fwhm_pixels if fwhm_pixels > 0 else self.default_pixel_scale
            
            # Apply reasonable bounds based on instrument capabilities
            return max(0.03, min(0.5, estimated_scale))
        else:
            return self.default_pixel_scale
    
    def preprocess_image(self, 
                        image: np.ndarray,
                        background_method: str = 'corners',
                        noise_method: str = 'mad',
                        verbose: bool = True) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Scientific preprocessing of astronomical image.
        
        Parameters:
        -----------
        image : np.ndarray
            Raw 2D image
        background_method : str
            Background estimation method ('corners', 'sigma_clip', 'percentile')
        noise_method : str
            Noise estimation method ('mad', 'std', 'robust')
        verbose : bool
            Print processing information
            
        Returns:
        --------
        processed_image : np.ndarray
            Background-subtracted image
        preprocessing_data : dict
            Processing metadata and derived products
        """
        if verbose:
            print("\n🔬 Scientific image preprocessing...")
        
        # 1. Background estimation and subtraction
        background = self._estimate_background(image, method=background_method)
        image_bg_sub = image - background
        
        if verbose:
            print(f"   Background level ({background_method}): {background:.3f}")
            print(f"   Background-subtracted range: [{np.min(image_bg_sub):.1f}, {np.max(image_bg_sub):.1f}]")
        
        # 2. Noise estimation
        noise_std = self._estimate_noise(image, background, method=noise_method)
        
        if verbose:
            print(f"   Estimated noise ({noise_method}): {noise_std:.3f}")
        
        # 3. Signal-to-noise ratio map
        snr_map = np.divide(image_bg_sub, noise_std, 
                           out=np.zeros_like(image_bg_sub), 
                           where=noise_std != 0)
        
        # 4. Detection threshold and mask
        detection_threshold = self.detection_sigma * noise_std
        detection_mask = image_bg_sub > detection_threshold
        
        # 5. Quality metrics
        detected_pixels = np.sum(detection_mask)
        detection_fraction = detected_pixels / image.size
        max_snr = np.max(snr_map)
        
        if verbose:
            print(f"   Detection threshold ({self.detection_sigma}σ): {detection_threshold:.3f}")
            print(f"   Detected pixels: {detected_pixels} ({detection_fraction*100:.2f}%)")
            print(f"   Maximum S/N: {max_snr:.1f}")
        
        # Compile preprocessing data
        preprocessing_data = {
            'background': background,
            'background_method': background_method,
            'noise_std': noise_std,
            'noise_method': noise_method,
            'detection_threshold': detection_threshold,
            'detection_sigma': self.detection_sigma,
            'snr_map': snr_map,
            'detection_mask': detection_mask,
            'detected_pixels': detected_pixels,
            'detection_fraction': detection_fraction,
            'max_snr': max_snr,
            'processing_quality': self._assess_processing_quality(
                detection_fraction, max_snr, noise_std
            )
        }
        
        return image_bg_sub, preprocessing_data
    
    def _estimate_background(self, image: np.ndarray, method: str = 'corners') -> float:
        """
        Estimate background level using various methods.
        
        Parameters:
        -----------
        image : np.ndarray
            2D image array
        method : str
            Background estimation method
            
        Returns:
        --------
        background : float
            Estimated background level
        """
        if method == 'corners':
            # Use image corners (avoiding central object)
            h, w = image.shape
            corner_size = min(h, w) // 8
            
            corners = [
                image[:corner_size, :corner_size],              # top-left
                image[:corner_size, -corner_size:],             # top-right
                image[-corner_size:, :corner_size],             # bottom-left
                image[-corner_size:, -corner_size:]             # bottom-right
            ]
            
            corner_data = np.concatenate([c.flatten() for c in corners])
            background = np.median(corner_data)
            
        elif method == 'percentile':
            # Use low percentile of entire image
            background = np.percentile(image, 5)
            
        elif method == 'sigma_clip':
            # Iterative sigma clipping
            data = image.flatten()
            for _ in range(3):  # 3 iterations
                mean = np.mean(data)
                std = np.std(data)
                mask = np.abs(data - mean) < 3 * std
                data = data[mask]
                if len(data) < 100:  # Safety check
                    break
            background = np.mean(data)
            
        else:
            raise ValueError(f"Unknown background method: {method}")
        
        return background
    
    def _estimate_noise(self, image: np.ndarray, background: float, method: str = 'mad') -> float:
        """
        Estimate noise level using robust statistics.
        
        Parameters:
        -----------
        image : np.ndarray
            2D image array
        background : float
            Background level
        method : str
            Noise estimation method
            
        Returns:
        --------
        noise_std : float
            Estimated noise standard deviation
        """
        if method == 'mad':
            # Median Absolute Deviation (robust to outliers)
            # Use background-subtracted data
            bg_sub = image - background
            
            # Use corners for noise estimation (avoiding signal)
            h, w = image.shape
            corner_size = min(h, w) // 8
            
            corners = [
                bg_sub[:corner_size, :corner_size],
                bg_sub[:corner_size, -corner_size:],
                bg_sub[-corner_size:, :corner_size],
                bg_sub[-corner_size:, -corner_size:]
            ]
            
            corner_data = np.concatenate([c.flatten() for c in corners])
            
            # MAD estimator: σ ≈ 1.4826 × MAD
            mad = np.median(np.abs(corner_data - np.median(corner_data)))
            noise_std = 1.4826 * mad
            
        elif method == 'std':
            # Standard deviation of corners
            h, w = image.shape
            corner_size = min(h, w) // 8
            
            corners = [
                image[:corner_size, :corner_size],
                image[:corner_size, -corner_size:],
                image[-corner_size:, :corner_size],
                image[-corner_size:, -corner_size:]
            ]
            
            corner_data = np.concatenate([c.flatten() for c in corners])
            noise_std = np.std(corner_data)
            
        elif method == 'robust':
            # Robust estimator using percentiles
            bg_sub = image - background
            
            # Use central 80% of distribution for noise
            p10 = np.percentile(bg_sub, 10)
            p90 = np.percentile(bg_sub, 90)
            
            # Convert percentile range to standard deviation
            # For normal distribution: P90 - P10 ≈ 2.56σ
            noise_std = (p90 - p10) / 2.56
            
        else:
            raise ValueError(f"Unknown noise method: {method}")
        
        # Ensure positive noise estimate
        return max(noise_std, 1e-6)
    
    def _assess_processing_quality(self, 
                                 detection_fraction: float, 
                                 max_snr: float, 
                                 noise_std: float) -> Dict[str, Any]:
        """
        Assess quality of preprocessing for fractal analysis.
        
        Parameters:
        -----------
        detection_fraction : float
            Fraction of pixels above detection threshold
        max_snr : float
            Maximum signal-to-noise ratio
        noise_std : float
            Noise standard deviation
            
        Returns:
        --------
        quality : dict
            Quality assessment metrics
        """
        # Quality criteria for fractal analysis
        quality_flags = []
        
        # 1. Detection fraction check
        if detection_fraction < 0.01:
            quality_flags.append("Low detection fraction (<1%)")
        elif detection_fraction > 0.5:
            quality_flags.append("Very high detection fraction (>50%)")
        
        # 2. Signal-to-noise check
        if max_snr < 10:
            quality_flags.append("Low S/N (<10)")
        elif max_snr > 1000:
            quality_flags.append("Very high S/N (>1000)")
        
        # 3. Noise level check
        if noise_std < 1:
            quality_flags.append("Very low noise (<1)")
        elif noise_std > 100:
            quality_flags.append("High noise level (>100)")
        
        # Overall quality assessment
        if len(quality_flags) == 0:
            overall_quality = "Excellent"
            quality_score = 5
        elif len(quality_flags) <= 2:
            overall_quality = "Good"
            quality_score = 4
        elif len(quality_flags) <= 4:
            overall_quality = "Fair"
            quality_score = 3
        else:
            overall_quality = "Poor"
            quality_score = 2
        
        return {
            'overall_quality': overall_quality,
            'quality_score': quality_score,
            'quality_flags': quality_flags,
            'detection_fraction': detection_fraction,
            'max_snr': max_snr,
            'suitable_for_fractal_analysis': quality_score >= 3
        }
    
    def create_quality_report(self, 
                            image: np.ndarray, 
                            processed_image: np.ndarray,
                            preprocessing_data: Dict[str, Any],
                            verbose: bool = True) -> Dict[str, Any]:
        """
        Generate comprehensive quality report for preprocessing.
        
        Parameters:
        -----------
        image : np.ndarray
            Original image
        processed_image : np.ndarray
            Processed image
        preprocessing_data : dict
            Preprocessing metadata
        verbose : bool
            Print detailed report
            
        Returns:
        --------
        report : dict
            Comprehensive quality report
        """
        quality = preprocessing_data['processing_quality']
        
        if verbose:
            print("\n📋 PREPROCESSING QUALITY REPORT")
            print("=" * 50)
            print(f"Overall Quality: {quality['overall_quality']} ({quality['quality_score']}/5)")
            print(f"Suitable for Fractal Analysis: {'Yes' if quality['suitable_for_fractal_analysis'] else 'No'}")
            
            if quality['quality_flags']:
                print("\nQuality Flags:")
                for flag in quality['quality_flags']:
                    print(f"  ⚠️  {flag}")
            else:
                print("\n✅ No quality issues detected")
            
            print(f"\nDetection Statistics:")
            print(f"  Detection threshold: {preprocessing_data['detection_threshold']:.3f}")
            print(f"  Detected pixels: {preprocessing_data['detected_pixels']}")
            print(f"  Detection fraction: {preprocessing_data['detection_fraction']*100:.2f}%")
            print(f"  Maximum S/N: {preprocessing_data['max_snr']:.1f}")
            
            print(f"\nNoise Statistics:")
            print(f"  Background level: {preprocessing_data['background']:.3f}")
            print(f"  Noise std: {preprocessing_data['noise_std']:.3f}")
            print(f"  Dynamic range: {np.max(processed_image)/preprocessing_data['noise_std']:.1f}")
        
        # Additional statistics
        report = {
            'preprocessing_quality': quality,
            'statistics': {
                'original_range': (np.min(image), np.max(image)),
                'processed_range': (np.min(processed_image), np.max(processed_image)),
                'background_level': preprocessing_data['background'],
                'noise_level': preprocessing_data['noise_std'],
                'detection_threshold': preprocessing_data['detection_threshold'],
                'detected_pixels': preprocessing_data['detected_pixels'],
                'detection_fraction': preprocessing_data['detection_fraction'],
                'max_snr': preprocessing_data['max_snr'],
                'dynamic_range': np.max(processed_image) / preprocessing_data['noise_std']
            },
            'recommendations': self._generate_recommendations(quality)
        }
        
        return report
    
    def _generate_recommendations(self, quality: Dict[str, Any]) -> List[str]:
        """Generate processing recommendations based on quality assessment."""
        recommendations = []
        
        if quality['quality_score'] < 3:
            recommendations.append("Consider different preprocessing parameters")
        
        for flag in quality['quality_flags']:
            if "Low detection fraction" in flag:
                recommendations.append("Lower detection threshold or check for faint sources")
            elif "Very high detection fraction" in flag:
                recommendations.append("Increase detection threshold to avoid noise contamination")
            elif "Low S/N" in flag:
                recommendations.append("Consider image stacking or longer exposures")
            elif "High noise level" in flag:
                recommendations.append("Apply noise reduction or check calibration")
        
        if not recommendations:
            recommendations.append("Image quality is suitable for fractal analysis")
        
        return recommendations

class DataLoader:
    """
    Specialized loader for various astronomical data formats.
    """
    
    @staticmethod
    def load_fits(filepath: str, hdu: int = 0) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Load FITS file (requires astropy).
        
        Parameters:
        -----------
        filepath : str
            Path to FITS file
        hdu : int
            HDU number to load
            
        Returns:
        --------
        image : np.ndarray
            2D image data
        header : dict
            FITS header metadata
        """
        try:
            from astropy.io import fits
            
            with fits.open(filepath) as hdul:
                image = hdul[hdu].data.astype(np.float64)
                header = dict(hdul[hdu].header)
                
            return image, header
            
        except ImportError:
            raise ImportError("astropy required for FITS support. Install with: pip install astropy")
    
    @staticmethod
    def load_jwst_pipeline(filepath: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Load JWST pipeline products with full metadata.
        
        Parameters:
        -----------
        filepath : str
            Path to JWST file
            
        Returns:
        --------
        image : np.ndarray
            Science image
        metadata : dict
            Complete JWST metadata
        """
        try:
            from astropy.io import fits
            
            with fits.open(filepath) as hdul:
                # Science data
                image = hdul['SCI'].data.astype(np.float64)
                
                # Extract JWST-specific metadata
                header = hdul[0].header
                metadata = {
                    'instrument': header.get('INSTRUME', 'UNKNOWN'),
                    'filter': header.get('FILTER', 'UNKNOWN'),
                    'exposure_time': header.get('EXPTIME', 0),
                    'pixel_scale': header.get('PIXAR_A2', 0.032),  # arcsec²/pixel
                    'observation_date': header.get('DATE-OBS', 'UNKNOWN'),
                    'target': header.get('TARGNAME', 'UNKNOWN'),
                    'full_header': dict(header)
                }
                
                # Calculate pixel scale in arcsec/pixel
                if 'PIXAR_A2' in header:
                    metadata['pixel_scale_linear'] = np.sqrt(header['PIXAR_A2'])
                else:
                    metadata['pixel_scale_linear'] = 0.032  # Default JWST
                
            return image, metadata
            
        except ImportError:
            raise ImportError("astropy required for JWST support. Install with: pip install astropy")
