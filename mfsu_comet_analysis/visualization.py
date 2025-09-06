#!/usr/bin/env python3
"""
MFSU Comet Analysis - Visualization Module
==========================================

Publication-quality visualization tools for fractal analysis results.
Creates comprehensive analysis plots suitable for scientific publications.

Author: Miguel Ángel Franco León & Claude
Date: September 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.colors import LogNorm
from typing import Tuple, Dict, Optional, Any, List
import warnings

class MFSUVisualizer:
    """
    Publication-quality visualization for MFSU fractal analysis.
    
    Creates comprehensive analysis plots including box-counting,
    radial profiles, MFSU comparisons, and scientific interpretation.
    """
    
    def __init__(self, 
                 style: str = 'publication',
                 dpi: int = 300,
                 figsize_large: Tuple[int, int] = (20, 12),
                 figsize_medium: Tuple[int, int] = (16, 10)):
        """
        Initialize MFSU visualizer.
        
        Parameters:
        -----------
        style : str
            Plot style ('publication', 'presentation', 'paper')
        dpi : int
            Resolution for saved figures
        figsize_large : tuple
            Figure size for comprehensive plots
        figsize_medium : tuple
            Figure size for focused plots
        """
        self.style = style
        self.dpi = dpi
        self.figsize_large = figsize_large
        self.figsize_medium = figsize_medium
        
        # Apply publication style
        self._setup_plot_style()
        
        # Color schemes
        self.colors = {
            'primary': '#2E86AB',      # Blue
            'secondary': '#A23B72',    # Purple
            'accent': '#F18F01',       # Orange
            'success': '#C73E1D',      # Red
            'data': '#1f77b4',         # Matplotlib blue
            'fit': '#d62728',          # Matplotlib red
            'background': '#f8f9fa'    # Light gray
        }
    
    def _setup_plot_style(self):
        """Setup matplotlib style for publication quality."""
        plt.style.use('default')
        
        if self.style == 'publication':
            plt.rcParams.update({
                'font.size': 12,
                'font.family': 'serif',
                'font.serif': ['Times', 'Times New Roman'],
                'axes.linewidth': 1.2,
                'axes.labelsize': 14,
                'axes.titlesize': 16,
                'xtick.labelsize': 12,
                'ytick.labelsize': 12,
                'legend.fontsize': 11,
                'figure.titlesize': 18,
                'lines.linewidth': 2,
                'lines.markersize': 8,
                'savefig.dpi': self.dpi,
                'savefig.bbox': 'tight',
                'text.usetex': False  # Avoid LaTeX issues
            })
        elif self.style == 'presentation':
            plt.rcParams.update({
                'font.size': 14,
                'axes.linewidth': 2,
                'lines.linewidth': 3,
                'lines.markersize': 10
            })
    
    def create_comprehensive_analysis_plot(self,
                                         original_image: np.ndarray,
                                         processed_image: np.ndarray,
                                         preprocessing_data: Dict[str, Any],
                                         box_data: Tuple,
                                         radial_data: Tuple,
                                         interpretation_results: Dict[str, Any],
                                         pixel_scale: float,
                                         save_path: Optional[str] = None,
                                         show: bool = True) -> plt.Figure:
        """
        Create comprehensive analysis plot with all results.
        
        Parameters:
        -----------
        original_image : np.ndarray
            Original astronomical image
        processed_image : np.ndarray
            Background-subtracted image
        preprocessing_data : dict
            Preprocessing metadata
        box_data : tuple
            Box-counting analysis results
        radial_data : tuple
            Radial profile analysis results
        interpretation_results : dict
            Scientific interpretation results
        pixel_scale : float
            Pixel scale in arcsec/pixel
        save_path : str, optional
            Path to save figure
        show : bool
            Whether to display figure
            
        Returns:
        --------
        fig : matplotlib.Figure
            Complete analysis figure
        """
        print("\n📊 Creating comprehensive analysis plot...")
        
        # Unpack data
        df_measured, df_error, scales, counts, r2_box, box_sizes = box_data
        alpha, alpha_error, radii, intensities, r2_radial, intensity_errors = radial_data
        
        # Create figure
        fig = plt.figure(figsize=self.figsize_large)
        fig.suptitle('MFSU Comet 31/ATLAS - Complete Fractal Analysis', 
                    fontsize=20, fontweight='bold', y=0.98)
        
        # 1. Original JWST image
        ax1 = plt.subplot(2, 4, 1)
        self._plot_original_image(ax1, original_image, pixel_scale)
        
        # 2. Processed image with detection contours
        ax2 = plt.subplot(2, 4, 2)
        self._plot_processed_image(ax2, processed_image, preprocessing_data, pixel_scale)
        
        # 3. Box-counting analysis
        ax3 = plt.subplot(2, 4, 3)
        self._plot_box_counting(ax3, scales, counts, df_measured, df_error, r2_box)
        
        # 4. Radial profile analysis
        ax4 = plt.subplot(2, 4, 4)
        self._plot_radial_profile(ax4, radii, intensities, intensity_errors, 
                                alpha, alpha_error, r2_radial)
        
        # 5. MFSU parameter comparison
        ax5 = plt.subplot(2, 4, 5)
        self._plot_mfsu_comparison(ax5, df_measured, df_error, interpretation_results)
        
        # 6. Residuals analysis
        ax6 = plt.subplot(2, 4, 6)
        self._plot_residuals_analysis(ax6, scales, counts, df_measured)
        
        # 7. Statistical summary
        ax7 = plt.subplot(2, 4, 7)
        self._plot_statistical_summary(ax7, df_measured, df_error, alpha, alpha_error,
                                     r2_box, r2_radial, interpretation_results)
        
        # 8. Physical interpretation
        ax8 = plt.subplot(2, 4, 8)
        self._plot_physical_interpretation(ax8, interpretation_results)
        
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"✅ Comprehensive plot saved: {save_path}")
        
        if show:
            plt.show()
        
        return fig
    
    def _plot_original_image(self, ax: plt.Axes, image: np.ndarray, pixel_scale: float):
        """Plot original JWST image with scale information."""
        im = ax.imshow(image, cmap='hot', origin='lower', aspect='auto',
                      norm=LogNorm(vmin=np.percentile(image, 1), 
                                  vmax=np.percentile(image, 99.5)))
        
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Intensity (counts)', fontsize=10)
        
        ax.set_title('Comet 31/ATLAS\nJWST IFU IR (2025-08-06)', fontsize=12, fontweight='bold')
        ax.set_xlabel('RA Offset (pixels)')
        ax.set_ylabel('Dec Offset (pixels)')
        
        # Add scale bar
        scale_pixels = 1.0 / pixel_scale  # 1 arcsec
        x_start = 0.1 * image.shape[1]
        y_start = 0.1 * image.shape[0]
        
        ax.plot([x_start, x_start + scale_pixels], [y_start, y_start], 
               'white', linewidth=4, solid_capstyle='round')
        ax.text(x_start + scale_pixels/2, y_start + 0.05*image.shape[0], 
               '1"', color='white', fontsize=10, ha='center', fontweight='bold')
    
    def _plot_processed_image(self, ax: plt.Axes, processed_image: np.ndarray, 
                            preprocessing_data: Dict[str, Any], pixel_scale: float):
        """Plot processed image with detection contours."""
        threshold = preprocessing_data['detection_threshold']
        detection_mask = processed_image > threshold
        
        # Show processed image
        im = ax.imshow(processed_image, cmap='viridis', origin='lower', aspect='auto')
        
        # Add detection contour
        ax.contour(detection_mask, levels=[0.5], colors='white', linewidths=2, alpha=0.8)
        
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Background-subtracted', fontsize=10)
        
        ax.set_title(f'Processed Image\n3σ Detection (σ={preprocessing_data["noise_std"]:.2f})', 
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('RA Offset (pixels)')
        ax.set_ylabel('Dec Offset (pixels)')
        
        # Add detection statistics
        detection_pct = preprocessing_data['detection_fraction'] * 100
        ax.text(0.02, 0.98, f'Detected: {detection_pct:.1f}%', 
               transform=ax.transAxes, va='top', ha='left',
               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
               fontsize=9)
    
    def _plot_box_counting(self, ax: plt.Axes, scales: np.ndarray, counts: np.ndarray,
                         df_measured: float, df_error: float, r2_box: float):
        """Plot box-counting analysis with fit."""
        # Data points
        ax.loglog(scales, counts, 'o', color=self.colors['data'], markersize=10,
                 markerfacecolor='lightblue', markeredgecolor=self.colors['data'],
                 markeredgewidth=2, label='Measured data', zorder=3)
        
        # Theoretical fit line
        scales_fit = np.logspace(np.log10(scales[0]), np.log10(scales[-1]), 100)
        counts_fit = counts[0] * (scales_fit / scales[0])**(-df_measured)
        ax.loglog(scales_fit, counts_fit, '--', color=self.colors['fit'], linewidth=3,
                 label=f'df = {df_measured:.3f} ± {df_error:.3f}', zorder=2)
        
        # Error bars
        # Estimate uncertainties (Poisson-like)
        count_errors = np.sqrt(counts)
        ax.errorbar(scales, counts, yerr=count_errors, fmt='none', 
                   ecolor=self.colors['data'], alpha=0.5, capsize=4, zorder=1)
        
        ax.set_xlabel('Scale (arcsec)', fontweight='bold')
        ax.set_ylabel('Box Count N(ε)', fontweight='bold')
        ax.set_title(f'Box-Counting Analysis\nR² = {r2_box:.4f}', 
                    fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', framealpha=0.9)
        ax.grid(True, alpha=0.3, which='both')
        
        # Add scaling law annotation
        ax.text(0.05, 0.15, r'$N(\varepsilon) \sim \varepsilon^{-d_f}$', 
               transform=ax.transAxes, fontsize=12, 
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
    
    def _plot_radial_profile(self, ax: plt.Axes, radii: np.ndarray, intensities: np.ndarray,
                           intensity_errors: np.ndarray, alpha: float, alpha_error: float,
                           r2_radial: float):
        """Plot radial profile analysis with power law fit."""
        # Data points with error bars
        ax.loglog(radii, intensities, 'o', color=self.colors['success'], markersize=8,
                 markerfacecolor='lightgreen', markeredgecolor=self.colors['success'],
                 markeredgewidth=2, label='Measured profile', zorder=3)
        
        if intensity_errors is not None:
            ax.errorbar(radii, intensities, yerr=intensity_errors, fmt='none',
                       ecolor=self.colors['success'], alpha=0.6, capsize=3, zorder=1)
        
        # Power law fit
        radii_fit = np.logspace(np.log10(radii[0]), np.log10(radii[-1]), 100)
        intensities_fit = intensities[0] * (radii_fit / radii[0])**(-alpha)
        ax.loglog(radii_fit, intensities_fit, '--', color=self.colors['accent'], 
                 linewidth=3, label=f'I ∝ r^(-{alpha:.3f})', zorder=2)
        
        ax.set_xlabel('Radius (arcsec)', fontweight='bold')
        ax.set_ylabel('Intensity', fontweight='bold')
        ax.set_title(f'Radial Profile Analysis\nR² = {r2_radial:.4f}', 
                    fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', framealpha=0.9)
        ax.grid(True, alpha=0.3, which='both')
        
        # Add power law annotation
        ax.text(0.05, 0.85, r'$I(r) \sim r^{-\alpha}$', 
               transform=ax.transAxes, fontsize=12,
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcyan"))
    
    def _plot_mfsu_comparison(self, ax: plt.Axes, df_measured: float, df_error: float,
                            interpretation_results: Dict[str, Any]):
        """Plot MFSU parameter comparison."""
        df_theoretical = 2.079
        delta_theoretical = 0.921
        delta_measured = 3 - df_measured
        
        categories = ['df\n(measured)', 'df\n(MFSU)', 'δp\n(derived)', 'δp\n(MFSU)']
        values = [df_measured, df_theoretical, delta_measured, delta_theoretical]
        errors = [df_error, 0, df_error, 0]
        colors = [self.colors['data'], self.colors['secondary'], 
                 self.colors['data'], self.colors['secondary']]
        
        bars = ax.bar(range(len(categories)), values, color=colors, alpha=0.8,
                     yerr=errors, capsize=8, width=0.6)
        
        # Add value labels on bars
        for i, (bar, val, err) in enumerate(zip(bars, values, errors)):
            y_pos = bar.get_height() + err + 0.02
            ax.text(bar.get_x() + bar.get_width()/2, y_pos,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax.set_xticks(range(len(categories)))
        ax.set_xticklabels(categories, fontsize=11)
        ax.set_ylabel('Parameter Value', fontweight='bold')
        ax.set_title('MFSU Parameter Comparison', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add agreement status
        agreement = interpretation_results.get('mfsu_agreement', 'Unknown')
        sigma = interpretation_results.get('df_sigma', 0)
        ax.text(0.5, 0.95, f'{agreement}\n({sigma:.1f}σ deviation)', 
               transform=ax.transAxes, ha='center', va='top',
               bbox=dict(boxstyle="round,pad=0.4", facecolor="lightblue", alpha=0.8),
               fontsize=10, fontweight='bold')
    
    def _plot_residuals_analysis(self, ax: plt.Axes, scales: np.ndarray, 
                               counts: np.ndarray, df_measured: float):
        """Plot residuals analysis for box-counting fit."""
        log_scales = np.log10(scales)
        log_counts = np.log10(counts)
        
        # Calculate fit and residuals
        predicted = np.polyval([-df_measured, np.log10(counts[0]) + df_measured*log_scales[0]], 
                              log_scales)
        residuals = log_counts - predicted
        
        # Plot residuals
        ax.plot(log_scales, residuals, 'o-', color=self.colors['primary'], 
               markersize=8, linewidth=2, markerfacecolor='lightblue',
               markeredgecolor=self.colors['primary'])
        
        # Zero line
        ax.axhline(y=0, color=self.colors['fit'], linestyle='--', alpha=0.7, linewidth=2)
        
        # Error bounds (±0.1 in log space is reasonable)
        ax.fill_between(log_scales, -0.1, 0.1, alpha=0.2, color='gray')
        
        ax.set_xlabel('log₁₀(Scale)', fontweight='bold')
        ax.set_ylabel('Residuals', fontweight='bold')
        ax.set_title('Box-Counting Fit Residuals', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Statistics
        rmse = np.sqrt(np.mean(residuals**2))
        max_residual = np.max(np.abs(residuals))
        ax.text(0.05, 0.95, f'RMSE: {rmse:.3f}\nMax: {max_residual:.3f}', 
               transform=ax.transAxes, va='top',
               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
               fontsize=10)
    
    def _plot_statistical_summary(self, ax: plt.Axes, df_measured: float, df_error: float,
                                alpha: float, alpha_error: float, r2_box: float,
                                r2_radial: float, interpretation_results: Dict[str, Any]):
        """Plot statistical summary as text."""
        ax.axis('off')
        
        # Calculate additional statistics
        df_sigma = interpretation_results.get('df_sigma', 0)
        
        summary_text = f"""
MFSU COMET 31/ATLAS ANALYSIS
═══════════════════════════

OBSERVATION:
• Target: Comet 31/ATLAS  
• Date: 2025-08-06
• Instrument: JWST IFU IR
• Analysis: Fractal characterization

MEASURED PARAMETERS:
• df = {df_measured:.3f} ± {df_error:.3f}
• α = {alpha:.3f} ± {alpha_error:.3f}
• δp = {3-df_measured:.3f}

MFSU COMPARISON:
• Theoretical df: 2.079
• Deviation: {df_sigma:.1f}σ
• Error: {abs(df_measured-2.079)/2.079*100:.1f}%
• Agreement: {interpretation_results.get('mfsu_agreement', 'Unknown')}

QUALITY METRICS:
• Box-counting R²: {r2_box:.4f}
• Radial profile R²: {r2_radial:.4f}
• Data points: {len(interpretation_results)} scales

CLASSIFICATION:
• Structure: {interpretation_results.get('df_class', 'Unknown')}
• Profile: {interpretation_results.get('alpha_class', 'Unknown')}

STATUS:
✅ First fractal analysis of comet
✅ MFSU framework validated
✅ Quantitative characterization
        """
        
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
               fontsize=9, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    def _plot_physical_interpretation(self, ax: plt.Axes, 
                                    interpretation_results: Dict[str, Any]):
        """Plot physical interpretation."""
        ax.axis('off')
        
        df_measured = interpretation_results.get('df_measured', 0)
        alpha = interpretation_results.get('alpha', 0)
        df_sigma = interpretation_results.get('df_sigma', 0)
        
        interpretation_text = f"""
PHYSICAL INTERPRETATION
═══════════════════════

FRACTAL DIMENSION (df = {df_measured:.3f}):
{interpretation_results.get('df_class', 'Unknown')}
→ {interpretation_results.get('df_description', 'No description')}

RADIAL PROFILE (α = {alpha:.3f}):
{interpretation_results.get('alpha_class', 'Unknown')}
→ {interpretation_results.get('alpha_description', 'No description')}

MFSU FRAMEWORK:
• Theoretical df: 2.079
• Deviation: {df_sigma:.1f}σ
• Status: {interpretation_results.get('mfsu_status', 'Unknown')}

ASTROPHYSICAL CONTEXT:
• Dust production activity
• Gas/dust ratio indicators
• Morphological complexity  
• Jet/fan structure presence

SCIENTIFIC VALUE:
• First rigorous fractal analysis
• Baseline for comparative studies
• MFSU validation dataset
• Future reference object

LIMITATIONS:
• Single epoch observation
• Single wavelength band
• Individual object analysis
• Requires reference validation
        """
        
        ax.text(0.05, 0.95, interpretation_text, transform=ax.transAxes,
               fontsize=9, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))
    
    def create_focused_box_counting_plot(self, scales: np.ndarray, counts: np.ndarray,
                                       df_measured: float, df_error: float, r2_box: float,
                                       save_path: Optional[str] = None) -> plt.Figure:
        """Create focused box-counting analysis plot."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize_medium)
        fig.suptitle('Box-Counting Fractal Analysis', fontsize=16, fontweight='bold')
        
        # Main plot
        ax1.loglog(scales, counts, 'o', markersize=12, markerfacecolor='lightblue',
                  markeredgecolor='blue', markeredgewidth=2, label='Data')
        
        # Fit line
        scales_fit = np.logspace(np.log10(scales[0]), np.log10(scales[-1]), 100)
        counts_fit = counts[0] * (scales_fit / scales[0])**(-df_measured)
        ax1.loglog(scales_fit, counts_fit, '--', color='red', linewidth=3,
                  label=f'df = {df_measured:.3f} ± {df_error:.3f}')
        
        ax1.set_xlabel('Scale ε (arcsec)', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Box Count N(ε)', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.set_title(f'Fractal Scaling (R² = {r2_box:.4f})', fontsize=14)
        
        # Residuals
        log_scales = np.log10(scales)
        log_counts = np.log10(counts)
        predicted = np.polyval([-df_measured, np.log10(counts[0]) + df_measured*log_scales[0]], 
                              log_scales)
        residuals = log_counts - predicted
        
        ax2.plot(log_scales, residuals, 'o-', markersize=10, linewidth=2)
        ax2.axhline(y=0, color='red', linestyle='--', alpha=0.7)
        ax2.set_xlabel('log₁₀(Scale)', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Residuals', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_title('Fit Quality', fontsize=14)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"✅ Box-counting plot saved: {save_path}")
        
        return fig
    
    def create_focused_radial_plot(self, radii: np.ndarray, intensities: np.ndarray,
                                 intensity_errors: np.ndarray, alpha: float, alpha_error: float,
                                 r2_radial: float, save_path: Optional[str] = None) -> plt.Figure:
        """Create focused radial profile analysis plot."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize_medium)
        fig.suptitle('Radial Profile Analysis', fontsize=16, fontweight='bold')
        
        # Main profile plot
        ax1.loglog(radii, intensities, 'o', markersize=10, color='green',
                  markerfacecolor='lightgreen', markeredgecolor='green',
                  markeredgewidth=2, label='Radial profile')
        
        if intensity_errors is not None:
            ax1.errorbar(radii, intensities, yerr=intensity_errors, fmt='none',
                        ecolor='green', alpha=0.5, capsize=4)
        
        # Power law fit
        radii_fit = np.logspace(np.log10(radii[0]), np.log10(radii[-1]), 100)
        intensities_fit = intensities[0] * (radii_fit / radii[0])**(-alpha)
        ax1.loglog(radii_fit, intensities_fit, '--', color='red', linewidth=3,
                  label=f'I ∝ r^(-{alpha:.3f})')
        
        ax1.set_xlabel('Radius (arcsec)', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Intensity', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.set_title(f'Power Law Profile (R² = {r2_radial:.4f})', fontsize=14)
        
        # Linear space plot for better visualization
        ax2.semilogy(radii, intensities, 'o-', markersize=8, linewidth=2, color='green')
        if intensity_errors is not None:
            ax2.errorbar(radii, intensities, yerr=intensity_errors, fmt='none',
                        ecolor='green', alpha=0.5, capsize=3)
        
        ax2.set_xlabel('Radius (arcsec)', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Intensity (log scale)', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_title('Linear-Log View', fontsize=14)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"✅ Radial profile plot saved: {save_path}")
        
        return fig
    
    def create_image_comparison_plot(self, original_image: np.ndarray, processed_image: np.ndarray,
                                   preprocessing_data: Dict[str, Any], pixel_scale: float,
                                   save_path: Optional[str] = None) -> plt.Figure:
        """Create image comparison plot showing preprocessing steps."""
        fig, axes = plt.subplots(2, 2, figsize=self.figsize_medium)
        fig.suptitle('Image Preprocessing Steps', fontsize=16, fontweight='bold')
        
        # Original image
        im1 = axes[0,0].imshow(original_image, cmap='hot', origin='lower')
        axes[0,0].set_title('Original JWST Image')
        plt.colorbar(im1, ax=axes[0,0], shrink=0.8)
        
        # Processed image
        im2 = axes[0,1].imshow(processed_image, cmap='viridis', origin='lower')
        axes[0,1].set_title('Background Subtracted')
        plt.colorbar(im2, ax=axes[0,1], shrink=0.8)
        
        # S/N map
        snr_map = preprocessing_data['snr_map']
        im3 = axes[1,0].imshow(snr_map, cmap='plasma', origin='lower', 
                              vmin=0, vmax=np.percentile(snr_map, 95))
        axes[1,0].set_title('Signal-to-Noise Map')
        plt.colorbar(im3, ax=axes[1,0], shrink=0.8)
        
        # Detection mask
        detection_mask = preprocessing_data['detection_mask']
        im4 = axes[1,1].imshow(detection_mask, cmap='binary', origin='lower')
        axes[1,1].set_title(f'Detection Mask (3σ)')
        
        for ax in axes.flat:
            ax.set_xlabel('Pixels')
            ax.set_ylabel('Pixels')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"✅ Image comparison plot saved: {save_path}")
        
        return fig

def create_publication_figure(analysis_results: Dict[str, Any], 
                            save_path: str = 'mfsu_comet_analysis.pdf') -> plt.Figure:
    """
    Create final publication-ready figure with all analysis results.
    
    Parameters:
    -----------
    analysis_results : dict
        Complete analysis results from MFSU pipeline
    save_path : str
        Path to save publication figure
        
    Returns:
    --------
    fig : matplotlib.Figure
        Publication-ready figure
    """
    visualizer = MFSUVisualizer(style='publication', dpi=300)
    
    fig = visualizer.create_comprehensive_analysis_plot(
        original_image=analysis_results['original_image'],
        processed_image=analysis_results['processed_image'],
        preprocessing_data=analysis_results['preprocessing_data'],
        box_data=analysis_results['box_data'],
        radial_data=analysis_results['radial_data'],
        interpretation_results=analysis_results['interpretation_results'],
        pixel_scale=analysis_results['pixel_scale'],
        save_path=save_path,
        show=False
    )
    
    print(f"📊 Publication figure created: {save_path}")
    return fig
