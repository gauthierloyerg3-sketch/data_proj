"""
Output configuration for controlling what files are saved.

This module provides centralized control over output file generation,
preventing redundant files and allowing flexible configuration.
"""

from dataclasses import dataclass
from typing import Optional
from pathlib import Path


@dataclass
class OutputConfig:
    """
    Configuration for output file generation.
    
    Controls which files are saved to prevent redundancy and clutter.
    """
    
    # Model file saving strategy
    save_individual_models: bool = False  # Save files for all models
    save_best_model_only: bool = True  # Only save best model's detailed outputs
    
    # Specific output types
    save_metrics_txt: bool = False  # Individual .txt files (info already in comparison.csv)
    save_individual_recommendations: bool = False  # Individual .csv files (use final report)
    save_individual_plots: bool = True  # Keep plots for all models (useful for comparison)
    save_final_report: bool = True  # Always save final report
    
    # Plot types (only used if save_individual_plots or save_best_model_only is True)
    save_predictions_plot: bool = True  # Predictions vs actual (useful for comparison)
    save_residuals_plot: bool = True  # Residual diagnostics (useful for comparison)
    save_feature_importance_plot: bool = True  # Feature importance (useful for comparison)
    save_sales_trends_plot: bool = True  # Sales trends (SAVED ONCE, not per model - same data)
    save_recommendations_plot: bool = False  # Recommendations visualization (info in final report)
    
    # Cleanup
    cleanup_redundant_files: bool = True  # Remove redundant files after processing
    
    def should_save_model_files(self, model_name: str, is_best: bool = False) -> bool:
        """
        Check if files should be saved for a given model.
        
        Args:
            model_name: Name of the model
            is_best: Whether this is the best performing model
            
        Returns:
            True if files should be saved for this model
        """
        if self.save_individual_models:
            return True
        if self.save_best_model_only and is_best:
            return True
        return False
    
    def should_save_metrics_txt(self) -> bool:
        """Check if individual metrics .txt files should be saved."""
        return self.save_metrics_txt
    
    def should_save_individual_recommendations(self) -> bool:
        """Check if individual recommendations .csv files should be saved."""
        return self.save_individual_recommendations
    
    def should_save_plot_type(self, plot_type: str, is_best: bool = False) -> bool:
        """
        Check if a specific plot type should be saved.
        
        Args:
            plot_type: One of 'predictions', 'residuals', 'feature_importance', 
                      'sales_trends', 'recommendations'
            is_best: Whether this is the best performing model
            
        Returns:
            True if this plot type should be saved
        """
        # Sales trends is saved once, not per model
        if plot_type == "sales_trends":
            return self.save_sales_trends_plot
        
        # Recommendations plot is optional
        if plot_type == "recommendations":
            return self.save_recommendations_plot and (self.save_individual_plots or is_best)
        
        # Other plots follow the main configuration
        should_save = self.save_individual_plots or (self.save_best_model_only and is_best)
        
        if plot_type == "predictions":
            return should_save and self.save_predictions_plot
        if plot_type == "residuals":
            return should_save and self.save_residuals_plot
        if plot_type == "feature_importance":
            return should_save and self.save_feature_importance_plot
        
        return False


def cleanup_redundant_files(output_config: OutputConfig, best_model_name: str):
    """
    Remove redundant files after processing.
    
    Removes individual model files if only best model should be kept,
    and removes old/unused file patterns.
    
    Args:
        output_config: Output configuration
        best_model_name: Name of the best performing model
    """
    if not output_config.cleanup_redundant_files:
        return
    
    metrics_dir = Path("results/metrics")
    figures_dir = Path("results/figures")
    
    removed_count = 0
    
    # Remove individual metrics .txt files if not configured
    if not output_config.save_metrics_txt and metrics_dir.exists():
        for txt_file in metrics_dir.glob("*_metrics.txt"):
            txt_file.unlink()
            removed_count += 1
            print(f"Removed redundant file: {txt_file.name}")
    
    # Remove individual recommendations .csv files if not configured
    if not output_config.save_individual_recommendations and metrics_dir.exists():
        for csv_file in metrics_dir.glob("*_recommendations.csv"):
            # Keep final_order_recommendations.csv
            if "final" in csv_file.name.lower():
                continue
            if best_model_name and best_model_name in csv_file.stem:
                # Keep best model's recommendations if present
                continue
            else:
                csv_file.unlink()
                removed_count += 1
                print(f"Removed redundant file: {csv_file.name}")
    
    # Remove old recommendation report files (if they exist from old code)
    if metrics_dir.exists():
        for report_file in metrics_dir.glob("*_recommendations_report.csv"):
            report_file.unlink()
            removed_count += 1
            print(f"Removed old file: {report_file.name}")
    
    # Clean up redundant plot files
    if figures_dir.exists():
        # Remove model-specific sales_trends plots (keep only sales_trends.png without model prefix)
        # Since sales_trends is based on training data, it's identical for all models
        sales_trends_shared = figures_dir / "sales_trends.png"
        for sales_file in figures_dir.glob("*_sales_trends.png"):
            # Keep the shared one, remove model-specific ones
            if sales_file.name != "sales_trends.png":
                sales_file.unlink()
                removed_count += 1
                print(f"Removed duplicate file: {sales_file.name} (sales_trends is same for all models)")
        
        # Remove recommendations plots if not configured
        if not output_config.save_recommendations_plot:
            for rec_file in figures_dir.glob("*_recommendations.png"):
                rec_file.unlink()
                removed_count += 1
                print(f"Removed file: {rec_file.name} (info in final report)")
    
    if removed_count > 0:
        print(f"\nCleaned up {removed_count} redundant file(s)\n")
    else:
        print("No redundant files found to clean up\n")
