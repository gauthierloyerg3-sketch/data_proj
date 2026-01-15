"""
Main entry point for Retail Demand Forecasting Project.

Orchestrates the complete workflow:
1. Load and preprocess data
2. Train all three models
3. Evaluate and compare models
4. Generate predictions
5. Create order recommendations
6. Save visualizations and metrics

Run with: python main.py
"""

import warnings
# Suppress numerical warnings from sklearn (divide by zero, overflow, etc.)
# These are handled by our data cleaning functions
warnings.filterwarnings('ignore', category=RuntimeWarning, module='sklearn')

import numpy as np
import pandas as pd
import random
from pathlib import Path

from src.config import ProjectConfig

# Central configuration
CONFIG = ProjectConfig()

# Set random seeds for reproducibility
RANDOM_STATE = CONFIG.training.random_state
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

from src.data_loader import load_and_preprocess_data
from src.models import (
    prepare_features_and_target,
    prepare_features_and_target_revenue,
    train_all_models,
    train_revenue_models,
    compare_models,
    cross_validate_all_models,
)
from src.evaluation import (
    evaluate_and_visualize,
    interpret_model_performance,
    analyze_per_product_performance,
    create_final_recommendations_report,
    evaluate_revenue_predictions,
    create_revenue_forecast_report,
    plot_revenue_forecasts,
    analyze_weather_seasonality_region_impact,
)
from src.output_config import OutputConfig, cleanup_redundant_files
from src.output_formatter import (
    print_header,
    print_section,
    print_progress,
    print_success,
    print_table,
    print_executive_summary,
    print_recommendations_table,
    print_urgency_summary,
    print_model_performance_summary,
)


def main():
    """
    Main workflow for demand forecasting and order recommendations.
    """
    verbose = CONFIG.output.verbose
    show_technical = CONFIG.output.show_technical_details
    use_emoji = CONFIG.output.use_emoji

    try:
        # Processing phase - group all technical steps
        # Step 1: Load and preprocess data
        if CONFIG.output.show_progress:
            print_progress("Processing data and training models", done=False)
        train_df, test_df = load_and_preprocess_data(
            data_path=CONFIG.paths.raw_data,
            save_processed=True,
            processed_path=CONFIG.paths.processed_data,
            add_stock_to_processed=CONFIG.preprocessing.add_stock_to_processed,
            days_of_coverage=CONFIG.preprocessing.days_of_coverage,
            coverage_method=CONFIG.preprocessing.coverage_method,
            stock_path=CONFIG.paths.stock_data,
            test_size=CONFIG.preprocessing.test_size,
            min_history_days=CONFIG.preprocessing.min_history_days,
            lags=CONFIG.preprocessing.lags,
            moving_average_windows=CONFIG.preprocessing.moving_average_windows,
            verbose=verbose,
        )

        # Extract stock from processed data
        from src.data_loader import extract_stock_from_dataframe
        current_stock = extract_stock_from_dataframe(test_df)

        # Step 2: Prepare features and targets
        X_train, y_train = prepare_features_and_target(train_df)
        X_test, y_test = prepare_features_and_target(test_df)

        # Step 3: Train all models
        models, scalers = train_all_models(
            X_train, y_train, train_df=train_df, save_models=True, 
            include_baselines=True,
            use_tuning=CONFIG.training.use_hyperparameter_tuning,
            include_xgboost=CONFIG.training.include_xgboost,
            tuning_n_iter=CONFIG.training.tuning_n_iter,
            tuning_cv_splits=CONFIG.training.tuning_cv_splits,
            verbose=verbose
        )

        # Step 3.5: Cross-validation (silent unless verbose)
        cv_results = cross_validate_all_models(
            X_train, y_train,
            n_splits=CONFIG.training.n_splits,
            include_xgboost=CONFIG.training.include_xgboost,
            verbose=verbose
        )

        # Step 4: Evaluate and compare models
        comparison_df = compare_models(
            models, X_test, y_test, scalers=scalers, test_df=test_df, cost_config=CONFIG.costs
        )

        # Save comparison results
        comparison_path = Path("results/metrics/model_comparison.csv")
        comparison_path.parent.mkdir(parents=True, exist_ok=True)
        comparison_df.to_csv(comparison_path, index=False)

        # Step 4.5: Model interpretation analysis (silent unless verbose)
        if show_technical:
            interpret_model_performance(models, X_train, y_train, comparison_df)

        if CONFIG.output.show_progress:
            print_progress("Processing data and training models", done=True)
        else:
            print()  # Add spacing even if progress is suppressed

        # Step 5: Detailed evaluation and visualization for each model (silent)
        # Configure output: only save essential files, prevent redundancy
        output_config = OutputConfig(
            save_individual_models=False,  # Only save best model's detailed files
            save_best_model_only=True,
            save_metrics_txt=False,  # Info already in model_comparison.csv
            save_individual_recommendations=False,  # Use final report only
            save_individual_plots=True,  # Keep essential plots for comparison
            save_final_report=True,
            cleanup_redundant_files=True,
            # Plot configuration
            save_predictions_plot=True,  # Keep for model comparison
            save_residuals_plot=True,  # Keep for diagnostics
            save_feature_importance_plot=True,  # Keep for comparison
            save_sales_trends_plot=True,  # Save ONCE (same for all models)
            save_recommendations_plot=False,  # Info already in final report
        )

        all_metrics = {}
        all_recommendations = {}  # Use dict instead of list for proper mapping
        best_model_name = comparison_df.loc[comparison_df["RMSE"].idxmin(), "Model"]

        for model_name, model in models.items():
            # Skip baselines from detailed evaluation (they're already in comparison)
            if model_name.startswith("baseline_"):
                continue
                
            is_best = model_name == best_model_name
            model_scaler = scalers.get(model_name)
            metrics, recommendations = evaluate_and_visualize(
                model,
                model_name,
                X_test,
                y_test,
                test_df,
                train_df,
                output_config=output_config,
                is_best_model=is_best,
                current_stock=current_stock,
                safety_stock_multiplier=CONFIG.recommendation.safety_stock_multiplier,
                target_days_of_coverage=CONFIG.recommendation.target_days_of_coverage,
                scaler=model_scaler,
            )
            all_metrics[model_name] = metrics
            all_recommendations[model_name] = recommendations

        # Step 5.5: Per-product performance analysis (silent unless verbose)
        best_model = models[best_model_name]
        best_model_scaler = scalers.get(best_model_name)
        per_product_df = analyze_per_product_performance(
            best_model, X_test, y_test, test_df, best_model_name, scaler=best_model_scaler
        )
        
        # Step 6: Revenue forecasting (if enabled)
        if CONFIG.forecasting.forecast_revenue:
            if verbose:
                print("Training revenue forecasting models...")
            
            # Prepare revenue features and target
            X_train_revenue, y_train_revenue = prepare_features_and_target_revenue(train_df, target_col="revenue")
            X_test_revenue, y_test_revenue = prepare_features_and_target_revenue(test_df, target_col="revenue")
            
            # Train revenue models
            revenue_models, revenue_scalers = train_revenue_models(
                X_train_revenue, y_train_revenue, save_models=True, verbose=verbose
            )
            
            # Compare revenue models
            revenue_comparison_df = compare_models(
                revenue_models, X_test_revenue, y_test_revenue, 
                scalers=revenue_scalers, test_df=test_df, cost_config=None
            )
            
            # Save revenue comparison
            revenue_comparison_path = Path("results/metrics/revenue_model_comparison.csv")
            revenue_comparison_path.parent.mkdir(parents=True, exist_ok=True)
            revenue_comparison_df.to_csv(revenue_comparison_path, index=False)
            
            # Get best revenue model
            best_revenue_model_name = revenue_comparison_df.loc[revenue_comparison_df["RMSE"].idxmin(), "Model"]
            best_revenue_model = revenue_models[best_revenue_model_name]
            best_revenue_scaler = revenue_scalers.get(best_revenue_model_name)
            
            # Evaluate best revenue model
            if best_revenue_scaler is not None:
                X_test_revenue_scaled = pd.DataFrame(
                    best_revenue_scaler.transform(X_test_revenue),
                    columns=X_test_revenue.columns,
                    index=X_test_revenue.index,
                )
                y_pred_revenue = best_revenue_model.predict(X_test_revenue_scaled)
            else:
                y_pred_revenue = best_revenue_model.predict(X_test_revenue)
            
            y_pred_revenue_series = pd.Series(y_pred_revenue, index=y_test_revenue.index).clip(lower=0)
            
            # Calculate revenue metrics
            revenue_metrics = evaluate_revenue_predictions(y_test_revenue, y_pred_revenue_series)
            
            # Create revenue forecast report
            revenue_report = create_revenue_forecast_report(
                y_pred_revenue_series, test_df, best_revenue_model_name
            )
            
            # Plot revenue forecasts
            plot_revenue_forecasts(
                y_test_revenue, y_pred_revenue_series, test_df, 
                best_revenue_model_name, "results/figures/revenue_forecasts.png"
            )
            
            # Analyze Weather/Seasonality/Region impact
            impact_df = analyze_weather_seasonality_region_impact(
                best_revenue_model, X_test_revenue, y_test_revenue, 
                test_df, best_revenue_model_name, scaler=best_revenue_scaler
            )
            
            if verbose:
                print(f"Revenue forecasting completed. Best model: {best_revenue_model_name}")
                print(f"Revenue MAE: {revenue_metrics['MAE']:.2f}, RMSE: {revenue_metrics['RMSE']:.2f}")

        # Get best model recommendations for display
        best_model_recommendations = all_recommendations.get(best_model_name)
        best_metrics = comparison_df[comparison_df["Model"] == best_model_name].iloc[0]

        # Calculate summary statistics
        if best_model_recommendations is not None:
            urgency_counts = best_model_recommendations["reorder_urgency"].value_counts().to_dict()
            total_order_qty = best_model_recommendations["recommended_order"].sum()
            total_products = len(best_model_recommendations)
            
            # Calculate total cost if available
            total_cost = None
            if "total_cost" in best_metrics:
                total_cost = best_metrics["total_cost"]
        else:
            urgency_counts = {}
            total_order_qty = 0.0
            total_products = 0

        # Display model performance summary (before executive summary)
        print_model_performance_summary(
            comparison_df=comparison_df,
            best_model_name=best_model_name,
            best_metrics=best_metrics,
        )

        # Display executive summary and recommendations
        model_rmse = best_metrics.get("RMSE") if "RMSE" in best_metrics else None
        print_executive_summary(
            total_products=total_products,
            urgency_counts=urgency_counts,
            total_order_qty=total_order_qty,
            total_cost=total_cost,
            model_name=best_model_name,
            model_rmse=model_rmse,
            use_emoji=use_emoji,
        )

        # Display recommendations table
        if best_model_recommendations is not None:
            print_recommendations_table(best_model_recommendations, use_emoji=use_emoji)

        # Create and save final report
        if output_config.save_final_report and best_model_recommendations is not None:
            final_report = create_final_recommendations_report(
                best_model_recommendations, best_model_name
            )

        # Cleanup redundant files (silent)
        if output_config.cleanup_redundant_files:
            cleanup_redundant_files(output_config, best_model_name)

        # Technical details section (optional)
        if show_technical:
            print_section("TECHNICAL DETAILS", level=2)
            print("\nModel Comparison:")
            print_table(comparison_df, title=None)
            
            if not cv_results.empty:
                print("\nCross-Validation Results:")
                print_table(cv_results, title=None)
            
            if not per_product_df.empty:
                print("\nPer-Product Performance:")
                print_table(per_product_df, title=None)
            
            print(f"\nBest Model: {best_model_name}")
            print(f"  MAE: {best_metrics['MAE']:.4f}")
            print(f"  RMSE: {best_metrics['RMSE']:.4f}")
            if total_cost is not None:
                print(f"  Total Cost: ${total_cost:,.2f}")

        # Final summary
        print()
        print_section("REPORTS SAVED", level=2)
        print_success(f"Order recommendations: results/metrics/final_order_recommendations.csv")
        if show_technical:
            print_success(f"Model comparison: results/metrics/model_comparison.csv")
            print_success(f"Cross-validation: results/metrics/cross_validation_comparison.csv")
        print()

    except FileNotFoundError as e:
        from src.output_formatter import print_error
        print_error(f"File not found: {e}")
        print("Please ensure the dataset is in data/raw/supermarket_sales.csv")
        return 1

    except Exception as e:
        from src.output_formatter import print_error
        print_error(f"{type(e).__name__}: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
