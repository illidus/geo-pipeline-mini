"""
Visualization functions for soil pipeline results.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional


def create_results_visualization(predictions_file: str, output_file: str = "assets/results.png") -> str:
    """
    Create visualization of pipeline results.
    
    Args:
        predictions_file: Path to predictions.csv file
        output_file: Path to save the visualization
        
    Returns:
        Path to saved visualization file
    """
    # Load predictions
    df = pd.read_csv(predictions_file)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Geo Pipeline Mini - Soil Property Analysis Results', fontsize=16, fontweight='bold')
    
    # Target properties
    properties = ['soil_organic_matter', 'clay_content', 'ph']
    
    # Row 1: Actual vs Predicted scatter plots
    for i, prop in enumerate(properties):
        ax = axes[0, i]
        actual_col = f"{prop}_actual"
        pred_col = f"{prop}_predicted"
        
        if actual_col in df.columns and pred_col in df.columns:
            actual = df[actual_col]
            predicted = df[pred_col]
            
            # Scatter plot
            ax.scatter(actual, predicted, alpha=0.7, s=50)
            
            # Perfect prediction line
            min_val = min(actual.min(), predicted.min())
            max_val = max(actual.max(), predicted.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='Perfect prediction')
            
            # Calculate R²
            corr_matrix = np.corrcoef(actual, predicted)
            r_squared = corr_matrix[0, 1] ** 2
            
            ax.set_xlabel(f'Actual {prop.replace("_", " ").title()}')
            ax.set_ylabel(f'Predicted {prop.replace("_", " ").title()}')
            ax.set_title(f'{prop.replace("_", " ").title()}\nR² = {r_squared:.3f}')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    # Row 2: Spatial distribution maps
    for i, prop in enumerate(properties):
        ax = axes[1, i]
        actual_col = f"{prop}_actual"
        
        if actual_col in df.columns:
            # Create spatial map
            scatter = ax.scatter(df['x'], df['y'], c=df[actual_col], 
                               cmap='viridis', s=100, alpha=0.8)
            
            ax.set_xlabel('X Coordinate')
            ax.set_ylabel('Y Coordinate')
            ax.set_title(f'{prop.replace("_", " ").title()} Spatial Distribution')
            ax.set_aspect('equal')
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label(prop.replace("_", " ").title())
    
    plt.tight_layout()
    
    # Ensure output directory exists
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save figure
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Visualization saved: {output_file}")
    return output_file


def create_performance_chart(metrics_file: str, output_file: str = "assets/performance.png") -> str:
    """
    Create performance metrics visualization.
    
    Args:
        metrics_file: Path to metrics.json file
        output_file: Path to save the chart
        
    Returns:
        Path to saved chart file
    """
    import json
    
    # Load metrics
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Pipeline Performance Metrics', fontsize=16, fontweight='bold')
    
    # Model performance
    model_perf = metrics['model_performance']
    properties = list(model_perf.keys())
    r2_scores = [model_perf[prop]['r2_score'] for prop in properties]
    model_types = [model_perf[prop]['model_type'] for prop in properties]
    
    # Bar chart of R² scores
    bars = ax1.bar(range(len(properties)), r2_scores, 
                   color=['skyblue', 'lightcoral', 'lightgreen'])
    ax1.set_xlabel('Soil Properties')
    ax1.set_ylabel('R² Score')
    ax1.set_title('Model Performance (R² Scores)')
    ax1.set_xticks(range(len(properties)))
    ax1.set_xticklabels([prop.replace('_', ' ').title() for prop in properties], rotation=45)
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    
    # Add model type labels on bars
    for i, (bar, model_type) in enumerate(zip(bars, model_types)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{model_type}\n{height:.3f}',
                ha='center', va='bottom', fontsize=10)
    
    # Processing times
    times = metrics['processing_times']
    steps = list(times.keys())
    durations = list(times.values())
    
    # Pie chart of processing times
    ax2.pie(durations, labels=steps, autopct='%1.1f%%', startangle=90)
    ax2.set_title(f'Processing Time Breakdown\nTotal: {sum(durations):.2f}s')
    
    plt.tight_layout()
    
    # Ensure output directory exists
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save figure
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📈 Performance chart saved: {output_file}")
    return output_file


if __name__ == "__main__":
    # Generate visualizations if files exist
    predictions_file = "outputs/predictions.csv"
    metrics_file = "outputs/metrics.json"
    
    if Path(predictions_file).exists():
        create_results_visualization(predictions_file)
        print("✅ Results visualization created")
    else:
        print(f"❌ Predictions file not found: {predictions_file}")
    
    if Path(metrics_file).exists():
        create_performance_chart(metrics_file)
        print("✅ Performance chart created")
    else:
        print(f"❌ Metrics file not found: {metrics_file}")