#!/usr/bin/env python3
"""
Geo Pipeline Mini - Transform gamma + satellite data into soil property indices
Produces outputs/predictions.csv, outputs/metrics.json, and assets/results.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import time
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')


@dataclass
class PipelineConfig:
    """Configuration for the soil property pipeline."""
    input_dir: str = "data/raw"
    output_dir: str = "outputs"
    assets_dir: str = "assets"
    window_size: int = 5  # For windowed statistics
    test_size: float = 0.2
    random_state: int = 42


class SoilPipeline:
    """Main pipeline for processing gamma/satellite data into soil indices."""
    
    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()
        self.data = {}
        self.features = {}
        self.models = {}
        self.predictions = {}
        self.metrics = {}
        
        # Create directories
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.assets_dir).mkdir(parents=True, exist_ok=True)
    
    def generate_synthetic_data(self, n_samples: int = 1000) -> pd.DataFrame:
        """Generate synthetic gamma radiometric and satellite data."""
        print("🌍 Generating synthetic field data...")
        
        np.random.seed(self.config.random_state)
        
        # Generate spatial coordinates (Midwest US)
        longitude = np.random.uniform(-100, -80, n_samples)
        latitude = np.random.uniform(35, 45, n_samples)
        
        # Generate synthetic gamma radiometric data
        # Potassium (K) - typically 0.5-3.5% in soils
        potassium_pct = np.random.gamma(2, 0.8, n_samples)
        
        # Uranium (U) - typically 1-4 ppm in soils  
        uranium_ppm = np.random.exponential(2.5, n_samples)
        
        # Thorium (Th) - typically 5-15 ppm in soils
        thorium_ppm = np.random.gamma(3, 3, n_samples)
        
        # Generate synthetic satellite indices
        # NDVI (Normalized Difference Vegetation Index) 0-1
        ndvi = np.random.beta(2, 2, n_samples)
        
        # NDWI (Normalized Difference Water Index) -1 to 1
        ndwi = np.random.normal(0.1, 0.3, n_samples)
        ndwi = np.clip(ndwi, -1, 1)
        
        # Clay index from satellite data
        clay_index = np.random.beta(1.5, 3, n_samples)
        
        # Elevation (meters)
        elevation = np.random.normal(300, 150, n_samples)
        elevation = np.clip(elevation, 0, 1000)
        
        # Create realistic soil property targets
        # Organic matter (0.5-8%)
        organic_matter = (
            0.3 * potassium_pct + 
            0.4 * ndvi + 
            0.1 * uranium_ppm +
            0.2 * (elevation / 1000) +
            np.random.normal(0, 0.5, n_samples)
        )
        organic_matter = np.clip(organic_matter, 0.5, 8.0)
        
        # Soil pH (4.5-8.5)
        ph = (
            6.5 + 
            0.3 * (thorium_ppm / 10) + 
            0.2 * clay_index - 
            0.1 * uranium_ppm +
            0.1 * (potassium_pct - 2) +
            np.random.normal(0, 0.4, n_samples)
        )
        ph = np.clip(ph, 4.5, 8.5)
        
        # Cation Exchange Capacity (5-40 meq/100g)
        cec = (
            5 + 
            8 * clay_index + 
            3 * (organic_matter / 8) +
            2 * (potassium_pct / 3) +
            np.random.normal(0, 2, n_samples)
        )
        cec = np.clip(cec, 5, 40)
        
        # Create DataFrame
        data = pd.DataFrame({
            'longitude': longitude,
            'latitude': latitude,
            'potassium_pct': potassium_pct,
            'uranium_ppm': uranium_ppm,
            'thorium_ppm': thorium_ppm,
            'ndvi': ndvi,
            'ndwi': ndwi,
            'clay_index': clay_index,
            'elevation': elevation,
            'organic_matter_pct': organic_matter,
            'ph': ph,
            'cec_meq_100g': cec
        })
        
        print(f"Generated {len(data)} samples with {data.shape[1]} features")
        self.data['raw'] = data
        return data
    
    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create additional features from raw gamma and satellite data."""
        print("🔧 Engineering features...")
        
        features = data.copy()
        
        # Gamma radiometric ratios (standard in geophysics)
        features['k_th_ratio'] = features['potassium_pct'] / (features['thorium_ppm'] + 0.001)
        features['u_k_ratio'] = features['uranium_ppm'] / (features['potassium_pct'] + 0.001)
        features['th_k_ratio'] = features['thorium_ppm'] / (features['potassium_pct'] + 0.001)
        features['u_th_ratio'] = features['uranium_ppm'] / (features['thorium_ppm'] + 0.001)
        
        # Vegetation-adjusted indices
        features['gamma_vigor'] = features['potassium_pct'] * features['ndvi']
        features['moisture_gamma'] = features['ndwi'] * features['uranium_ppm']
        features['elevation_ndvi'] = features['elevation'] * features['ndvi'] / 1000
        
        # Composite soil indices
        features['soil_complexity'] = (
            features['clay_index'] * 
            features['k_th_ratio'] * 
            (1 + features['ndvi'])
        )
        
        features['fertility_index'] = (
            features['potassium_pct'] * 0.4 +
            features['ndvi'] * 0.3 +
            (1 / (features['uranium_ppm'] + 1)) * 0.3
        )
        
        # Spatial features
        features['lat_lon_interaction'] = features['latitude'] * features['longitude']
        features['distance_to_center'] = np.sqrt(
            (features['longitude'] + 90) ** 2 + 
            (features['latitude'] - 40) ** 2
        )
        
        print(f"Created {len(features.columns)} total features")
        self.features = features
        return features
    
    def train_models(self, features: pd.DataFrame) -> Dict[str, Dict]:
        """Train prediction models for soil properties."""
        print("🤖 Training prediction models...")
        
        # Define target variables
        targets = ['organic_matter_pct', 'ph', 'cec_meq_100g']
        
        # Feature columns (exclude targets and coordinates)
        feature_cols = [col for col in features.columns 
                       if col not in targets + ['longitude', 'latitude']]
        X = features[feature_cols]
        
        models = {}
        metrics = {}
        
        for target in targets:
            print(f"  Training {target} model...")
            
            y = features[target]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.config.test_size, 
                random_state=self.config.random_state
            )
            
            # Train multiple models
            rf_model = RandomForestRegressor(
                n_estimators=100, 
                random_state=self.config.random_state,
                n_jobs=-1
            )
            ridge_model = Ridge(alpha=1.0)
            
            # Fit models
            rf_model.fit(X_train, y_train)
            ridge_model.fit(X_train, y_train)
            
            # Evaluate models
            rf_pred = rf_model.predict(X_test)
            ridge_pred = ridge_model.predict(X_test)
            
            rf_r2 = r2_score(y_test, rf_pred)
            ridge_r2 = r2_score(y_test, ridge_pred)
            
            # Select best model
            if rf_r2 > ridge_r2:
                best_model = rf_model
                best_pred = rf_pred
                model_type = "RandomForest"
            else:
                best_model = ridge_model
                best_pred = ridge_pred
                model_type = "Ridge"
            
            models[target] = {
                'model': best_model,
                'feature_names': feature_cols,
                'model_type': model_type
            }
            
            # Calculate comprehensive metrics
            rmse = np.sqrt(mean_squared_error(y_test, best_pred))
            mae = np.mean(np.abs(y_test - best_pred))
            
            metrics[target] = {
                'r2_score': float(rf_r2 if rf_r2 > ridge_r2 else ridge_r2),
                'rmse': float(rmse),
                'mae': float(mae),
                'model_type': model_type,
                'n_features': len(feature_cols),
                'n_train': len(X_train),
                'n_test': len(X_test),
                'target_mean': float(y.mean()),
                'target_std': float(y.std())
            }
            
            print(f"    {target}: R² = {metrics[target]['r2_score']:.3f}, "
                  f"RMSE = {metrics[target]['rmse']:.3f} ({model_type})")
        
        self.models = models
        self.metrics = metrics
        return models, metrics
    
    def generate_predictions(self, features: pd.DataFrame) -> pd.DataFrame:
        """Generate predictions for all locations."""
        print("📊 Generating predictions...")
        
        # Start with coordinates
        predictions = features[['longitude', 'latitude']].copy()
        
        # Feature columns
        feature_cols = self.models[list(self.models.keys())[0]]['feature_names']
        X = features[feature_cols]
        
        for target, model_info in self.models.items():
            model = model_info['model']
            
            # Generate predictions
            pred = model.predict(X)
            predictions[f'{target}_predicted'] = pred
            
            # Add prediction uncertainty (confidence intervals)
            if hasattr(model, 'predict') and hasattr(model, 'estimators_'):
                # For RandomForest, calculate prediction std across trees
                tree_preds = np.array([tree.predict(X) for tree in model.estimators_])
                pred_std = np.std(tree_preds, axis=0)
            else:
                # For other models, use simple approach
                pred_std = np.std(pred) * 0.2
            
            predictions[f'{target}_std'] = pred_std
            predictions[f'{target}_lower_ci'] = pred - 1.96 * pred_std
            predictions[f'{target}_upper_ci'] = pred + 1.96 * pred_std
        
        print(f"Generated predictions for {len(predictions)} locations")
        self.predictions = predictions
        return predictions
    
    def create_results_visualization(self) -> plt.Figure:
        """Create and save results visualization."""
        print("📈 Creating visualization assets...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Geo Pipeline Mini - Soil Property Predictions', fontsize=16, fontweight='bold')
        
        # Plot 1: Organic Matter predictions map
        ax1 = axes[0, 0]
        scatter1 = ax1.scatter(
            self.predictions['longitude'], 
            self.predictions['latitude'],
            c=self.predictions['organic_matter_pct_predicted'],
            cmap='viridis',
            alpha=0.7,
            s=20
        )
        ax1.set_xlabel('Longitude')
        ax1.set_ylabel('Latitude')
        ax1.set_title('Organic Matter Predictions (%)')
        plt.colorbar(scatter1, ax=ax1, label='Organic Matter %')
        
        # Plot 2: pH predictions map
        ax2 = axes[0, 1]
        scatter2 = ax2.scatter(
            self.predictions['longitude'], 
            self.predictions['latitude'],
            c=self.predictions['ph_predicted'],
            cmap='RdYlBu_r',
            alpha=0.7,
            s=20
        )
        ax2.set_xlabel('Longitude')
        ax2.set_ylabel('Latitude')
        ax2.set_title('pH Predictions')
        plt.colorbar(scatter2, ax=ax2, label='pH')
        
        # Plot 3: Model performance metrics
        ax3 = axes[1, 0]
        targets = list(self.metrics.keys())
        r2_scores = [self.metrics[target]['r2_score'] for target in targets]
        
        bars = ax3.bar(targets, r2_scores, color=['green', 'blue', 'orange'])
        ax3.set_ylabel('R² Score')
        ax3.set_title('Model Performance (R² Score)')
        ax3.set_ylim([0, 1])
        
        # Add value labels on bars
        for bar, score in zip(bars, r2_scores):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{score:.3f}', ha='center', va='bottom')
        
        # Rotate x-axis labels for readability
        ax3.tick_params(axis='x', rotation=45)
        
        # Plot 4: CEC predictions map
        ax4 = axes[1, 1]
        scatter4 = ax4.scatter(
            self.predictions['longitude'], 
            self.predictions['latitude'],
            c=self.predictions['cec_meq_100g_predicted'],
            cmap='plasma',
            alpha=0.7,
            s=20
        )
        ax4.set_xlabel('Longitude')
        ax4.set_ylabel('Latitude')
        ax4.set_title('CEC Predictions (meq/100g)')
        plt.colorbar(scatter4, ax=ax4, label='CEC meq/100g')
        
        plt.tight_layout()
        
        # Save plot
        results_path = Path(self.config.assets_dir) / 'results.png'
        plt.savefig(results_path, dpi=300, bbox_inches='tight')
        print(f"📈 Results visualization saved to {results_path}")
        
        return fig
    
    def save_outputs(self):
        """Save predictions and metrics to output files."""
        print("💾 Saving pipeline outputs...")
        
        # Save predictions CSV
        predictions_path = Path(self.config.output_dir) / 'predictions.csv'
        self.predictions.to_csv(predictions_path, index=False)
        print(f"   Predictions saved: {predictions_path}")
        
        # Save metrics JSON
        metrics_path = Path(self.config.output_dir) / 'metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        print(f"   Metrics saved: {metrics_path}")
        
        return predictions_path, metrics_path
    
    def run_pipeline(self) -> Tuple[pd.DataFrame, Dict]:
        """Execute the complete soil property prediction pipeline."""
        print("🚀 Starting Geo Pipeline Mini")
        print("=" * 60)
        
        start_time = time.time()
        
        # Step 1: Generate synthetic data
        data = self.generate_synthetic_data(n_samples=1000)
        
        # Step 2: Engineer features
        features = self.engineer_features(data)
        
        # Step 3: Train models
        models, metrics = self.train_models(features)
        
        # Step 4: Generate predictions
        predictions = self.generate_predictions(features)
        
        # Step 5: Create visualizations
        fig = self.create_results_visualization()
        
        # Step 6: Save outputs
        pred_path, metrics_path = self.save_outputs()
        
        end_time = time.time()
        runtime = end_time - start_time
        
        print(f"\n✅ Pipeline completed successfully!")
        print(f"⏱️  Runtime: {runtime:.2f} seconds")
        print(f"\n📁 Generated Outputs:")
        print(f"   📊 {pred_path} ({len(predictions)} predictions)")
        print(f"   📈 {metrics_path} ({len(metrics)} models)")
        print(f"   🖼️  {Path(self.config.assets_dir) / 'results.png'}")
        
        print(f"\n🎯 Model Performance Summary:")
        for target, metric in metrics.items():
            print(f"   {target:20}: R² = {metric['r2_score']:.3f}, "
                  f"RMSE = {metric['rmse']:.3f} ({metric['model_type']})")
        
        return predictions, metrics


def main():
    """Main entry point for the pipeline."""
    pipeline = SoilPipeline()
    predictions, metrics = pipeline.run_pipeline()
    return predictions, metrics


if __name__ == "__main__":
    predictions, metrics = main()