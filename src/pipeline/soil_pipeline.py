"""
Geo Pipeline Mini - Transform gamma + satellite data into soil property indices
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


class SoilPipeline:
    """Minimal but real pipeline: ingest → clean → feature-engineer → model → export."""
    
    def __init__(self, data_dir: str, output_dir: str):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.timings = {}
        self.metrics = {}
        self.scaler = StandardScaler()
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Ingest demo data."""
        start_time = time.time()
        print("📡 Loading demo field data...")
        
        # Load field measurements 
        field_file = self.data_dir / "demo_field.csv"
        targets_file = self.data_dir / "soil_targets.csv"
        
        if not field_file.exists():
            raise FileNotFoundError(f"Field data not found: {field_file}")
        if not targets_file.exists():
            raise FileNotFoundError(f"Target data not found: {targets_file}")
        
        field_data = pd.read_csv(field_file)
        soil_targets = pd.read_csv(targets_file)
        
        print(f"  Loaded {len(field_data)} field measurements")
        print(f"  Loaded {len(soil_targets)} soil target values")
        
        self.timings['load'] = time.time() - start_time
        return field_data, soil_targets
    
    def clean_data(self, field_data: pd.DataFrame, soil_targets: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Clean and validate data."""
        start_time = time.time()
        print("🧹 Cleaning data...")
        
        # Check for missing values
        field_clean = field_data.dropna()
        targets_clean = soil_targets.dropna()
        
        print(f"  Removed {len(field_data) - len(field_clean)} rows with missing field data")
        print(f"  Removed {len(soil_targets) - len(targets_clean)} rows with missing target data")
        
        # Basic validation
        if field_clean.empty or targets_clean.empty:
            raise ValueError("No valid data remaining after cleaning")
        
        # Simple scaling for gamma data (in-place normalization)
        gamma_cols = ['gamma_k', 'gamma_th', 'gamma_u']
        for col in gamma_cols:
            if col in field_clean.columns:
                field_clean[col] = (field_clean[col] - field_clean[col].mean()) / field_clean[col].std()
        
        self.timings['clean'] = time.time() - start_time
        return field_clean, targets_clean
    
    def feature_engineer(self, field_data: pd.DataFrame) -> pd.DataFrame:
        """Engineer features with windowed stats and simple transforms."""
        start_time = time.time()
        print("⚙️ Engineering features...")
        
        features_df = field_data.copy()
        
        # Create spatial grid for windowing
        grid_x = sorted(field_data['x'].unique())
        grid_y = sorted(field_data['y'].unique())
        
        # Windowed statistics (simple 3x3 neighborhood)
        windowed_features = []
        
        for _, row in field_data.iterrows():
            x, y = row['x'], row['y']
            
            # Define 3x3 window
            x_neighbors = [xi for xi in grid_x if abs(xi - x) <= 1]
            y_neighbors = [yi for yi in grid_y if abs(yi - y) <= 1]
            
            # Get neighborhood data
            neighborhood = field_data[
                field_data['x'].isin(x_neighbors) & 
                field_data['y'].isin(y_neighbors)
            ]
            
            # Calculate windowed statistics
            features = {
                'x': x, 'y': y,
                'elev_mean': neighborhood['elevation'].mean(),
                'elev_std': neighborhood['elevation'].std(),
                'gamma_k_mean': neighborhood['gamma_k'].mean(),
                'gamma_th_mean': neighborhood['gamma_th'].mean(),
                'gamma_u_mean': neighborhood['gamma_u'].mean(),
                'ndvi_mean': neighborhood['ndvi'].mean(),
                'moisture_mean': neighborhood['soil_moisture'].mean(),
                'temp_mean': neighborhood['temperature'].mean(),
                # Simple math transforms
                'gamma_ratio_k_th': row['gamma_k'] / (row['gamma_th'] + 1e-8),
                'gamma_ratio_k_u': row['gamma_k'] / (row['gamma_u'] + 1e-8),
                'temp_elev_interaction': row['temperature'] * row['elevation'] / 1000,
                'moisture_ndvi_product': row['soil_moisture'] * row['ndvi']
            }
            windowed_features.append(features)
        
        features_df = pd.DataFrame(windowed_features)
        
        print(f"  Generated {len(features_df.columns) - 2} features")  # -2 for x,y coords
        self.timings['feature_engineer'] = time.time() - start_time
        return features_df
    
    def train_models(self, features_df: pd.DataFrame, targets_df: pd.DataFrame) -> Dict[str, Any]:
        """Train simple models (ridge/RF) for soil properties."""
        start_time = time.time()
        print("🤖 Training models...")
        
        # Merge features with targets
        merged = features_df.merge(targets_df, on=['x', 'y'], how='inner')
        
        if merged.empty:
            raise ValueError("No matching coordinates between features and targets")
        
        # Prepare feature matrix (exclude coordinates and targets)
        feature_cols = [col for col in merged.columns 
                       if col not in ['x', 'y', 'soil_organic_matter', 'clay_content', 'ph']]
        X = merged[feature_cols]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        models = {}
        target_properties = ['soil_organic_matter', 'clay_content', 'ph']
        
        for target_prop in target_properties:
            y = merged[target_prop]
            
            # Split data (small dataset, so use 80/20 split)
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )
            
            # Train Ridge and Random Forest
            ridge = Ridge(alpha=1.0, random_state=42)
            rf = RandomForestRegressor(n_estimators=10, max_depth=3, random_state=42)
            
            ridge.fit(X_train, y_train)
            rf.fit(X_train, y_train)
            
            # Evaluate both models
            ridge_pred = ridge.predict(X_test)
            rf_pred = rf.predict(X_test)
            
            ridge_r2 = r2_score(y_test, ridge_pred)
            rf_r2 = r2_score(y_test, rf_pred)
            
            # Select best model
            if rf_r2 > ridge_r2:
                best_model = rf
                best_r2 = rf_r2
                model_type = "RandomForest"
            else:
                best_model = ridge
                best_r2 = ridge_r2
                model_type = "Ridge"
            
            models[target_prop] = {
                'model': best_model,
                'type': model_type,
                'r2_score': best_r2,
                'rmse': np.sqrt(mean_squared_error(y_test, best_model.predict(X_test)))
            }
            
            print(f"  {target_prop}: {model_type}, R² = {best_r2:.3f}")
        
        # Generate predictions for full dataset
        predictions = {}
        for target_prop, model_info in models.items():
            pred_values = model_info['model'].predict(X_scaled)
            predictions[target_prop] = pred_values
        
        # Create predictions dataframe
        pred_df = merged[['x', 'y']].copy()
        for target_prop, pred_values in predictions.items():
            pred_df[f"{target_prop}_predicted"] = pred_values
            pred_df[f"{target_prop}_actual"] = merged[target_prop]
        
        self.timings['train'] = time.time() - start_time
        return models, pred_df
    
    def export_results(self, models: Dict[str, Any], predictions_df: pd.DataFrame) -> str:
        """Export predictions and metrics."""
        start_time = time.time()
        print("💾 Exporting results...")
        
        # Save predictions
        pred_file = self.output_dir / "predictions.csv"
        predictions_df.to_csv(pred_file, index=False)
        
        # Save metrics
        metrics = {
            'model_performance': {
                prop: {
                    'model_type': info['type'],
                    'r2_score': float(info['r2_score']),
                    'rmse': float(info['rmse'])
                }
                for prop, info in models.items()
            },
            'processing_times': self.timings,
            'data_summary': {
                'total_samples': len(predictions_df),
                'features_engineered': len([col for col in predictions_df.columns if col not in ['x', 'y']]) // 2,
                'target_properties': len(models)
            }
        }
        
        metrics_file = self.output_dir / "metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        self.timings['export'] = time.time() - start_time
        
        print(f"  Predictions saved: {pred_file}")
        print(f"  Metrics saved: {metrics_file}")
        
        return str(pred_file)
    
    def run_pipeline(self) -> Dict[str, Any]:
        """Run the complete pipeline."""
        print("🌍 Geo Pipeline Mini - Soil Property Analysis")
        print("=" * 50)
        
        total_start = time.time()
        
        try:
            # Execute pipeline steps
            field_data, soil_targets = self.load_data()
            field_clean, targets_clean = self.clean_data(field_data, soil_targets)
            features = self.feature_engineer(field_clean)
            models, predictions = self.train_models(features, targets_clean)
            pred_file = self.export_results(models, predictions)
            
            total_time = time.time() - total_start
            self.timings['total'] = total_time
            
            # Summary
            avg_r2 = np.mean([info['r2_score'] for info in models.values()])
            
            print(f"\n✅ Pipeline completed in {total_time:.2f}s")
            print(f"📊 Average R² score: {avg_r2:.3f}")
            print(f"📁 Results saved to: {self.output_dir}")
            
            return {
                'status': 'completed',
                'output_file': pred_file,
                'avg_r2_score': avg_r2,
                'processing_time': total_time,
                'models': {prop: info['type'] for prop, info in models.items()}
            }
            
        except Exception as e:
            print(f"❌ Pipeline failed: {e}")
            return {'status': 'failed', 'error': str(e)}


def create_parser():
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        description="Geo Pipeline Mini - Transform field data into soil property predictions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  %(prog)s run_pipeline --data data/demo --out outputs
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Run pipeline command
    run_parser = subparsers.add_parser('run_pipeline', help='Run the complete soil analysis pipeline')
    run_parser.add_argument('--data', required=True, help='Path to input data directory')
    run_parser.add_argument('--out', required=True, help='Path to output directory')
    
    return parser


def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    if args.command == 'run_pipeline':
        pipeline = SoilPipeline(args.data, args.out)
        result = pipeline.run_pipeline()
        return 0 if result['status'] == 'completed' else 1
    else:
        print(f"Unknown command: {args.command}")
        return 1


if __name__ == '__main__':
    sys.exit(main())