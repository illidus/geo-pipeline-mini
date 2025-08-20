"""
Geo Pipeline Mini - Transform gamma + satellite data into soil property indices
"""

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from rasterio.transform import from_bounds
# import geopandas as gpd  # Not used in this implementation
from pathlib import Path
import time
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
# import seaborn as sns  # Not used in this implementation


@dataclass
class PipelineConfig:
    """Configuration for the soil property pipeline."""
    input_dir: str = "data/raw"
    output_dir: str = "data/processed"
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
        self.benchmarks = {}
        
        # Create directories
        Path(self.config.input_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
    
    def generate_sample_data(self) -> Dict[str, np.ndarray]:
        """Generate sample gamma radiometric and satellite data."""
        print("Generating sample field data...")
        start_time = time.time()
        
        # Field dimensions (500x500 pixels, ~2.5km x 2.5km at 5m resolution)
        height, width = 500, 500
        
        # Create coordinate system (UTM-like)
        bounds = (300000, 4600000, 302500, 4602500)  # UTM coordinates
        transform = from_bounds(*bounds, width, height)
        
        # Generate gamma radiometric data (simulating airborne gamma survey)
        np.random.seed(42)
        
        # Potassium (K) - related to clay content and soil fertility
        k_base = np.random.normal(1.2, 0.3, (height, width))
        k_trend = np.linspace(0.8, 1.6, width)[np.newaxis, :] * np.ones((height, 1))
        potassium = np.clip(k_base * k_trend + np.random.normal(0, 0.1, (height, width)), 0.1, 3.0)
        
        # Thorium (Th) - related to clay minerals
        th_base = np.random.normal(8.5, 2.0, (height, width))
        th_pattern = 0.3 * np.sin(np.linspace(0, 4*np.pi, height))[:, np.newaxis] * np.ones((1, width))
        thorium = np.clip(th_base + th_pattern + np.random.normal(0, 0.5, (height, width)), 2.0, 15.0)
        
        # Uranium (U) - related to organic matter and phosphates
        u_base = np.random.normal(2.1, 0.4, (height, width))
        u_spots = np.random.choice([0, 1], size=(height, width), p=[0.95, 0.05])
        uranium = np.clip(u_base + u_spots * np.random.normal(1.5, 0.3, (height, width)), 0.5, 6.0)
        
        # Total count (TC) - overall gamma activity
        total_count = potassium * 310 + thorium * 4.1 + uranium * 9.3 + np.random.normal(50, 10, (height, width))
        
        # Satellite data (simulating Landsat-8 like bands)
        # Band 1 (Coastal/Blue) - related to soil organic matter
        blue = np.random.beta(2, 3, (height, width)) * 0.3 + 0.05
        
        # Band 4 (Red) - vegetation and soil brightness
        red = np.random.beta(1.5, 2, (height, width)) * 0.4 + 0.1
        
        # Band 5 (NIR) - vegetation content
        nir = red * np.random.uniform(1.5, 3.0, (height, width)) + np.random.normal(0, 0.05, (height, width))
        nir = np.clip(nir, 0.1, 0.8)
        
        # Band 6 (SWIR1) - soil moisture and clay content
        swir1 = np.random.beta(2, 4, (height, width)) * 0.5 + 0.1
        
        # Band 7 (SWIR2) - clay minerals
        swir2 = swir1 * np.random.uniform(0.7, 1.2, (height, width)) + np.random.normal(0, 0.03, (height, width))
        swir2 = np.clip(swir2, 0.05, 0.4)
        
        # Digital Elevation Model (DEM)
        x_coords = np.linspace(0, width-1, width)
        y_coords = np.linspace(0, height-1, height)
        X, Y = np.meshgrid(x_coords, y_coords)
        
        # Create realistic terrain with multiple hills and valleys
        elevation = (
            20 * np.sin(X * 0.01) * np.cos(Y * 0.008) +
            15 * np.sin(X * 0.005 + Y * 0.007) +
            10 * np.cos(X * 0.015) * np.sin(Y * 0.012) +
            np.random.normal(0, 2, (height, width))
        )
        elevation = elevation + 350  # Base elevation 350m
        
        self.data = {
            'potassium': potassium,
            'thorium': thorium,
            'uranium': uranium,
            'total_count': total_count,
            'blue': blue,
            'red': red,
            'nir': nir,
            'swir1': swir1,
            'swir2': swir2,
            'elevation': elevation,
            'transform': transform,
            'bounds': bounds
        }
        
        self.benchmarks['data_generation'] = time.time() - start_time
        print(f"Generated sample data in {self.benchmarks['data_generation']:.2f} seconds")
        return self.data
    
    def clean_data(self) -> Dict[str, np.ndarray]:
        """Clean and validate raster data."""
        print("Cleaning raster data...")
        start_time = time.time()
        
        if not self.data:
            raise ValueError("No data to clean. Run generate_sample_data() first.")
        
        cleaned_data = {}
        
        for band_name, band_data in self.data.items():
            if band_name in ['transform', 'bounds']:
                cleaned_data[band_name] = band_data
                continue
            
            # Remove outliers (values beyond 3 standard deviations)
            mean_val = np.mean(band_data)
            std_val = np.std(band_data)
            
            # Create mask for valid data
            valid_mask = np.abs(band_data - mean_val) <= 3 * std_val
            
            # Fill outliers with median of surrounding valid pixels
            cleaned_band = band_data.copy()
            if not np.all(valid_mask):
                # Simple median filtering for outliers
                from scipy import ndimage
                median_filtered = ndimage.median_filter(band_data, size=3)
                cleaned_band[~valid_mask] = median_filtered[~valid_mask]
            
            # Ensure no NaN or infinite values
            cleaned_band = np.nan_to_num(cleaned_band, nan=np.median(band_data))
            
            cleaned_data[band_name] = cleaned_band
        
        self.data = cleaned_data
        self.benchmarks['data_cleaning'] = time.time() - start_time
        print(f"Cleaned data in {self.benchmarks['data_cleaning']:.2f} seconds")
        return self.data
    
    def engineer_features(self) -> Dict[str, np.ndarray]:
        """Engineer features including windowed statistics and terrain features."""
        print("Engineering features...")
        start_time = time.time()
        
        if not self.data:
            raise ValueError("No data available. Run generate_sample_data() and clean_data() first.")
        
        features = {}
        height, width = self.data['potassium'].shape
        
        # Calculate spectral indices
        features['ndvi'] = (self.data['nir'] - self.data['red']) / (self.data['nir'] + self.data['red'] + 1e-8)
        features['soil_brightness'] = (self.data['red'] + self.data['swir1'] + self.data['swir2']) / 3
        features['clay_index'] = self.data['swir1'] / self.data['swir2']
        features['iron_oxide'] = self.data['red'] / self.data['blue']
        
        # Gamma ratios (standard in radiometric surveys)
        features['k_th_ratio'] = self.data['potassium'] / (self.data['thorium'] + 1e-8)
        features['k_u_ratio'] = self.data['potassium'] / (self.data['uranium'] + 1e-8)
        features['th_u_ratio'] = self.data['thorium'] / (self.data['uranium'] + 1e-8)
        
        # Terrain features
        elevation = self.data['elevation']
        
        # Calculate slope (gradient magnitude)
        grad_y, grad_x = np.gradient(elevation)
        features['slope'] = np.sqrt(grad_x**2 + grad_y**2)
        
        # Calculate aspect (slope direction)
        features['aspect'] = np.arctan2(grad_y, grad_x)
        
        # Calculate curvature (second derivative)
        grad_xx, grad_xy = np.gradient(grad_x)
        grad_yx, grad_yy = np.gradient(grad_y)
        features['curvature'] = -(grad_xx + grad_yy)
        
        # Topographic Wetness Index (TWI)
        # Simplified calculation for demonstration
        flow_accumulation = np.ones_like(elevation) * 100  # Simplified
        features['twi'] = np.log(flow_accumulation / (features['slope'] + 1e-8))
        
        # Windowed statistics for spatial context
        window_size = self.config.window_size
        half_window = window_size // 2
        
        for base_feature in ['potassium', 'thorium', 'uranium', 'ndvi', 'soil_brightness']:
            base_data = features.get(base_feature, self.data.get(base_feature))
            if base_data is None:
                continue
            
            # Initialize output arrays
            mean_array = np.zeros_like(base_data)
            std_array = np.zeros_like(base_data)
            
            # Calculate windowed statistics
            for i in range(height):
                for j in range(width):
                    # Define window bounds
                    i_start = max(0, i - half_window)
                    i_end = min(height, i + half_window + 1)
                    j_start = max(0, j - half_window)
                    j_end = min(width, j + half_window + 1)
                    
                    # Extract window
                    window_data = base_data[i_start:i_end, j_start:j_end]
                    
                    # Calculate statistics
                    mean_array[i, j] = np.mean(window_data)
                    std_array[i, j] = np.std(window_data)
            
            features[f'{base_feature}_mean_{window_size}x{window_size}'] = mean_array
            features[f'{base_feature}_std_{window_size}x{window_size}'] = std_array
        
        # Copy original bands
        for band in ['potassium', 'thorium', 'uranium', 'total_count', 'blue', 'red', 'nir', 'swir1', 'swir2', 'elevation']:
            features[band] = self.data[band]
        
        # Copy transform and bounds
        features['transform'] = self.data['transform']
        features['bounds'] = self.data['bounds']
        
        self.features = features
        self.benchmarks['feature_engineering'] = time.time() - start_time
        print(f"Engineered {len([k for k in features.keys() if k not in ['transform', 'bounds']])} features in {self.benchmarks['feature_engineering']:.2f} seconds")
        return features
    
    def create_soil_indices(self) -> Dict[str, np.ndarray]:
        """Create soil property indices using simple models."""
        print("Creating soil property indices...")
        start_time = time.time()
        
        if not self.features:
            raise ValueError("No features available. Run engineer_features() first.")
        
        # Prepare feature matrix (excluding metadata)
        feature_names = [k for k in self.features.keys() if k not in ['transform', 'bounds']]
        height, width = self.features['potassium'].shape
        
        # Flatten features for modeling
        X = np.stack([self.features[name].flatten() for name in feature_names], axis=1)
        
        # Create synthetic soil property targets based on known relationships
        # Soil Organic Matter (SOM) - related to vegetation and uranium
        som_target = (
            0.3 * self.features['ndvi'].flatten() +
            0.2 * self.features['uranium'].flatten() / 5.0 +  # Normalize uranium
            0.1 * (1 - self.features['soil_brightness'].flatten()) +
            np.random.normal(0, 0.05, X.shape[0])
        )
        som_target = np.clip(som_target, 0.5, 8.0)  # Reasonable SOM range 0.5-8%
        
        # Clay Content - related to thorium, K/Th ratio, and clay index
        clay_target = (
            0.4 * self.features['thorium'].flatten() / 15.0 +  # Normalize thorium
            0.3 * (1 / (self.features['k_th_ratio'].flatten() + 1e-8)) * 0.5 +
            0.2 * self.features['clay_index'].flatten() +
            np.random.normal(0, 0.03, X.shape[0])
        )
        clay_target = np.clip(clay_target, 5, 60)  # Clay content 5-60%
        
        # pH - inverse relationship with some gamma elements
        ph_target = (
            7.0 - 0.5 * (self.features['uranium'].flatten() / 5.0) +
            0.3 * (self.features['potassium'].flatten() / 2.0) +
            np.random.normal(0, 0.2, X.shape[0])
        )
        ph_target = np.clip(ph_target, 4.5, 8.5)  # pH range 4.5-8.5
        
        # Train models for each soil property
        models = {}
        soil_indices = {}
        
        for target_name, target_data in [('SOM', som_target), ('Clay', clay_target), ('pH', ph_target)]:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, target_data, test_size=self.config.test_size, random_state=self.config.random_state
            )
            
            # Train Ridge regression (faster, more stable)
            ridge_model = Ridge(alpha=1.0, random_state=self.config.random_state)
            ridge_model.fit(X_train, y_train)
            
            # Train Random Forest (potentially better performance)
            rf_model = RandomForestRegressor(
                n_estimators=50, max_depth=10, random_state=self.config.random_state, n_jobs=-1
            )
            rf_model.fit(X_train, y_train)
            
            # Evaluate models
            ridge_pred = ridge_model.predict(X_test)
            rf_pred = rf_model.predict(X_test)
            
            ridge_r2 = r2_score(y_test, ridge_pred)
            rf_r2 = r2_score(y_test, rf_pred)
            
            # Select best model
            if rf_r2 > ridge_r2:
                best_model = rf_model
                best_r2 = rf_r2
                model_type = "RandomForest"
            else:
                best_model = ridge_model
                best_r2 = ridge_r2
                model_type = "Ridge"
            
            # Predict full map
            predictions = best_model.predict(X).reshape(height, width)
            soil_indices[target_name] = predictions
            
            models[target_name] = {
                'model': best_model,
                'type': model_type,
                'r2_score': best_r2,
                'feature_names': feature_names
            }
            
            print(f"{target_name}: {model_type} model, R² = {best_r2:.3f}")
        
        self.models = models
        self.features.update(soil_indices)
        
        self.benchmarks['modeling'] = time.time() - start_time
        print(f"Created soil indices in {self.benchmarks['modeling']:.2f} seconds")
        return soil_indices
    
    def export_geotiff(self, output_name: str = "soil_properties.tif") -> str:
        """Export soil indices as multi-band GeoTIFF."""
        print(f"Exporting GeoTIFF: {output_name}")
        start_time = time.time()
        
        # Define soil property bands to export
        soil_bands = ['SOM', 'Clay', 'pH']
        
        if not all(band in self.features for band in soil_bands):
            raise ValueError("Soil indices not available. Run create_soil_indices() first.")
        
        output_path = Path(self.config.output_dir) / output_name
        height, width = self.features['SOM'].shape
        
        # Create GeoTIFF with multiple bands
        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=len(soil_bands),
            dtype=np.float32,
            crs='EPSG:32613',  # UTM Zone 13N
            transform=self.features['transform'],
            compress='lzw'
        ) as dst:
            for i, band_name in enumerate(soil_bands, 1):
                dst.write(self.features[band_name].astype(np.float32), i)
                dst.set_band_description(i, band_name)
        
        self.benchmarks['export'] = time.time() - start_time
        print(f"Exported GeoTIFF in {self.benchmarks['export']:.2f} seconds")
        return str(output_path)
    
    def export_predictions_csv(self) -> str:
        """Export predictions and actual values as CSV for analysis."""
        print("Creating predictions CSV...")
        
        # Create outputs directory
        outputs_dir = Path("outputs")
        outputs_dir.mkdir(exist_ok=True)
        
        # Sample a subset of pixels for CSV (avoid huge files)
        height, width = self.features['SOM'].shape
        sample_size = min(1000, height * width)  # Sample up to 1000 pixels
        
        # Create sampling indices
        np.random.seed(42)
        pixel_indices = np.random.choice(height * width, size=sample_size, replace=False)
        rows, cols = np.unravel_index(pixel_indices, (height, width))
        
        # Get coordinates using transform
        transform = self.features['transform']
        x_coords = []
        y_coords = []
        for row, col in zip(rows, cols):
            x, y = rasterio.transform.xy(transform, row, col)
            x_coords.append(x)
            y_coords.append(y)
        
        # Create predictions DataFrame
        predictions_data = {
            'x': x_coords,
            'y': y_coords,
        }
        
        # Add actual and predicted values for each soil property
        for prop_name in ['SOM', 'Clay', 'pH']:
            # Get actual values (targets used for training)
            actual_key = f'{prop_name.lower()}_target' if prop_name != 'SOM' else 'som_target'
            if actual_key in self.features:
                actual_values = self.features[actual_key].flatten()[pixel_indices]
                predictions_data[f'soil_{prop_name.lower()}_actual'] = actual_values
            
            # Get predicted values
            predicted_values = self.features[prop_name].flatten()[pixel_indices]
            predictions_data[f'soil_{prop_name.lower()}_predicted'] = predicted_values
        
        # Create DataFrame and save
        predictions_df = pd.DataFrame(predictions_data)
        csv_path = outputs_dir / "predictions.csv"
        predictions_df.to_csv(csv_path, index=False)
        
        print(f"Saved predictions CSV: {csv_path}")
        return str(csv_path)
    
    def export_metrics_json(self) -> str:
        """Export model performance metrics as JSON."""
        print("Creating metrics JSON...")
        
        # Create outputs directory
        outputs_dir = Path("outputs")
        outputs_dir.mkdir(exist_ok=True)
        
        # Prepare metrics in expected format
        metrics = {
            "model_performance": {},
            "processing_times": self.benchmarks,
            "data_summary": {
                "height": self.features['SOM'].shape[0],
                "width": self.features['SOM'].shape[1], 
                "total_pixels": self.features['SOM'].size
            }
        }
        
        # Add model performance for each soil property
        for prop_name, model_info in self.models.items():
            metrics["model_performance"][f"soil_{prop_name.lower()}"] = {
                "r2_score": model_info['r2_score'],
                "model_type": model_info['type']
            }
        
        # Save metrics
        metrics_path = outputs_dir / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"Saved metrics JSON: {metrics_path}")
        return str(metrics_path)
    
    def create_results_visualization(self) -> str:
        """Create results visualization and save as PNG."""
        print("Creating results visualization...")
        
        # Create assets directory
        assets_dir = Path("assets")
        assets_dir.mkdir(exist_ok=True)
        
        # Create a 2x2 subplot figure
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Geo Pipeline Mini - Soil Property Results', fontsize=16, fontweight='bold')
        
        # Plot 1: SOM (Soil Organic Matter) map
        ax1 = axes[0, 0]
        im1 = ax1.imshow(self.features['SOM'], cmap='viridis', aspect='equal')
        ax1.set_title('Soil Organic Matter (%)')
        ax1.set_xlabel('X (pixels)')
        ax1.set_ylabel('Y (pixels)')
        plt.colorbar(im1, ax=ax1, shrink=0.8)
        
        # Plot 2: Clay content map
        ax2 = axes[0, 1]
        im2 = ax2.imshow(self.features['Clay'], cmap='YlOrBr', aspect='equal')
        ax2.set_title('Clay Content (%)')
        ax2.set_xlabel('X (pixels)')
        ax2.set_ylabel('Y (pixels)')
        plt.colorbar(im2, ax=ax2, shrink=0.8)
        
        # Plot 3: pH map
        ax3 = axes[1, 0]
        im3 = ax3.imshow(self.features['pH'], cmap='RdYlBu_r', aspect='equal')
        ax3.set_title('Soil pH')
        ax3.set_xlabel('X (pixels)')
        ax3.set_ylabel('Y (pixels)')
        plt.colorbar(im3, ax=ax3, shrink=0.8)
        
        # Plot 4: Model performance comparison
        ax4 = axes[1, 1]
        properties = list(self.models.keys())
        r2_scores = [self.models[prop]['r2_score'] for prop in properties]
        model_types = [self.models[prop]['type'] for prop in properties]
        
        bars = ax4.bar(properties, r2_scores, color=['green', 'orange', 'blue'])
        ax4.set_title('Model Performance (R² Score)')
        ax4.set_ylabel('R² Score')
        ax4.set_ylim([0, 1])
        
        # Add value labels and model types
        for i, (bar, r2, model_type) in enumerate(zip(bars, r2_scores, model_types)):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{r2:.3f}\n({model_type})', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        # Save the figure
        results_path = assets_dir / "results.png"
        plt.savefig(results_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved results visualization: {results_path}")
        return str(results_path)

    def run_pipeline(self) -> Dict[str, any]:
        """Run the complete pipeline."""
        print("🌍 Geo Pipeline Mini - Soil Property Analysis")
        print("=" * 50)
        
        pipeline_start = time.time()
        
        # Execute pipeline steps
        self.generate_sample_data()
        self.clean_data()
        self.engineer_features()
        soil_indices = self.create_soil_indices()
        geotiff_path = self.export_geotiff()
        
        # Export outputs as claimed in README
        predictions_csv = self.export_predictions_csv()
        metrics_json = self.export_metrics_json()
        results_png = self.create_results_visualization()
        
        total_time = time.time() - pipeline_start
        self.benchmarks['total_pipeline'] = total_time
        
        # Generate summary
        summary = {
            'pipeline_status': 'completed',
            'output_geotiff': geotiff_path,
            'predictions_csv': predictions_csv,
            'metrics_json': metrics_json,
            'results_png': results_png,
            'soil_indices': list(soil_indices.keys()),
            'model_performance': {name: info['r2_score'] for name, info in self.models.items()},
            'benchmarks': self.benchmarks,
            'data_dimensions': {
                'height': self.features['SOM'].shape[0],
                'width': self.features['SOM'].shape[1],
                'total_pixels': self.features['SOM'].size
            }
        }
        
        # Save benchmark results
        benchmark_file = Path(self.config.output_dir) / 'pipeline_benchmarks.json'
        with open(benchmark_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n📊 Pipeline Summary")
        print("-" * 30)
        print(f"Total processing time: {total_time:.2f} seconds")
        print(f"Data dimensions: {summary['data_dimensions']['height']}x{summary['data_dimensions']['width']} pixels")
        print(f"Output GeoTIFF: {geotiff_path}")
        print(f"Predictions CSV: {predictions_csv}")
        print(f"Metrics JSON: {metrics_json}")
        print(f"Results PNG: {results_png}")
        
        print(f"\nModel Performance (R² scores):")
        for prop, r2 in summary['model_performance'].items():
            print(f"  {prop}: {r2:.3f}")
        
        print(f"\nBenchmark Results:")
        for step, duration in self.benchmarks.items():
            print(f"  {step}: {duration:.2f}s")
        
        return summary


if __name__ == "__main__":
    # Run the pipeline
    config = PipelineConfig()
    pipeline = SoilPipeline(config)
    
    results = pipeline.run_pipeline()
    
    print(f"\n✅ Pipeline completed successfully!")
    print(f"Check output directory: {config.output_dir}")