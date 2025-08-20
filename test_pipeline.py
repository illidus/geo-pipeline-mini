#!/usr/bin/env python3
"""
Unit tests for Geo Pipeline Mini
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import shutil
import json
import os

from soil_pipeline import SoilPipeline, PipelineConfig


class TestGeoPipelineMini:
    """Test suite for the geo pipeline."""
    
    @pytest.fixture
    def pipeline(self):
        """Create a test pipeline instance."""
        config = PipelineConfig(
            output_dir="test_outputs",
            assets_dir="test_assets",
            random_state=42
        )
        return SoilPipeline(config)
    
    @pytest.fixture
    def sample_data(self, pipeline):
        """Generate sample data for testing."""
        return pipeline.generate_synthetic_data(n_samples=100)
    
    def test_pipeline_initialization(self, pipeline):
        """Test pipeline initializes correctly."""
        assert pipeline.config is not None
        assert pipeline.config.random_state == 42
        assert pipeline.data == {}
        assert pipeline.models == {}
        assert pipeline.metrics == {}
    
    def test_directory_creation(self, pipeline):
        """Test that output directories are created."""
        assert Path(pipeline.config.output_dir).exists()
        assert Path(pipeline.config.assets_dir).exists()
    
    def test_generate_synthetic_data(self, pipeline):
        """Test synthetic data generation."""
        data = pipeline.generate_synthetic_data(n_samples=50)
        
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 50
        
        # Check required columns exist
        required_cols = [
            'longitude', 'latitude', 'potassium_pct', 'uranium_ppm', 
            'thorium_ppm', 'ndvi', 'ndwi', 'clay_index', 'elevation',
            'organic_matter_pct', 'ph', 'cec_meq_100g'
        ]
        for col in required_cols:
            assert col in data.columns
        
        # Check realistic value ranges
        assert data['organic_matter_pct'].min() >= 0.5
        assert data['organic_matter_pct'].max() <= 8.0
        assert data['ph'].min() >= 4.5
        assert data['ph'].max() <= 8.5
        assert data['longitude'].min() >= -100
        assert data['longitude'].max() <= -80
        assert data['latitude'].min() >= 35
        assert data['latitude'].max() <= 45
    
    def test_engineer_features(self, pipeline, sample_data):
        """Test feature engineering."""
        features = pipeline.engineer_features(sample_data)
        
        assert len(features) == len(sample_data)
        assert features.shape[1] > sample_data.shape[1]  # Should have more columns
        
        # Check engineered features exist
        engineered_features = [
            'k_th_ratio', 'u_k_ratio', 'th_k_ratio', 'u_th_ratio',
            'gamma_vigor', 'moisture_gamma', 'elevation_ndvi',
            'soil_complexity', 'fertility_index', 'lat_lon_interaction',
            'distance_to_center'
        ]
        for feature in engineered_features:
            assert feature in features.columns
        
        # Check no infinite values
        assert not np.isinf(features['k_th_ratio']).any()
        assert not np.isinf(features['u_k_ratio']).any()
    
    def test_train_models(self, pipeline, sample_data):
        """Test model training."""
        features = pipeline.engineer_features(sample_data)
        models, metrics = pipeline.train_models(features)
        
        # Check models exist for all targets
        expected_targets = ['organic_matter_pct', 'ph', 'cec_meq_100g']
        assert set(models.keys()) == set(expected_targets)
        assert set(metrics.keys()) == set(expected_targets)
        
        # Check model structure
        for target in expected_targets:
            assert 'model' in models[target]
            assert 'feature_names' in models[target]
            assert 'model_type' in models[target]
            
            # Check metrics
            assert 'r2_score' in metrics[target]
            assert 'rmse' in metrics[target]
            assert 'mae' in metrics[target]
            assert metrics[target]['r2_score'] >= 0  # R² should be reasonable
    
    def test_generate_predictions(self, pipeline, sample_data):
        """Test prediction generation."""
        features = pipeline.engineer_features(sample_data)
        models, metrics = pipeline.train_models(features)
        predictions = pipeline.generate_predictions(features)
        
        assert len(predictions) == len(features)
        assert 'longitude' in predictions.columns
        assert 'latitude' in predictions.columns
        
        # Check predictions exist for all targets
        targets = ['organic_matter_pct', 'ph', 'cec_meq_100g']
        for target in targets:
            assert f'{target}_predicted' in predictions.columns
            assert f'{target}_std' in predictions.columns
            assert f'{target}_lower_ci' in predictions.columns
            assert f'{target}_upper_ci' in predictions.columns
        
        # Check prediction ranges are reasonable
        assert predictions['organic_matter_pct_predicted'].min() >= 0
        assert predictions['ph_predicted'].min() >= 3  # Realistic soil pH range
        assert predictions['ph_predicted'].max() <= 10
    
    def test_create_results_visualization(self, pipeline, sample_data):
        """Test visualization creation."""
        features = pipeline.engineer_features(sample_data)
        models, metrics = pipeline.train_models(features)
        predictions = pipeline.generate_predictions(features)
        
        fig = pipeline.create_results_visualization()
        
        # Check that assets file is created
        results_path = Path(pipeline.config.assets_dir) / 'results.png'
        assert results_path.exists()
        
        # Check figure has correct structure
        assert len(fig.axes) == 4  # Should have 4 subplots
    
    def test_save_outputs(self, pipeline, sample_data):
        """Test output saving."""
        features = pipeline.engineer_features(sample_data)
        models, metrics = pipeline.train_models(features)
        predictions = pipeline.generate_predictions(features)
        
        pred_path, metrics_path = pipeline.save_outputs()
        
        # Check files exist
        assert pred_path.exists()
        assert metrics_path.exists()
        
        # Check file contents
        saved_predictions = pd.read_csv(pred_path)
        assert len(saved_predictions) == len(predictions)
        
        with open(metrics_path, 'r') as f:
            saved_metrics = json.load(f)
        assert 'organic_matter_pct' in saved_metrics
        assert 'ph' in saved_metrics
        assert 'cec_meq_100g' in saved_metrics
    
    def test_full_pipeline_integration(self, pipeline):
        """Test the complete pipeline runs without errors."""
        predictions, metrics = pipeline.run_pipeline()
        
        # Check outputs
        assert isinstance(predictions, pd.DataFrame)
        assert isinstance(metrics, dict)
        assert len(predictions) > 0
        assert len(metrics) > 0
        
        # Check required files exist
        assert Path(pipeline.config.output_dir, 'predictions.csv').exists()
        assert Path(pipeline.config.output_dir, 'metrics.json').exists()
        assert Path(pipeline.config.assets_dir, 'results.png').exists()
    
    def test_deterministic_behavior(self):
        """Test that pipeline produces deterministic results."""
        config1 = PipelineConfig(random_state=42, output_dir="test1", assets_dir="test1")
        config2 = PipelineConfig(random_state=42, output_dir="test2", assets_dir="test2")
        
        pipeline1 = SoilPipeline(config1)
        pipeline2 = SoilPipeline(config2)
        
        data1 = pipeline1.generate_synthetic_data(n_samples=100)
        data2 = pipeline2.generate_synthetic_data(n_samples=100)
        
        # Should be identical due to same random seed
        pd.testing.assert_frame_equal(data1, data2)
    
    def teardown_method(self, method):
        """Clean up test directories after each test."""
        test_dirs = ['test_outputs', 'test_assets', 'test1', 'test2']
        for dir_name in test_dirs:
            if Path(dir_name).exists():
                shutil.rmtree(dir_name)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])