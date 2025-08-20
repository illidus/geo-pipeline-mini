#!/usr/bin/env python3
"""
Unit tests for Geo Pipeline Mini - Updated to match current code structure
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
    def temp_dir(self):
        """Create temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def pipeline(self, temp_dir):
        """Create a test pipeline instance."""
        config = PipelineConfig(
            input_dir=f"{temp_dir}/data/raw",
            output_dir=f"{temp_dir}/data/processed"
        )
        # Change to temp directory for outputs
        original_cwd = os.getcwd()
        os.chdir(temp_dir)
        pipeline = SoilPipeline(config)
        yield pipeline
        os.chdir(original_cwd)
    
    def test_pipeline_initialization(self, pipeline):
        """Test pipeline initializes correctly."""
        assert pipeline.config is not None
        assert pipeline.config.random_state == 42
        assert pipeline.data == {}
        assert pipeline.models == {}
        assert pipeline.benchmarks == {}
    
    def test_directory_creation(self, pipeline):
        """Test that required directories are created."""
        assert Path(pipeline.config.input_dir).exists()
        assert Path(pipeline.config.output_dir).exists()
    
    def test_generate_sample_data(self, pipeline):
        """Test synthetic data generation."""
        data = pipeline.generate_sample_data()
        
        assert isinstance(data, dict)
        assert 'transform' in pipeline.features
        assert 'potassium' in data
        assert 'thorium' in data
        assert 'uranium' in data
        
        # Check data dimensions
        height, width = data['potassium'].shape
        assert height == 500
        assert width == 500
        
        # Check realistic value ranges
        assert data['potassium'].min() >= 0.1
        assert data['potassium'].max() <= 3.0
        assert data['thorium'].min() >= 2.0
        assert data['thorium'].max() <= 15.0
        
        # Check benchmarking
        assert 'data_generation' in pipeline.benchmarks
    
    def test_clean_data(self, pipeline):
        """Test data cleaning step."""
        pipeline.generate_sample_data()
        cleaned = pipeline.clean_data()
        
        # Should return processed data dictionary
        assert isinstance(cleaned, dict)
        assert 'potassium' in cleaned
        assert 'thorium' in cleaned
        assert 'uranium' in cleaned
        
        # Check benchmarking
        assert 'data_cleaning' in pipeline.benchmarks
    
    def test_engineer_features(self, pipeline):
        """Test feature engineering."""
        pipeline.generate_sample_data()
        pipeline.clean_data()
        features = pipeline.engineer_features()
        
        assert isinstance(features, dict)
        
        # Check that new features were created
        assert 'ndvi' in features
        assert 'clay_ratio' in features
        assert 'brightness' in features
        
        # Check benchmarking
        assert 'feature_engineering' in pipeline.benchmarks
    
    def test_create_soil_indices(self, pipeline):
        """Test soil indices creation and model training."""
        pipeline.generate_sample_data()
        pipeline.clean_data()
        pipeline.engineer_features()
        soil_indices = pipeline.create_soil_indices()
        
        # Check that soil properties were predicted
        assert 'SOM' in soil_indices  # Soil Organic Matter
        assert 'Clay' in soil_indices  # Clay content
        assert 'pH' in soil_indices    # pH
        
        # Check that models were trained
        assert 'SOM' in pipeline.models
        assert 'Clay' in pipeline.models
        assert 'pH' in pipeline.models
        
        # Check model metadata
        for model_name, model_info in pipeline.models.items():
            assert 'model' in model_info
            assert 'type' in model_info
            assert 'r2_score' in model_info
            assert model_info['type'] in ['Ridge', 'RandomForest']
            assert 0 <= model_info['r2_score'] <= 1
        
        # Check benchmarking
        assert 'modeling' in pipeline.benchmarks
    
    def test_export_geotiff(self, pipeline):
        """Test GeoTIFF export functionality."""
        pipeline.generate_sample_data()
        pipeline.clean_data()
        pipeline.engineer_features()
        pipeline.create_soil_indices()
        
        geotiff_path = pipeline.export_geotiff()
        
        # Check file was created
        assert Path(geotiff_path).exists()
        assert geotiff_path.endswith('.tif')
        
        # Check benchmarking
        assert 'export' in pipeline.benchmarks
    
    def test_export_predictions_csv(self, pipeline):
        """Test CSV predictions export."""
        pipeline.generate_sample_data()
        pipeline.clean_data()
        pipeline.engineer_features()
        pipeline.create_soil_indices()
        
        csv_path = pipeline.export_predictions_csv()
        
        # Check file was created
        assert Path(csv_path).exists()
        assert csv_path == "outputs/predictions.csv"
        
        # Check CSV contents
        df = pd.read_csv(csv_path)
        assert len(df) > 0
        assert 'x' in df.columns
        assert 'y' in df.columns
        
        # Check for actual/predicted columns
        expected_cols = [
            'soil_som_predicted', 'soil_clay_predicted', 'soil_ph_predicted'
        ]
        for col in expected_cols:
            assert col in df.columns
    
    def test_export_metrics_json(self, pipeline):
        """Test metrics JSON export."""
        pipeline.generate_sample_data()
        pipeline.clean_data()
        pipeline.engineer_features()
        pipeline.create_soil_indices()
        
        metrics_path = pipeline.export_metrics_json()
        
        # Check file was created
        assert Path(metrics_path).exists()
        assert metrics_path == "outputs/metrics.json"
        
        # Check JSON contents
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
        assert 'model_performance' in metrics
        assert 'processing_times' in metrics
        assert 'data_summary' in metrics
        
        # Check model performance structure
        perf = metrics['model_performance']
        assert 'soil_som' in perf
        assert 'soil_clay' in perf
        assert 'soil_ph' in perf
        
        for prop_metrics in perf.values():
            assert 'r2_score' in prop_metrics
            assert 'model_type' in prop_metrics
    
    def test_create_results_visualization(self, pipeline):
        """Test results visualization creation."""
        pipeline.generate_sample_data()
        pipeline.clean_data()
        pipeline.engineer_features()
        pipeline.create_soil_indices()
        
        png_path = pipeline.create_results_visualization()
        
        # Check file was created
        assert Path(png_path).exists()
        assert png_path == "assets/results.png"
        assert Path(png_path).suffix == '.png'
    
    def test_full_pipeline_integration(self, pipeline):
        """Test the complete pipeline runs without errors."""
        summary = pipeline.run_pipeline()
        
        # Check pipeline completed successfully
        assert isinstance(summary, dict)
        assert summary['pipeline_status'] == 'completed'
        
        # Check all required outputs exist
        assert Path("outputs/predictions.csv").exists()
        assert Path("outputs/metrics.json").exists()
        assert Path("assets/results.png").exists()
        
        # Check GeoTIFF output
        assert 'output_geotiff' in summary
        assert Path(summary['output_geotiff']).exists()
        
        # Check summary structure
        assert 'model_performance' in summary
        assert 'benchmarks' in summary
        assert 'data_dimensions' in summary
        
        # Check data dimensions
        dims = summary['data_dimensions']
        assert dims['height'] == 500
        assert dims['width'] == 500
        assert dims['total_pixels'] == 500 * 500
    
    def test_deterministic_behavior(self, temp_dir):
        """Test that pipeline produces deterministic results."""
        # Create two identical pipelines
        config1 = PipelineConfig(
            input_dir=f"{temp_dir}/input1",
            output_dir=f"{temp_dir}/output1",
            random_state=42
        )
        config2 = PipelineConfig(
            input_dir=f"{temp_dir}/input2", 
            output_dir=f"{temp_dir}/output2",
            random_state=42
        )
        
        original_cwd = os.getcwd()
        
        try:
            # Run first pipeline
            os.chdir(temp_dir)
            pipeline1 = SoilPipeline(config1)
            pipeline1.generate_sample_data()
            data1 = pipeline1.data
            
            # Run second pipeline
            pipeline2 = SoilPipeline(config2)
            pipeline2.generate_sample_data() 
            data2 = pipeline2.data
            
            # Should produce identical results due to same random seed
            assert np.array_equal(data1['potassium'], data2['potassium'])
            assert np.array_equal(data1['thorium'], data2['thorium'])
            assert np.array_equal(data1['uranium'], data2['uranium'])
            
        finally:
            os.chdir(original_cwd)
    
    def test_benchmark_tracking(self, pipeline):
        """Test that benchmarks are properly tracked."""
        pipeline.run_pipeline()
        
        # Check that all expected benchmarks exist
        expected_benchmarks = [
            'data_generation', 'data_cleaning', 'feature_engineering',
            'modeling', 'export', 'total_pipeline'
        ]
        
        for benchmark in expected_benchmarks:
            assert benchmark in pipeline.benchmarks
            assert pipeline.benchmarks[benchmark] > 0  # Should take some time


if __name__ == '__main__':
    pytest.main([__file__, '-v'])