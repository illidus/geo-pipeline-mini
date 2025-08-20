"""
Unit tests for the Geo Pipeline Mini
"""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, Mock
import rasterio

from soil_pipeline import SoilPipeline, PipelineConfig


class TestPipelineConfig:
    def test_default_config(self):
        config = PipelineConfig()
        assert config.input_dir == "data/raw"
        assert config.output_dir == "data/processed"
        assert config.window_size == 5
        assert config.test_size == 0.2
        assert config.random_state == 42
    
    def test_custom_config(self):
        config = PipelineConfig(
            input_dir="custom/input",
            output_dir="custom/output",
            window_size=7,
            test_size=0.3,
            random_state=123
        )
        assert config.input_dir == "custom/input"
        assert config.output_dir == "custom/output"
        assert config.window_size == 7
        assert config.test_size == 0.3
        assert config.random_state == 123


class TestSoilPipeline:
    @pytest.fixture
    def temp_config(self):
        """Create temporary directory configuration for testing."""
        temp_dir = tempfile.mkdtemp()
        config = PipelineConfig(
            input_dir=f"{temp_dir}/input",
            output_dir=f"{temp_dir}/output"
        )
        yield config
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def pipeline(self, temp_config):
        return SoilPipeline(temp_config)
    
    def test_pipeline_initialization(self, pipeline, temp_config):
        assert pipeline.config == temp_config
        assert pipeline.data == {}
        assert pipeline.features == {}
        assert pipeline.models == {}
        assert pipeline.benchmarks == {}
        
        # Check directories are created
        assert Path(temp_config.input_dir).exists()
        assert Path(temp_config.output_dir).exists()
    
    def test_generate_sample_data(self, pipeline):
        data = pipeline.generate_sample_data()
        
        # Check that all expected bands are present
        expected_bands = [
            'potassium', 'thorium', 'uranium', 'total_count',
            'blue', 'red', 'nir', 'swir1', 'swir2', 'elevation'
        ]
        
        for band in expected_bands:
            assert band in data
            assert isinstance(data[band], np.ndarray)
            assert data[band].shape == (500, 500)  # Expected dimensions
        
        # Check metadata
        assert 'transform' in data
        assert 'bounds' in data
        
        # Check data ranges are realistic
        assert 0.1 <= np.min(data['potassium']) <= np.max(data['potassium']) <= 3.0
        assert 2.0 <= np.min(data['thorium']) <= np.max(data['thorium']) <= 15.0
        assert 0.5 <= np.min(data['uranium']) <= np.max(data['uranium']) <= 6.0
        assert 0.05 <= np.min(data['blue']) <= np.max(data['blue']) <= 0.35
        
        # Check benchmark was recorded
        assert 'data_generation' in pipeline.benchmarks
        assert pipeline.benchmarks['data_generation'] > 0
    
    def test_clean_data(self, pipeline):
        # First generate data
        pipeline.generate_sample_data()
        
        # Introduce some outliers manually for testing
        original_potassium = pipeline.data['potassium'].copy()
        pipeline.data['potassium'][0, 0] = 999  # Extreme outlier
        
        cleaned_data = pipeline.clean_data()
        
        # Check that outlier was handled
        assert cleaned_data['potassium'][0, 0] != 999
        assert 'data_cleaning' in pipeline.benchmarks
        
        # Check that reasonable values are preserved
        mean_diff = np.abs(np.mean(cleaned_data['potassium']) - np.mean(original_potassium))
        assert mean_diff < 0.1  # Should not change drastically
    
    def test_clean_data_without_generation(self, pipeline):
        with pytest.raises(ValueError, match="No data to clean"):
            pipeline.clean_data()
    
    def test_engineer_features(self, pipeline):
        # Setup: generate and clean data first
        pipeline.generate_sample_data()
        pipeline.clean_data()
        
        features = pipeline.engineer_features()
        
        # Check spectral indices are created
        spectral_indices = ['ndvi', 'soil_brightness', 'clay_index', 'iron_oxide']
        for index in spectral_indices:
            assert index in features
            assert isinstance(features[index], np.ndarray)
        
        # Check gamma ratios
        gamma_ratios = ['k_th_ratio', 'k_u_ratio', 'th_u_ratio']
        for ratio in gamma_ratios:
            assert ratio in features
            assert isinstance(features[ratio], np.ndarray)
        
        # Check terrain features
        terrain_features = ['slope', 'aspect', 'curvature', 'twi']
        for feature in terrain_features:
            assert feature in features
            assert isinstance(features[feature], np.ndarray)
        
        # Check windowed statistics
        window_size = pipeline.config.window_size
        windowed_features = [f'potassium_mean_{window_size}x{window_size}', f'ndvi_std_{window_size}x{window_size}']
        for feature in windowed_features:
            assert feature in features
        
        # Check NDVI calculation
        ndvi = features['ndvi']
        red = features['red']
        nir = features['nir']
        expected_ndvi = (nir - red) / (nir + red + 1e-8)
        np.testing.assert_allclose(ndvi, expected_ndvi, rtol=1e-6)
        
        # Check benchmark
        assert 'feature_engineering' in pipeline.benchmarks
    
    def test_engineer_features_without_data(self, pipeline):
        with pytest.raises(ValueError, match="No data available"):
            pipeline.engineer_features()
    
    def test_create_soil_indices(self, pipeline):
        # Setup: run pipeline up to feature engineering
        pipeline.generate_sample_data()
        pipeline.clean_data()
        pipeline.engineer_features()
        
        soil_indices = pipeline.create_soil_indices()
        
        # Check that soil properties are created
        expected_properties = ['SOM', 'Clay', 'pH']
        for prop in expected_properties:
            assert prop in soil_indices
            assert isinstance(soil_indices[prop], np.ndarray)
            assert soil_indices[prop].shape == (500, 500)
        
        # Check that models are trained
        assert len(pipeline.models) == 3
        for prop in expected_properties:
            assert prop in pipeline.models
            assert 'model' in pipeline.models[prop]
            assert 'r2_score' in pipeline.models[prop]
            assert 'type' in pipeline.models[prop]
            
            # R² should be reasonable (not negative, not too perfect)
            r2 = pipeline.models[prop]['r2_score']
            assert -0.5 <= r2 <= 1.0
        
        # Check realistic value ranges
        assert 0.5 <= np.min(soil_indices['SOM']) <= np.max(soil_indices['SOM']) <= 8.0
        assert 5 <= np.min(soil_indices['Clay']) <= np.max(soil_indices['Clay']) <= 60
        assert 4.5 <= np.min(soil_indices['pH']) <= np.max(soil_indices['pH']) <= 8.5
        
        # Check benchmark
        assert 'modeling' in pipeline.benchmarks
    
    def test_create_soil_indices_without_features(self, pipeline):
        with pytest.raises(ValueError, match="No features available"):
            pipeline.create_soil_indices()
    
    def test_export_geotiff(self, pipeline):
        # Setup: run pipeline up to modeling
        pipeline.generate_sample_data()
        pipeline.clean_data()
        pipeline.engineer_features()
        pipeline.create_soil_indices()
        
        output_file = "test_soil_properties.tif"
        geotiff_path = pipeline.export_geotiff(output_file)
        
        # Check that file was created
        assert Path(geotiff_path).exists()
        
        # Verify GeoTIFF structure
        with rasterio.open(geotiff_path) as src:
            assert src.count == 3  # Should have 3 bands (SOM, Clay, pH)
            assert src.height == 500
            assert src.width == 500
            assert src.dtype == 'float32'
            
            # Check band descriptions
            expected_bands = ['SOM', 'Clay', 'pH']
            for i, expected_band in enumerate(expected_bands, 1):
                band_desc = src.get_band_description(i)
                assert band_desc == expected_band
        
        # Check benchmark
        assert 'export' in pipeline.benchmarks
    
    def test_export_geotiff_without_soil_indices(self, pipeline):
        with pytest.raises(ValueError, match="Soil indices not available"):
            pipeline.export_geotiff()
    
    def test_run_pipeline(self, pipeline):
        results = pipeline.run_pipeline()
        
        # Check that results contain all expected sections
        assert 'pipeline_status' in results
        assert 'output_geotiff' in results
        assert 'soil_indices' in results
        assert 'model_performance' in results
        assert 'benchmarks' in results
        assert 'data_dimensions' in results
        
        assert results['pipeline_status'] == 'completed'
        assert len(results['soil_indices']) == 3
        assert len(results['model_performance']) == 3
        
        # Check that output file exists
        assert Path(results['output_geotiff']).exists()
        
        # Check that benchmark file was created
        benchmark_file = Path(pipeline.config.output_dir) / 'pipeline_benchmarks.json'
        assert benchmark_file.exists()
        
        # Verify all pipeline steps have benchmarks
        expected_benchmarks = [
            'data_generation', 'data_cleaning', 'feature_engineering', 
            'modeling', 'export', 'total_pipeline'
        ]
        for benchmark in expected_benchmarks:
            assert benchmark in results['benchmarks']
            assert results['benchmarks'][benchmark] > 0
    
    def test_pipeline_reproducibility(self, temp_config):
        """Test that pipeline produces consistent results with same random seed."""
        # Run pipeline twice with same config
        pipeline1 = SoilPipeline(temp_config)
        results1 = pipeline1.run_pipeline()
        
        pipeline2 = SoilPipeline(temp_config)
        results2 = pipeline2.run_pipeline()
        
        # Results should be very similar (allowing for small numerical differences)
        for prop in ['SOM', 'Clay', 'pH']:
            data1 = pipeline1.features[prop]
            data2 = pipeline2.features[prop]
            
            # Mean difference should be very small
            mean_diff = np.abs(np.mean(data1) - np.mean(data2))
            assert mean_diff < 0.001, f"Reproducibility failed for {prop}"
    
    def test_edge_cases(self, pipeline):
        """Test pipeline behavior with edge cases."""
        # Test with minimal data
        pipeline.generate_sample_data()
        
        # Artificially reduce data size for edge case testing
        original_shape = pipeline.data['potassium'].shape
        small_data = {}
        
        for key, value in pipeline.data.items():
            if isinstance(value, np.ndarray):
                # Take small subset
                small_data[key] = value[:50, :50]
            else:
                small_data[key] = value
        
        pipeline.data = small_data
        
        # Should still work with small datasets
        pipeline.clean_data()
        features = pipeline.engineer_features()
        
        # Verify small data still produces valid features
        assert features['ndvi'].shape == (50, 50)
        assert 'slope' in features
        assert features['slope'].shape == (50, 50)


if __name__ == "__main__":
    pytest.main([__file__])