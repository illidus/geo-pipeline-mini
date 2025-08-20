"""
Unit tests for the Geo Pipeline Mini
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import shutil
from pathlib import Path
import json

from src.pipeline.soil_pipeline import SoilPipeline


class TestSoilPipeline:
    
    @pytest.fixture
    def temp_dirs(self):
        """Create temporary directories for testing."""
        temp_dir = tempfile.mkdtemp()
        data_dir = Path(temp_dir) / "data"
        output_dir = Path(temp_dir) / "output"
        
        data_dir.mkdir(parents=True)
        
        # Create sample data files
        field_data = pd.DataFrame({
            'x': [0, 1, 0, 1],
            'y': [0, 0, 1, 1],
            'elevation': [100.0, 101.0, 99.0, 102.0],
            'gamma_k': [1.2, 1.3, 1.1, 1.4],
            'gamma_th': [8.5, 8.7, 8.2, 9.1],
            'gamma_u': [2.1, 2.0, 2.3, 1.9],
            'ndvi': [0.65, 0.68, 0.62, 0.71],
            'soil_moisture': [0.25, 0.24, 0.26, 0.23],
            'temperature': [18.5, 18.7, 18.3, 19.1]
        })
        
        soil_targets = pd.DataFrame({
            'x': [0, 1, 0, 1],
            'y': [0, 0, 1, 1],
            'soil_organic_matter': [3.2, 3.4, 2.9, 3.6],
            'clay_content': [24.5, 25.1, 23.8, 26.2],
            'ph': [6.8, 6.9, 6.7, 7.0]
        })
        
        field_data.to_csv(data_dir / "demo_field.csv", index=False)
        soil_targets.to_csv(data_dir / "soil_targets.csv", index=False)
        
        yield str(data_dir), str(output_dir)
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    def test_pipeline_initialization(self, temp_dirs):
        """Test pipeline initialization."""
        data_dir, output_dir = temp_dirs
        pipeline = SoilPipeline(data_dir, output_dir)
        
        assert pipeline.data_dir == Path(data_dir)
        assert pipeline.output_dir == Path(output_dir)
        assert Path(output_dir).exists()
    
    def test_load_data(self, temp_dirs):
        """Test data loading."""
        data_dir, output_dir = temp_dirs
        pipeline = SoilPipeline(data_dir, output_dir)
        
        field_data, soil_targets = pipeline.load_data()
        
        assert isinstance(field_data, pd.DataFrame)
        assert isinstance(soil_targets, pd.DataFrame)
        assert len(field_data) == 4
        assert len(soil_targets) == 4
        
        # Check required columns
        field_cols = ['x', 'y', 'elevation', 'gamma_k', 'gamma_th', 'gamma_u', 'ndvi']
        for col in field_cols:
            assert col in field_data.columns
        
        target_cols = ['x', 'y', 'soil_organic_matter', 'clay_content', 'ph']
        for col in target_cols:
            assert col in soil_targets.columns
    
    def test_clean_data(self, temp_dirs):
        """Test data cleaning."""
        data_dir, output_dir = temp_dirs
        pipeline = SoilPipeline(data_dir, output_dir)
        
        field_data, soil_targets = pipeline.load_data()
        field_clean, targets_clean = pipeline.clean_data(field_data, soil_targets)
        
        # Should have same number of rows (no missing data in test data)
        assert len(field_clean) == len(field_data)
        assert len(targets_clean) == len(soil_targets)
        
        # Check that gamma columns are normalized (mean ≈ 0, std ≈ 1)
        gamma_cols = ['gamma_k', 'gamma_th', 'gamma_u']
        for col in gamma_cols:
            assert abs(field_clean[col].mean()) < 0.1  # Close to 0
            assert abs(field_clean[col].std() - 1.0) < 0.1  # Close to 1
    
    def test_clean_data_with_missing_values(self, temp_dirs):
        """Test data cleaning with missing values."""
        data_dir, output_dir = temp_dirs
        pipeline = SoilPipeline(data_dir, output_dir)
        
        field_data, soil_targets = pipeline.load_data()
        
        # Introduce missing values
        field_data.loc[0, 'gamma_k'] = np.nan
        soil_targets.loc[1, 'ph'] = np.nan
        
        field_clean, targets_clean = pipeline.clean_data(field_data, soil_targets)
        
        # Should have fewer rows
        assert len(field_clean) == 3  # Removed 1 row
        assert len(targets_clean) == 3  # Removed 1 row
        assert not field_clean.isnull().any().any()
        assert not targets_clean.isnull().any().any()
    
    def test_feature_engineering(self, temp_dirs):
        """Test feature engineering."""
        data_dir, output_dir = temp_dirs
        pipeline = SoilPipeline(data_dir, output_dir)
        
        field_data, _ = pipeline.load_data()
        field_clean, _ = pipeline.clean_data(field_data, _)
        features = pipeline.feature_engineer(field_clean)
        
        assert isinstance(features, pd.DataFrame)
        assert len(features) == len(field_clean)
        
        # Check for engineered features
        expected_features = [
            'elev_mean', 'elev_std', 'gamma_k_mean', 'gamma_th_mean', 'gamma_u_mean',
            'gamma_ratio_k_th', 'gamma_ratio_k_u', 'temp_elev_interaction', 'moisture_ndvi_product'
        ]
        
        for feature in expected_features:
            assert feature in features.columns
        
        # Check that ratios are computed correctly (allow negative values after normalization)
        assert not features['gamma_ratio_k_th'].isnull().any()
        assert not features['gamma_ratio_k_u'].isnull().any()
        assert np.isfinite(features['gamma_ratio_k_th']).all()
        assert np.isfinite(features['gamma_ratio_k_u']).all()
    
    def test_train_models(self, temp_dirs):
        """Test model training."""
        data_dir, output_dir = temp_dirs
        pipeline = SoilPipeline(data_dir, output_dir)
        
        field_data, soil_targets = pipeline.load_data()
        field_clean, targets_clean = pipeline.clean_data(field_data, soil_targets)
        features = pipeline.feature_engineer(field_clean)
        
        models, predictions = pipeline.train_models(features, targets_clean)
        
        # Check models structure
        assert isinstance(models, dict)
        target_properties = ['soil_organic_matter', 'clay_content', 'ph']
        
        for prop in target_properties:
            assert prop in models
            assert 'model' in models[prop]
            assert 'type' in models[prop]
            assert 'r2_score' in models[prop]
            assert 'rmse' in models[prop]
            assert models[prop]['type'] in ['Ridge', 'RandomForest']
        
        # Check predictions structure
        assert isinstance(predictions, pd.DataFrame)
        assert len(predictions) == len(features)
        
        for prop in target_properties:
            assert f"{prop}_predicted" in predictions.columns
            assert f"{prop}_actual" in predictions.columns
    
    def test_export_results(self, temp_dirs):
        """Test results export."""
        data_dir, output_dir = temp_dirs
        pipeline = SoilPipeline(data_dir, output_dir)
        
        # Create dummy models and predictions
        models = {
            'soil_organic_matter': {
                'type': 'Ridge',
                'r2_score': 0.85,
                'rmse': 0.15
            }
        }
        
        predictions = pd.DataFrame({
            'x': [0, 1],
            'y': [0, 1],
            'soil_organic_matter_predicted': [3.1, 3.3],
            'soil_organic_matter_actual': [3.2, 3.4]
        })
        
        pred_file = pipeline.export_results(models, predictions)
        
        # Check that files are created
        assert Path(pred_file).exists()
        assert Path(output_dir) / "metrics.json" == Path(pipeline.output_dir) / "metrics.json"
        assert (Path(output_dir) / "metrics.json").exists()
        
        # Verify predictions file content
        saved_predictions = pd.read_csv(pred_file)
        assert len(saved_predictions) == len(predictions)
        assert 'soil_organic_matter_predicted' in saved_predictions.columns
        
        # Verify metrics file content
        with open(Path(output_dir) / "metrics.json", 'r') as f:
            metrics = json.load(f)
        
        assert 'model_performance' in metrics
        assert 'processing_times' in metrics
        assert 'data_summary' in metrics
    
    def test_run_pipeline_success(self, temp_dirs):
        """Test successful pipeline execution."""
        data_dir, output_dir = temp_dirs
        pipeline = SoilPipeline(data_dir, output_dir)
        
        result = pipeline.run_pipeline()
        
        assert result['status'] == 'completed'
        assert 'output_file' in result
        assert 'avg_r2_score' in result
        assert 'processing_time' in result
        assert 'models' in result
        
        # Check that output files exist
        assert Path(result['output_file']).exists()
        assert (Path(output_dir) / "metrics.json").exists()
    
    def test_run_pipeline_missing_data(self, temp_dirs):
        """Test pipeline with missing input files."""
        data_dir, output_dir = temp_dirs
        
        # Remove one of the required files
        (Path(data_dir) / "demo_field.csv").unlink()
        
        pipeline = SoilPipeline(data_dir, output_dir)
        result = pipeline.run_pipeline()
        
        assert result['status'] == 'failed'
        assert 'error' in result
    
    def test_feature_engineering_edge_cases(self, temp_dirs):
        """Test feature engineering with edge cases."""
        data_dir, output_dir = temp_dirs
        pipeline = SoilPipeline(data_dir, output_dir)
        
        # Create data with zero values that could cause division issues
        field_data = pd.DataFrame({
            'x': [0],
            'y': [0],
            'elevation': [100.0],
            'gamma_k': [0.0],  # Zero value
            'gamma_th': [0.0],  # Zero value
            'gamma_u': [1.0],
            'ndvi': [0.5],
            'soil_moisture': [0.2],
            'temperature': [18.0]
        })
        
        features = pipeline.feature_engineer(field_data)
        
        # Should handle division by zero gracefully
        assert not features['gamma_ratio_k_th'].isnull().any()
        assert not features['gamma_ratio_k_u'].isnull().any()
        assert np.isfinite(features['gamma_ratio_k_th']).all()
        assert np.isfinite(features['gamma_ratio_k_u']).all()
    
    def test_pipeline_data_consistency(self, temp_dirs):
        """Test that pipeline maintains data consistency throughout."""
        data_dir, output_dir = temp_dirs
        pipeline = SoilPipeline(data_dir, output_dir)
        
        result = pipeline.run_pipeline()
        
        # Load results and verify consistency
        predictions = pd.read_csv(result['output_file'])
        
        # Check that we have predictions for all target properties
        target_properties = ['soil_organic_matter', 'clay_content', 'ph']
        for prop in target_properties:
            assert f"{prop}_predicted" in predictions.columns
            assert f"{prop}_actual" in predictions.columns
            
            # Predictions should be reasonable (not NaN, not extremely large)
            pred_col = f"{prop}_predicted"
            assert not predictions[pred_col].isnull().any()
            assert (predictions[pred_col] > 0).all()  # Should be positive values
            assert (predictions[pred_col] < 100).all()  # Should be reasonable range