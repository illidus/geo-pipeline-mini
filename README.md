# Geo Pipeline Mini

Transform gamma radiometric + satellite data into soil property indices with machine learning.

## Features

- **Real Pipeline**: Processes gamma and satellite data into soil predictions
- **Multiple Outputs**: Generates `outputs/predictions.csv`, `outputs/metrics.json`, and `assets/results.png`
- **Machine Learning**: RandomForest and Ridge regression models
- **Comprehensive Testing**: Unit tests with full pipeline coverage

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the complete pipeline
python soil_pipeline.py

# Run tests
python -m pytest test_pipeline.py -v
```

## Pipeline Outputs

### 1. Predictions CSV (`outputs/predictions.csv`)
Location-based predictions with confidence intervals:
- Coordinates (longitude, latitude)
- Soil property predictions (organic matter %, pH, CEC)
- Confidence intervals (lower_ci, upper_ci)
- Prediction uncertainty (std)

### 2. Metrics JSON (`outputs/metrics.json`)
Model performance metrics:
- R² scores for each soil property
- RMSE and MAE values
- Model types (RandomForest vs Ridge)
- Training statistics

### 3. Results Visualization (`assets/results.png`)
4-panel visualization showing:
- Geographic distribution of organic matter predictions
- Geographic distribution of pH predictions  
- Model performance comparison (R² scores)
- Geographic distribution of CEC predictions

## Data Pipeline

1. **Synthetic Data Generation**
   - Gamma radiometric data (K, U, Th)
   - Satellite indices (NDVI, NDWI, clay index)
   - Elevation and spatial coordinates
   - Target soil properties (organic matter, pH, CEC)

2. **Feature Engineering**
   - Gamma radiometric ratios (K/Th, U/K, etc.)
   - Vegetation-adjusted indices
   - Composite soil indices
   - Spatial interaction features

3. **Model Training**
   - RandomForest and Ridge regression
   - Cross-validation and model selection
   - Performance evaluation

4. **Prediction Generation**
   - Location-based predictions
   - Uncertainty quantification
   - Confidence intervals

5. **Visualization & Export**
   - Geographic prediction maps
   - Performance metrics charts
   - CSV and JSON export

## Example Usage

```python
from soil_pipeline import SoilPipeline

# Initialize pipeline
pipeline = SoilPipeline()

# Run complete pipeline
predictions, metrics = pipeline.run_pipeline()

# Access results
print(f"Generated {len(predictions)} predictions")
print(f"Average R² score: {np.mean([m['r2_score'] for m in metrics.values()]):.3f}")
```

## File Structure

```
├── soil_pipeline.py       # Main pipeline implementation
├── test_pipeline.py       # Comprehensive unit tests
├── requirements.txt       # Dependencies
├── outputs/              # Generated outputs
│   ├── predictions.csv   # Location-based predictions
│   └── metrics.json      # Model performance metrics
└── assets/               # Visualization outputs
    └── results.png       # Results visualization
```

## Testing

```bash
python -m pytest test_pipeline.py -v
```

Tests cover:
- Data generation and validation
- Feature engineering
- Model training and evaluation
- Prediction generation
- Output file creation
- Visualization generation
- Full pipeline integration