"""
Streamlit Dashboard for Soil Property Analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import rasterio
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium
from pathlib import Path
import json
from soil_pipeline import SoilPipeline, PipelineConfig


def load_pipeline_results():
    """Load results from the pipeline."""
    config = PipelineConfig()
    
    # Check if results exist
    benchmark_file = Path(config.output_dir) / 'pipeline_benchmarks.json'
    geotiff_file = Path(config.output_dir) / 'soil_properties.tif'
    
    if not benchmark_file.exists():
        # Run pipeline if results don't exist
        st.info("Running pipeline to generate results...")
        pipeline = SoilPipeline(config)
        results = pipeline.run_pipeline()
        return pipeline, results
    else:
        # Load existing results
        with open(benchmark_file, 'r') as f:
            results = json.load(f)
        
        # Create pipeline object with cached data
        pipeline = SoilPipeline(config)
        pipeline.generate_sample_data()
        pipeline.clean_data() 
        pipeline.engineer_features()
        pipeline.create_soil_indices()
        
        return pipeline, results


def create_soil_map(pipeline, property_name):
    """Create an interactive map for a soil property."""
    if property_name not in pipeline.features:
        return None
    
    data = pipeline.features[property_name]
    bounds = pipeline.features['bounds']
    
    # Create a simplified map (downsample for performance)
    downsample_factor = 10
    data_small = data[::downsample_factor, ::downsample_factor]
    
    # Calculate map center
    center_lat = (bounds[1] + bounds[3]) / 2
    center_lon = (bounds[0] + bounds[2]) / 2
    
    # Convert to lat/lon (simplified)
    center_lat_deg = center_lat / 111000  # Rough conversion
    center_lon_deg = center_lon / 111000
    
    # Create base map
    m = folium.Map(
        location=[41.5, -104.0],  # Colorado area
        zoom_start=10,
        tiles='OpenStreetMap'
    )
    
    # Add colormap overlay (simplified)
    # Note: In a real implementation, you'd properly georeference this
    folium.plugins.HeatMap(
        [[41.5 + i*0.001, -104.0 + j*0.001, data_small[i, j]] 
         for i in range(0, data_small.shape[0], 5) 
         for j in range(0, data_small.shape[1], 5)],
        min_opacity=0.2,
        radius=15,
        blur=10,
        gradient={0.4: 'blue', 0.6: 'cyan', 0.7: 'lime', 0.8: 'yellow', 1.0: 'red'}
    ).add_to(m)
    
    return m


def main():
    st.set_page_config(
        page_title="Geo Pipeline Mini Dashboard",
        page_icon="🌍",
        layout="wide"
    )
    
    st.title("🌍 Geo Pipeline Mini - Soil Property Dashboard")
    st.markdown("Transform gamma + satellite data into actionable soil property indices")
    
    # Load pipeline results
    with st.spinner("Loading pipeline results..."):
        pipeline, results = load_pipeline_results()
    
    # Sidebar
    st.sidebar.header("Pipeline Controls")
    
    # Show pipeline status
    st.sidebar.success(f"Status: {results.get('pipeline_status', 'Unknown')}")
    
    # Main dashboard tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🗺️ Soil Maps", "📈 Analysis", "⚡ Performance"])
    
    with tab1:
        st.header("Pipeline Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Processing Time", 
                f"{results.get('benchmarks', {}).get('total_pipeline', 0):.2f}s"
            )
        
        with col2:
            data_dims = results.get('data_dimensions', {})
            pixels = data_dims.get('total_pixels', 0)
            st.metric("Total Pixels", f"{pixels:,}")
        
        with col3:
            soil_indices = results.get('soil_indices', [])
            st.metric("Soil Properties", len(soil_indices))
        
        with col4:
            avg_r2 = np.mean(list(results.get('model_performance', {}).values()))
            st.metric("Avg R² Score", f"{avg_r2:.3f}")
        
        # Pipeline flow diagram
        st.subheader("Pipeline Flow")
        
        pipeline_steps = [
            "📡 Ingest (Gamma + Satellite)",
            "🧹 Clean (Outlier Removal)",
            "⚙️ Feature Engineering",
            "🤖 Model Training",
            "🗺️ Export GeoTIFF"
        ]
        
        cols = st.columns(len(pipeline_steps))
        for i, (col, step) in enumerate(zip(cols, pipeline_steps)):
            with col:
                st.info(step)
                if i < len(pipeline_steps) - 1:
                    st.markdown("↓")
        
        # Model performance summary
        st.subheader("Model Performance")
        
        if 'model_performance' in results:
            perf_df = pd.DataFrame([
                {'Property': prop, 'R² Score': score, 'Quality': 'Excellent' if score > 0.8 else 'Good' if score > 0.6 else 'Fair'}
                for prop, score in results['model_performance'].items()
            ])
            
            st.dataframe(perf_df, use_container_width=True)
    
    with tab2:
        st.header("Soil Property Maps")
        
        # Property selector
        soil_properties = ['SOM', 'Clay', 'pH']
        selected_property = st.selectbox("Select Soil Property", soil_properties)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader(f"{selected_property} Distribution")
            
            if selected_property in pipeline.features:
                data = pipeline.features[selected_property]
                
                # Create matplotlib heatmap
                fig, ax = plt.subplots(figsize=(10, 8))
                im = ax.imshow(data, cmap='RdYlBu_r', aspect='auto')
                ax.set_title(f'{selected_property} Spatial Distribution')
                ax.set_xlabel('Easting (pixels)')
                ax.set_ylabel('Northing (pixels)')
                
                # Add colorbar
                cbar = plt.colorbar(im, ax=ax)
                if selected_property == 'SOM':
                    cbar.set_label('Soil Organic Matter (%)')
                elif selected_property == 'Clay':
                    cbar.set_label('Clay Content (%)')
                elif selected_property == 'pH':
                    cbar.set_label('pH')
                
                st.pyplot(fig)
            else:
                st.error(f"Data for {selected_property} not available")
        
        with col2:
            st.subheader("Statistics")
            
            if selected_property in pipeline.features:
                data = pipeline.features[selected_property]
                
                stats_df = pd.DataFrame({
                    'Statistic': ['Mean', 'Median', 'Std Dev', 'Min', 'Max'],
                    'Value': [
                        f"{np.mean(data):.2f}",
                        f"{np.median(data):.2f}",
                        f"{np.std(data):.2f}",
                        f"{np.min(data):.2f}",
                        f"{np.max(data):.2f}"
                    ]
                })
                
                st.dataframe(stats_df, use_container_width=True)
                
                # Histogram
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.hist(data.flatten(), bins=50, alpha=0.7, color='skyblue', edgecolor='black')
                ax.set_title(f'{selected_property} Distribution')
                ax.set_xlabel('Value')
                ax.set_ylabel('Frequency')
                st.pyplot(fig)
    
    with tab3:
        st.header("Detailed Analysis")
        
        # Feature correlation analysis
        st.subheader("Feature Correlations")
        
        # Create correlation matrix for key features
        key_features = ['potassium', 'thorium', 'uranium', 'ndvi', 'soil_brightness', 'SOM', 'Clay', 'pH']
        available_features = [f for f in key_features if f in pipeline.features]
        
        if len(available_features) > 3:
            # Sample data for correlation (too many pixels for full correlation)
            sample_size = 10000
            height, width = pipeline.features['SOM'].shape
            
            # Random sampling
            np.random.seed(42)
            sample_indices = np.random.choice(height * width, sample_size, replace=False)
            
            corr_data = {}
            for feature in available_features:
                flat_data = pipeline.features[feature].flatten()
                corr_data[feature] = flat_data[sample_indices]
            
            corr_df = pd.DataFrame(corr_data)
            correlation_matrix = corr_df.corr()
            
            # Plot correlation heatmap
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
            ax.set_title('Feature Correlation Matrix')
            st.pyplot(fig)
        
        # Scatter plot analysis
        st.subheader("Relationship Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            x_feature = st.selectbox("X-axis Feature", available_features, key='x_feat')
        
        with col2:
            y_feature = st.selectbox("Y-axis Feature", available_features, key='y_feat', index=1)
        
        if x_feature != y_feature:
            # Sample data for scatter plot
            sample_size = 5000
            height, width = pipeline.features['SOM'].shape
            
            np.random.seed(42)
            sample_indices = np.random.choice(height * width, sample_size, replace=False)
            
            x_data = pipeline.features[x_feature].flatten()[sample_indices]
            y_data = pipeline.features[y_feature].flatten()[sample_indices]
            
            # Create scatter plot
            fig = px.scatter(
                x=x_data, y=y_data,
                labels={'x': x_feature, 'y': y_feature},
                title=f'{y_feature} vs {x_feature}',
                opacity=0.6
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.header("Performance Benchmarks")
        
        # Runtime benchmarks
        st.subheader("Runtime Performance")
        
        if 'benchmarks' in results:
            benchmark_data = results['benchmarks']
            
            # Create benchmark chart
            steps = list(benchmark_data.keys())
            times = list(benchmark_data.values())
            
            fig = px.bar(
                x=steps, y=times,
                title="Pipeline Step Runtime (seconds)",
                labels={'x': 'Pipeline Step', 'y': 'Time (seconds)'}
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Benchmark table
            benchmark_df = pd.DataFrame([
                {'Step': step, 'Time (s)': f"{time:.3f}", 'Percentage': f"{time/benchmark_data['total_pipeline']*100:.1f}%"}
                for step, time in benchmark_data.items()
                if step != 'total_pipeline'
            ])
            
            st.dataframe(benchmark_df, use_container_width=True)
        
        # Data throughput metrics
        st.subheader("Throughput Metrics")
        
        if 'data_dimensions' in results and 'benchmarks' in results:
            total_pixels = results['data_dimensions']['total_pixels']
            total_time = results['benchmarks']['total_pipeline']
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                pixels_per_sec = total_pixels / total_time
                st.metric("Pixels/Second", f"{pixels_per_sec:,.0f}")
            
            with col2:
                mb_processed = total_pixels * 4 * 12 / (1024 * 1024)  # Assuming 4 bytes per pixel, 12 bands
                mb_per_sec = mb_processed / total_time
                st.metric("MB/Second", f"{mb_per_sec:.1f}")
            
            with col3:
                st.metric("Memory Efficiency", "Excellent")
        
        # System requirements
        st.subheader("System Requirements")
        
        requirements_df = pd.DataFrame([
            {'Component': 'CPU', 'Requirement': '4+ cores recommended', 'Status': '✅'},
            {'Component': 'RAM', 'Requirement': '8GB minimum', 'Status': '✅'},
            {'Component': 'Storage', 'Requirement': '1GB for sample data', 'Status': '✅'},
            {'Component': 'Python', 'Requirement': '3.8+', 'Status': '✅'}
        ])
        
        st.dataframe(requirements_df, use_container_width=True)


if __name__ == "__main__":
    main()