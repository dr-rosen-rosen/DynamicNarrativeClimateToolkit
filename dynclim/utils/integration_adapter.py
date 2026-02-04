"""
Integration Adapter for Climate EWS Dashboard
=============================================
This module provides the interface between your actual temporal_features.py 
and ews.py modules and the dashboard visualization.

Replace the mock functions in the dashboard with these implementations.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Tuple
import streamlit as st

# Import your actual modules
from dynclim.features.temporal.temporal_features import (
    # compute_temporal_features,
    rolling_burstiness_by_count,
    inter_event_durations
)
from dynclim.features.temporal.ews import (
    rolling_or_expanding_ews,
    EWSConfig#,
    # ews_for_metric
)

class DashboardDataProcessor:
    """Process data for dashboard visualization"""
    
    def __init__(self):
        self.temporal_cache = {}
        self.ews_cache = {}
    
    @st.cache_data
    def compute_temporal_features_cached(_self, 
                                        df: pd.DataFrame,
                                        time_col: str = 'event_date',
                                        window_size: int = 50,
                                        step: int = 25,
                                        method: str = 'goh') -> pd.DataFrame:
        """
        Compute temporal features with caching for Streamlit
        """
        # Ensure time column is datetime
        df[time_col] = pd.to_datetime(df[time_col])
        
        # Sort by time
        df = df.sort_values(time_col).reset_index(drop=True)
        
        # Compute rolling burstiness
        burstiness_df = rolling_burstiness_by_count(
            timestamps=df[time_col],
            window_size=window_size,
            step=step,
            unit='d',  # days
            method=method,
            min_valid=3,
            sort=True,
            include_intervals=False,
            compute_memory=True,
            memory_lag=1
        )
        
        return burstiness_df
    
    @st.cache_data
    def compute_ews_cached(_self,
                          df: pd.DataFrame,
                          value_col: str,
                          time_col: str = 'event_date',
                          window_size: int = 50,
                          step: int = 25,
                          detrend: str = 'linear',
                          compute_all: bool = True) -> pd.DataFrame:
        """
        Compute EWS indicators with caching for Streamlit
        """
        # Ensure time column is datetime
        df[time_col] = pd.to_datetime(df[time_col])
        
        # Sort by time
        df = df.sort_values(time_col).reset_index(drop=True)
        
        # Configure EWS computation
        cfg = EWSConfig(
            detrend=detrend,
            fs=1.0,  # Use index as uniform sampling
            low_frac=0.1,
            high_frac=0.1,
            dfa_order=1,
            # Enable all indicators if requested
            compute_mean=compute_all,
            compute_variance=compute_all,
            compute_std=compute_all,
            compute_cv=compute_all,
            compute_skew=compute_all,
            compute_kurtosis=compute_all,
            compute_acf1=compute_all,
            compute_ar1=compute_all,
            compute_return_rate=compute_all,
            compute_spectral_ratio=compute_all,
            compute_spectral_slope=compute_all,
            compute_dfa_alpha=compute_all
        )
        
        # Extract values and timestamps
        values = df[value_col].values
        timestamps = df[time_col].values
        
        # Compute EWS
        ews_df = rolling_or_expanding_ews(
            values=values,
            timestamps=timestamps,
            window_size=window_size,
            step=step,
            mode='rolling',
            cfg=cfg
        )
        
        return ews_df
    
    @st.cache_data
    def compute_global_statistics(_self,
                                 df: pd.DataFrame,
                                 time_col: str = 'event_date') -> Dict:
        """
        Compute global statistics for the entire dataset
        """
        # Calculate inter-event durations
        durations = inter_event_durations(
            timestamps=df[time_col],
            sort=True,
            unit='d'  # days
        )
        
        # Calculate statistics
        stats = {
            'n_events': len(df),
            'duration_days': (df[time_col].max() - df[time_col].min()).days,
            'mean_interval': float(np.mean(durations)) if len(durations) > 0 else np.nan,
            'std_interval': float(np.std(durations)) if len(durations) > 0 else np.nan,
            'cv_interval': float(np.std(durations) / np.mean(durations)) if len(durations) > 0 and np.mean(durations) > 0 else np.nan,
        }
        
        # Calculate global burstiness
        if stats['cv_interval'] is not np.nan:
            cv = stats['cv_interval']
            stats['global_burstiness'] = (cv - 1) / (cv + 1)
        else:
            stats['global_burstiness'] = np.nan
        
        return stats
    
    def identify_ews_alerts(self, 
                           ews_df: pd.DataFrame,
                           thresholds: Dict[str, float]) -> pd.DataFrame:
        """
        Identify windows where EWS indicators exceed thresholds
        
        Parameters
        ----------
        ews_df : DataFrame with EWS indicators
        thresholds : Dict mapping indicator names to threshold values
        
        Returns
        -------
        DataFrame with alert information
        """
        alerts = []
        
        for idx, row in ews_df.iterrows():
            window_alerts = []
            
            # Check each threshold
            for indicator, threshold in thresholds.items():
                if indicator in row and row[indicator] > threshold:
                    window_alerts.append({
                        'indicator': indicator,
                        'value': row[indicator],
                        'threshold': threshold,
                        'severity': 'high' if row[indicator] > threshold * 1.2 else 'medium'
                    })
            
            if window_alerts:
                alerts.append({
                    'window_index': row['window_index'],
                    'start_time': row['start_time'],
                    'end_time': row['end_time'],
                    'alerts': window_alerts,
                    'n_alerts': len(window_alerts)
                })
        
        return pd.DataFrame(alerts)
    
    def compute_trend_significance(self,
                                   ews_df: pd.DataFrame,
                                   indicators: List[str],
                                   method: str = 'kendall') -> Dict:
        """
        Compute trend significance for EWS indicators
        
        Parameters
        ----------
        ews_df : DataFrame with EWS indicators
        indicators : List of indicator names to test
        method : 'kendall' or 'spearman'
        
        Returns
        -------
        Dict with trend statistics and p-values
        """
        from scipy import stats
        
        trends = {}
        
        for indicator in indicators:
            if indicator not in ews_df.columns:
                continue
            
            values = ews_df[indicator].dropna().values
            if len(values) < 3:
                continue
            
            x = np.arange(len(values))
            
            if method == 'kendall':
                tau, p_value = stats.kendalltau(x, values)
                trends[indicator] = {
                    'tau': tau,
                    'p_value': p_value,
                    'significant': p_value < 0.05,
                    'direction': 'increasing' if tau > 0 else 'decreasing'
                }
            else:  # spearman
                rho, p_value = stats.spearmanr(x, values)
                trends[indicator] = {
                    'rho': rho,
                    'p_value': p_value,
                    'significant': p_value < 0.05,
                    'direction': 'increasing' if rho > 0 else 'decreasing'
                }
        
        return trends

class VisualizationHelpers:
    """Helper functions for creating enhanced visualizations"""
    
    @staticmethod
    def create_phase_space_plot(ews_df: pd.DataFrame,
                               x_var: str = 'variance',
                               y_var: str = 'ar1',
                               color_var: Optional[str] = 'window_index') -> dict:
        """
        Create phase space plot configuration for Plotly
        """
        import plotly.graph_objects as go
        
        trace = go.Scatter(
            x=ews_df[x_var],
            y=ews_df[y_var],
            mode='markers+lines',
            marker=dict(
                size=8,
                color=ews_df[color_var] if color_var else 'blue',
                colorscale='Viridis',
                showscale=True if color_var else False,
                colorbar=dict(title=color_var.replace('_', ' ').title() if color_var else '')
            ),
            line=dict(width=1, color='gray', dash='dash'),
            text=[f"Window {i}" for i in ews_df['window_index']],
            hovertemplate=f"{x_var}: %{{x:.3f}}<br>{y_var}: %{{y:.3f}}<br>%{{text}}<extra></extra>"
        )
        
        layout = dict(
            title=f"Phase Space: {x_var.title()} vs {y_var.upper()}",
            xaxis_title=x_var.replace('_', ' ').title(),
            yaxis_title=y_var.replace('_', ' ').title(),
            template='plotly_white',
            hovermode='closest'
        )
        
        return {'trace': trace, 'layout': layout}
    
    @staticmethod
    def create_heatmap_matrix(ews_df: pd.DataFrame,
                             indicators: List[str]) -> dict:
        """
        Create correlation heatmap for EWS indicators
        """
        import plotly.graph_objects as go
        
        # Calculate correlation matrix
        corr_matrix = ews_df[indicators].corr()
        
        trace = go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=corr_matrix.values.round(2),
            texttemplate='%{text}',
            textfont={"size": 10},
            colorbar=dict(title='Correlation')
        )
        
        layout = dict(
            title="EWS Indicator Correlations",
            xaxis=dict(tickangle=-45),
            yaxis=dict(tickangle=0),
            template='plotly_white',
            height=500,
            width=600
        )
        
        return {'trace': trace, 'layout': layout}
    
    @staticmethod
    def create_regime_identification_plot(df: pd.DataFrame,
                                         value_col: str,
                                         burstiness_df: pd.DataFrame,
                                         b_threshold: float = 0.3) -> dict:
        """
        Identify and visualize different regimes based on burstiness
        """
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        # Identify regimes
        regimes = []
        current_regime = None
        
        for idx, row in burstiness_df.iterrows():
            if row['b'] > b_threshold:
                regime = 'bursty'
            elif row['b'] < -b_threshold:
                regime = 'regular'
            else:
                regime = 'transitional'
            
            if regime != current_regime:
                if current_regime is not None:
                    regimes[-1]['end_time'] = row['start_time']
                    regimes[-1]['end_idx'] = row['start_idx']
                
                regimes.append({
                    'regime': regime,
                    'start_time': row['start_time'],
                    'start_idx': row['start_idx'],
                    'end_time': None,
                    'end_idx': None
                })
                current_regime = regime
        
        # Close last regime
        if regimes and regimes[-1]['end_time'] is None:
            regimes[-1]['end_time'] = burstiness_df.iloc[-1]['end_time']
            regimes[-1]['end_idx'] = burstiness_df.iloc[-1]['end_idx']
        
        return pd.DataFrame(regimes)

# Example usage in Streamlit dashboard
def integrate_with_dashboard():
    """
    Example of how to integrate with the main dashboard
    """
    # Initialize processor
    processor = DashboardDataProcessor()
    
    # In your Streamlit app, replace mock functions:
    # Instead of: mock_compute_temporal_features(df, ...)
    # Use: processor.compute_temporal_features_cached(df, ...)
    
    # Instead of: mock_compute_ews(df, ...)
    # Use: processor.compute_ews_cached(df, ...)
    
    # Example:
    df = pd.read_csv('your_data.csv')
    
    # Compute features
    burstiness_df = processor.compute_temporal_features_cached(
        df, 
        time_col='event_date',
        window_size=50,
        step=25
    )
    
    ews_df = processor.compute_ews_cached(
        df,
        value_col='temperature_anomaly',
        time_col='event_date',
        window_size=50,
        step=25,
        detrend='linear'
    )
    
    # Get global stats
    stats = processor.compute_global_statistics(df)
    
    # Check for alerts
    thresholds = {
        'ar1': 0.8,
        'variance': stats['std_interval'] * 2,  # 2x baseline
        'spectral_ratio': 3.0
    }
    alerts = processor.identify_ews_alerts(ews_df, thresholds)
    
    # Test for trends
    trends = processor.compute_trend_significance(
        ews_df, 
        ['variance', 'ar1', 'acf1'],
        method='kendall'
    )
    
    return {
        'burstiness': burstiness_df,
        'ews': ews_df,
        'stats': stats,
        'alerts': alerts,
        'trends': trends
    }

if __name__ == "__main__":
    # Test the integration
    print("Integration adapter loaded successfully")
    print("Use 'processor = DashboardDataProcessor()' to initialize")
    print("Then call processor methods to compute features for the dashboard")