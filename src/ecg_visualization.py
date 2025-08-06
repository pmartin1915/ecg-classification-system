"""
ECG Visualization Module for Clinical Training and Analysis
Provides interactive ECG waveform display with educational annotations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings

warnings.filterwarnings('ignore')

class ECGVisualizer:
    """
    Comprehensive ECG visualization for clinical training and analysis.
    
    Supports:
    - Multi-lead ECG display
    - Educational annotations
    - Abnormality highlighting
    - Interactive features for learning
    """
    
    def __init__(self):
        self.lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        self.sampling_rate = 100  # Default 100 Hz
        
    def create_sample_ecg(self, condition: str = "normal", duration: int = 10) -> Dict[str, np.ndarray]:
        """
        Create sample ECG data for educational purposes.
        
        Args:
            condition: Type of ECG ('normal', 'stemi_anterior', 'stemi_inferior', 'afib', 'lbbb')
            duration: Duration in seconds
            
        Returns:
            Dictionary with lead names as keys and signal arrays as values
        """
        samples = self.sampling_rate * duration
        time = np.linspace(0, duration, samples)
        
        # Create base ECG pattern
        ecg_data = {}
        
        if condition == "normal":
            for i, lead in enumerate(self.lead_names):
                # Create realistic normal ECG pattern
                signal = self._create_normal_ecg_pattern(time, lead, i)
                ecg_data[lead] = signal
                
        elif condition == "stemi_anterior":
            for i, lead in enumerate(self.lead_names):
                signal = self._create_normal_ecg_pattern(time, lead, i)
                
                # Add ST elevation for anterior leads (V1-V4)
                if lead in ['V1', 'V2', 'V3', 'V4']:
                    signal = self._add_st_elevation(signal, time, severity=0.3)
                
                # Add reciprocal changes for inferior leads
                elif lead in ['II', 'III', 'aVF']:
                    signal = self._add_st_depression(signal, time, severity=0.1)
                    
                ecg_data[lead] = signal
                
        elif condition == "stemi_inferior":
            for i, lead in enumerate(self.lead_names):
                signal = self._create_normal_ecg_pattern(time, lead, i)
                
                # Add ST elevation for inferior leads
                if lead in ['II', 'III', 'aVF']:
                    signal = self._add_st_elevation(signal, time, severity=0.4)
                
                # Add reciprocal changes for high lateral leads
                elif lead in ['I', 'aVL']:
                    signal = self._add_st_depression(signal, time, severity=0.15)
                    
                ecg_data[lead] = signal
                
        elif condition == "afib":
            for i, lead in enumerate(self.lead_names):
                # Create irregular rhythm with absent P waves
                signal = self._create_afib_pattern(time, lead, i)
                ecg_data[lead] = signal
                
        elif condition == "lbbb":
            for i, lead in enumerate(self.lead_names):
                signal = self._create_normal_ecg_pattern(time, lead, i)
                
                # Add LBBB pattern - wide QRS with specific morphology
                signal = self._add_lbbb_pattern(signal, time, lead)
                ecg_data[lead] = signal
        
        return ecg_data
    
    def _create_normal_ecg_pattern(self, time: np.ndarray, lead: str, lead_index: int) -> np.ndarray:
        """Create a realistic normal ECG pattern for a specific lead."""
        signal = np.zeros_like(time)
        heart_rate = 70  # BPM
        rr_interval = 60.0 / heart_rate  # seconds
        
        # Generate QRS complexes
        for beat_time in np.arange(0, time[-1], rr_interval):
            beat_start = int(beat_time * self.sampling_rate)
            
            # P wave
            p_start = beat_start - int(0.02 * self.sampling_rate)
            p_end = beat_start + int(0.08 * self.sampling_rate)
            if p_start >= 0 and p_end < len(signal):
                p_amplitude = 0.1 + 0.05 * np.sin(lead_index * np.pi / 6)
                signal[p_start:p_end] += p_amplitude * np.exp(-((np.arange(p_end-p_start) - (p_end-p_start)/2)**2) / (2*(0.02*self.sampling_rate)**2))
            
            # QRS complex
            qrs_start = beat_start
            qrs_end = beat_start + int(0.08 * self.sampling_rate)
            if qrs_start >= 0 and qrs_end < len(signal):
                # Different QRS morphology for different leads
                if lead in ['I', 'II', 'V4', 'V5', 'V6']:
                    qrs_amplitude = 1.0 + 0.3 * np.sin(lead_index * np.pi / 4)
                elif lead in ['aVR']:
                    qrs_amplitude = -0.8
                else:
                    qrs_amplitude = 0.6 + 0.2 * np.cos(lead_index * np.pi / 3)
                    
                qrs_pattern = np.array([-0.1, -0.2, 1.0, -0.3, 0.1]) * qrs_amplitude
                qrs_indices = np.linspace(qrs_start, qrs_end-1, len(qrs_pattern)).astype(int)
                
                for i, idx in enumerate(qrs_indices):
                    if 0 <= idx < len(signal):
                        signal[idx] += qrs_pattern[i]
            
            # T wave
            t_start = beat_start + int(0.3 * self.sampling_rate)
            t_end = beat_start + int(0.5 * self.sampling_rate)
            if t_start >= 0 and t_end < len(signal):
                t_amplitude = 0.2 + 0.1 * np.cos(lead_index * np.pi / 8)
                if lead == 'aVR':
                    t_amplitude = -0.1
                signal[t_start:t_end] += t_amplitude * np.exp(-((np.arange(t_end-t_start) - (t_end-t_start)/2)**2) / (2*(0.1*self.sampling_rate)**2))
        
        # Add some realistic noise
        signal += np.random.normal(0, 0.01, len(signal))
        
        return signal
    
    def _add_st_elevation(self, signal: np.ndarray, time: np.ndarray, severity: float = 0.2) -> np.ndarray:
        """Add ST elevation to ECG signal."""
        modified_signal = signal.copy()
        heart_rate = 70
        rr_interval = 60.0 / heart_rate
        
        for beat_time in np.arange(0, time[-1], rr_interval):
            beat_start = int(beat_time * self.sampling_rate)
            st_start = beat_start + int(0.08 * self.sampling_rate)  # After QRS
            st_end = beat_start + int(0.3 * self.sampling_rate)   # Before T wave
            
            if st_start >= 0 and st_end < len(modified_signal):
                st_length = st_end - st_start
                # Create elevated ST segment
                elevation = severity * np.ones(st_length)
                modified_signal[st_start:st_end] += elevation
        
        return modified_signal
    
    def _add_st_depression(self, signal: np.ndarray, time: np.ndarray, severity: float = 0.1) -> np.ndarray:
        """Add ST depression to ECG signal."""
        return self._add_st_elevation(signal, time, -severity)
    
    def _create_afib_pattern(self, time: np.ndarray, lead: str, lead_index: int) -> np.ndarray:
        """Create atrial fibrillation pattern with irregular rhythm."""
        signal = np.zeros_like(time)
        
        # Irregular RR intervals (characteristic of AFib)
        current_time = 0
        while current_time < time[-1]:
            # Random RR interval between 0.4 and 1.2 seconds
            rr_interval = np.random.uniform(0.4, 1.2)
            
            beat_start = int(current_time * self.sampling_rate)
            
            # No clear P waves in AFib - add fibrillatory waves instead
            fib_waves = 0.03 * np.random.normal(0, 1, int(0.2 * self.sampling_rate))
            fib_start = max(0, beat_start - int(0.1 * self.sampling_rate))
            fib_end = min(len(signal), fib_start + len(fib_waves))
            signal[fib_start:fib_end] += fib_waves[:fib_end-fib_start]
            
            # QRS complex (similar to normal but without preceding P wave)
            qrs_start = beat_start
            qrs_end = beat_start + int(0.08 * self.sampling_rate)
            if qrs_start >= 0 and qrs_end < len(signal):
                if lead in ['I', 'II', 'V4', 'V5', 'V6']:
                    qrs_amplitude = 0.9 + 0.2 * np.sin(lead_index * np.pi / 4)
                elif lead in ['aVR']:
                    qrs_amplitude = -0.7
                else:
                    qrs_amplitude = 0.5 + 0.2 * np.cos(lead_index * np.pi / 3)
                    
                qrs_pattern = np.array([-0.1, -0.2, 1.0, -0.3, 0.1]) * qrs_amplitude
                qrs_indices = np.linspace(qrs_start, qrs_end-1, len(qrs_pattern)).astype(int)
                
                for i, idx in enumerate(qrs_indices):
                    if 0 <= idx < len(signal):
                        signal[idx] += qrs_pattern[i]
            
            current_time += rr_interval
        
        # Add more baseline noise characteristic of AFib
        signal += np.random.normal(0, 0.02, len(signal))
        
        return signal
    
    def _add_lbbb_pattern(self, signal: np.ndarray, time: np.ndarray, lead: str) -> np.ndarray:
        """Add left bundle branch block pattern."""
        modified_signal = signal.copy()
        heart_rate = 70
        rr_interval = 60.0 / heart_rate
        
        for beat_time in np.arange(0, time[-1], rr_interval):
            beat_start = int(beat_time * self.sampling_rate)
            
            # Widen QRS complex (characteristic of LBBB)
            qrs_start = beat_start
            qrs_end = beat_start + int(0.12 * self.sampling_rate)  # Widened QRS (>120ms)
            
            if qrs_start >= 0 and qrs_end < len(modified_signal):
                # Clear original QRS
                original_qrs_end = beat_start + int(0.08 * self.sampling_rate)
                if original_qrs_end < len(modified_signal):
                    modified_signal[qrs_start:original_qrs_end] *= 0.1
                
                # Add LBBB-specific morphology
                if lead in ['I', 'aVL', 'V5', 'V6']:
                    # Broad, notched R waves in lateral leads
                    lbbb_pattern = np.array([0.0, 0.3, 0.6, 0.4, 0.8, 0.5, 0.2])
                elif lead in ['V1', 'V2']:
                    # Deep QS pattern in right precordial leads
                    lbbb_pattern = np.array([0.0, -0.2, -0.6, -0.8, -0.4, -0.2, 0.0])
                else:
                    # Intermediate pattern
                    lbbb_pattern = np.array([0.0, 0.2, 0.4, 0.2, 0.5, 0.3, 0.1])
                
                pattern_indices = np.linspace(qrs_start, qrs_end-1, len(lbbb_pattern)).astype(int)
                
                for i, idx in enumerate(pattern_indices):
                    if 0 <= idx < len(modified_signal):
                        modified_signal[idx] += lbbb_pattern[i]
        
        return modified_signal
    
    def plot_12_lead_ecg(self, ecg_data: Dict[str, np.ndarray], title: str = "12-Lead ECG", 
                        annotations: Optional[Dict[str, List]] = None) -> go.Figure:
        """
        Create interactive 12-lead ECG plot using Plotly.
        
        Args:
            ecg_data: Dictionary with lead names as keys and signal arrays as values
            title: Plot title
            annotations: Dictionary with annotation information
            
        Returns:
            Plotly figure object
        """
        # Standard 12-lead layout (4 rows x 3 columns)
        lead_layout = [
            ['I', 'aVR', 'V1', 'V4'],
            ['II', 'aVL', 'V2', 'V5'],
            ['III', 'aVF', 'V3', 'V6']
        ]
        
        fig = make_subplots(
            rows=3, cols=4,
            subplot_titles=sum(lead_layout, []),
            vertical_spacing=0.08,
            horizontal_spacing=0.05
        )
        
        # Create time axis (assuming 100 Hz sampling rate)
        max_length = max(len(signal) for signal in ecg_data.values())
        time_axis = np.arange(max_length) / self.sampling_rate
        
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
        
        for row_idx, row in enumerate(lead_layout):
            for col_idx, lead in enumerate(row):
                if lead in ecg_data:
                    signal = ecg_data[lead]
                    
                    # Ensure signal matches time axis length
                    if len(signal) < max_length:
                        signal = np.pad(signal, (0, max_length - len(signal)), 'constant')
                    
                    fig.add_trace(
                        go.Scatter(
                            x=time_axis,
                            y=signal,
                            mode='lines',
                            name=lead,
                            line=dict(color=colors[col_idx % len(colors)], width=1.5),
                            showlegend=False
                        ),
                        row=row_idx + 1,
                        col=col_idx + 1
                    )
                    
                    # Add annotations if provided
                    if annotations and lead in annotations:
                        for annotation in annotations[lead]:
                            fig.add_annotation(
                                x=annotation.get('x', 0),
                                y=annotation.get('y', 0),
                                text=annotation.get('text', ''),
                                showarrow=True,
                                arrowhead=2,
                                arrowcolor=annotation.get('color', 'red'),
                                row=row_idx + 1,
                                col=col_idx + 1
                            )
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                font=dict(size=16, color='#2E86AB')
            ),
            height=600,
            showlegend=False,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        # Update axes
        fig.update_xaxes(
            title_text="Time (seconds)",
            showgrid=True,
            gridwidth=0.5,
            gridcolor='lightgray',
            row=3  # Only show x-axis title on bottom row
        )
        
        fig.update_yaxes(
            title_text="Amplitude (mV)",
            showgrid=True,
            gridwidth=0.5,
            gridcolor='lightgray',
            col=1  # Only show y-axis title on left column
        )
        
        return fig
    
    def create_educational_ecg_comparison(self, conditions: List[str]) -> go.Figure:
        """
        Create side-by-side comparison of different ECG conditions for educational purposes.
        
        Args:
            conditions: List of conditions to compare
            
        Returns:
            Plotly figure with comparison
        """
        fig = make_subplots(
            rows=len(conditions), cols=1,
            subplot_titles=[f"{condition.upper()} Pattern" for condition in conditions],
            vertical_spacing=0.1
        )
        
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#8E44AD']
        
        for idx, condition in enumerate(conditions):
            # Create sample ECG for this condition
            ecg_data = self.create_sample_ecg(condition, duration=5)
            
            # Use Lead II for comparison (most representative)
            if 'II' in ecg_data:
                signal = ecg_data['II']
                time_axis = np.arange(len(signal)) / self.sampling_rate
                
                fig.add_trace(
                    go.Scatter(
                        x=time_axis,
                        y=signal,
                        mode='lines',
                        name=condition.upper(),
                        line=dict(color=colors[idx % len(colors)], width=2),
                        showlegend=True
                    ),
                    row=idx + 1,
                    col=1
                )
        
        fig.update_layout(
            title="ECG Pattern Comparison for Educational Learning",
            height=150 * len(conditions),
            showlegend=True,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        fig.update_xaxes(title_text="Time (seconds)", showgrid=True)
        fig.update_yaxes(title_text="Amplitude (mV)", showgrid=True)
        
        return fig
    
    def highlight_ecg_features(self, ecg_data: Dict[str, np.ndarray], 
                              features: Dict[str, Dict]) -> go.Figure:
        """
        Create ECG plot with highlighted clinical features.
        
        Args:
            ecg_data: ECG signal data
            features: Dictionary of features to highlight
                     Format: {'feature_name': {'leads': [leads], 'time_ranges': [(start, end)], 'color': 'color'}}
        
        Returns:
            Plotly figure with highlighted features
        """
        fig = self.plot_12_lead_ecg(ecg_data, "ECG with Clinical Feature Highlights")
        
        # Add feature highlights
        for feature_name, feature_info in features.items():
            leads = feature_info.get('leads', [])
            time_ranges = feature_info.get('time_ranges', [])
            color = feature_info.get('color', 'red')
            
            for lead in leads:
                if lead in ecg_data:
                    for time_start, time_end in time_ranges:
                        # Add colored background to highlight the feature
                        fig.add_vrect(
                            x0=time_start,
                            x1=time_end,
                            fillcolor=color,
                            opacity=0.2,
                            layer="below",
                            line_width=0,
                        )
        
        return fig


def display_ecg_with_streamlit(visualizer: ECGVisualizer, condition: str, 
                              include_annotations: bool = False) -> None:
    """
    Display ECG visualization in Streamlit with optional annotations.
    
    Args:
        visualizer: ECGVisualizer instance
        condition: ECG condition to display
        include_annotations: Whether to include educational annotations
    """
    # Create sample ECG
    ecg_data = visualizer.create_sample_ecg(condition)
    
    # Create annotations if requested
    annotations = None
    if include_annotations:
        if condition == "stemi_anterior":
            annotations = {
                'V2': [{'x': 2.0, 'y': 0.5, 'text': 'ST Elevation', 'color': 'red'}],
                'V3': [{'x': 2.0, 'y': 0.5, 'text': 'ST Elevation', 'color': 'red'}],
                'II': [{'x': 2.0, 'y': -0.2, 'text': 'Reciprocal Depression', 'color': 'orange'}]
            }
        elif condition == "stemi_inferior":
            annotations = {
                'II': [{'x': 2.0, 'y': 0.4, 'text': 'ST Elevation', 'color': 'red'}],
                'III': [{'x': 2.0, 'y': 0.4, 'text': 'ST Elevation', 'color': 'red'}],
                'aVF': [{'x': 2.0, 'y': 0.4, 'text': 'ST Elevation', 'color': 'red'}]
            }
    
    # Create and display the plot
    fig = visualizer.plot_12_lead_ecg(ecg_data, f"{condition.upper()} ECG Pattern", annotations)
    st.plotly_chart(fig, use_container_width=True)