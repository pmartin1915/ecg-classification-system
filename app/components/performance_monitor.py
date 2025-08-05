"""
Performance Monitoring Dashboard
Real-time performance tracking and optimization for ECG analysis
"""
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any

class PerformanceMonitor:
    """Real-time performance monitoring and optimization dashboard"""
    
    def __init__(self):
        self.initialize_session_state()
    
    def initialize_session_state(self):
        """Initialize session state for performance tracking"""
        if 'performance_history' not in st.session_state:
            st.session_state.performance_history = []
        
        if 'performance_targets' not in st.session_state:
            st.session_state.performance_targets = {
                'total_time': 3.0,  # seconds
                'preprocessing_time': 0.5,
                'feature_extraction_time': 1.5,
                'prediction_time': 0.5,
                'formatting_time': 0.5
            }
    
    def render_performance_dashboard(self):
        """Render the main performance monitoring dashboard"""
        
        st.header("⚡ Performance Monitoring Dashboard")
        st.subheader("Real-Time ECG Analysis Performance Optimization")
        
        # Performance overview
        self.show_performance_overview()
        
        st.divider()
        
        # Performance tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Real-Time Metrics",
            "📈 Performance History", 
            "🎯 Optimization",
            "⚙️ Settings"
        ])
        
        with tab1:
            self.show_realtime_metrics()
        
        with tab2:
            self.show_performance_history()
        
        with tab3:
            self.show_optimization_suggestions()
        
        with tab4:
            self.show_performance_settings()
    
    def show_performance_overview(self):
        """Show high-level performance overview"""
        
        try:
            from app.utils.fast_prediction_pipeline import fast_pipeline
            stats = fast_pipeline.get_performance_stats()
        except Exception:
            stats = {
                'total_predictions': 0,
                'average_time': 0,
                'fastest_time': 0,
                'slowest_time': 0,
                'target_met_rate': 0,
                'models_loaded': []
            }
        
        # Key metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            avg_time = stats.get('average_time', 0)
            delta_color = "normal" if avg_time <= 3.0 else "inverse"
            st.metric(
                "Average Time", 
                f"{avg_time:.2f}s",
                delta=f"Target: 3.0s",
                delta_color=delta_color
            )
        
        with col2:
            target_rate = stats.get('target_met_rate', 0) * 100
            st.metric(
                "Target Met Rate",
                f"{target_rate:.1f}%",
                delta="Target: 90%+",
                delta_color="normal" if target_rate >= 90 else "inverse"
            )
        
        with col3:
            st.metric(
                "Total Predictions",
                f"{stats.get('total_predictions', 0):,}",
                delta="Sessions completed"
            )
        
        with col4:
            fastest = stats.get('fastest_time', 0)
            st.metric(
                "Fastest Time",
                f"{fastest:.2f}s",
                delta="Best performance"
            )
        
        with col5:
            models_count = len(stats.get('models_loaded', []))
            st.metric(
                "Models Loaded",
                f"{models_count}",
                delta="In memory"
            )
        
        # Performance status indicator
        if avg_time <= 2.0:
            st.success("🚀 **EXCELLENT PERFORMANCE** - System running at optimal speed!")
        elif avg_time <= 3.0:
            st.success("✅ **GOOD PERFORMANCE** - Meeting target response time")
        elif avg_time <= 5.0:
            st.warning("⚠️ **ACCEPTABLE PERFORMANCE** - Consider optimization")
        else:
            st.error("🐌 **SLOW PERFORMANCE** - Optimization needed")
    
    def show_realtime_metrics(self):
        """Show real-time performance metrics"""
        
        st.subheader("📊 Real-Time Performance Metrics")
        
        # Performance test section
        st.markdown("### 🧪 Performance Test")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if st.button("🚀 Run Performance Test", type="primary"):
                self.run_performance_test()
        
        with col2:
            auto_test = st.checkbox("Auto-refresh metrics", value=False)
        
        if auto_test:
            # Auto-refresh every 5 seconds
            time.sleep(1)
            st.rerun()
        
        # Current session metrics
        if st.session_state.performance_history:
            recent_performance = st.session_state.performance_history[-1]
            self.display_detailed_metrics(recent_performance)
    
    def run_performance_test(self):
        """Run a performance test with synthetic ECG data"""
        
        with st.spinner("Running performance test..."):
            try:
                from app.utils.fast_prediction_pipeline import fast_pipeline
                
                # Generate synthetic ECG data
                synthetic_ecg = self.generate_synthetic_ecg()
                
                # Run prediction with timing
                start_time = time.time()
                results = fast_pipeline.fast_predict(synthetic_ecg)
                total_time = time.time() - start_time
                
                # Store results
                performance_record = {
                    'timestamp': datetime.now(),
                    'total_time': total_time,
                    'timing': results.get('timing', {}),
                    'diagnosis': results.get('diagnosis', 'Unknown'),
                    'confidence': results.get('confidence', 0),
                    'model_type': results.get('model_info', {}).get('type', 'Unknown'),
                    'performance_grade': results.get('performance_grade', 'Unknown')
                }
                
                st.session_state.performance_history.append(performance_record)
                
                # Keep only last 100 records
                if len(st.session_state.performance_history) > 100:
                    st.session_state.performance_history = st.session_state.performance_history[-100:]
                
                st.success(f"✅ Performance test completed in {total_time:.2f} seconds!")
                
                # Show results
                self.display_detailed_metrics(performance_record)
                
            except Exception as e:
                st.error(f"Performance test failed: {e}")
    
    def generate_synthetic_ecg(self) -> np.ndarray:
        """Generate synthetic ECG data for testing"""
        
        # Create 12-lead synthetic ECG
        duration = 4  # seconds
        sampling_rate = 100  # Hz
        samples = duration * sampling_rate
        
        ecg_12_lead = np.zeros((12, samples))
        
        for lead in range(12):
            # Generate basic ECG waveform
            t = np.linspace(0, duration, samples)
            
            # Heart rate around 70 bpm
            heart_rate = 70 + np.random.normal(0, 5)
            qrs_frequency = heart_rate / 60
            
            # Basic ECG components
            p_wave = 0.1 * np.sin(2 * np.pi * qrs_frequency * t)
            qrs_complex = 0.8 * np.sin(2 * np.pi * 15 * t) * np.exp(-((t - 1) / 0.1)**2)
            t_wave = 0.2 * np.sin(2 * np.pi * qrs_frequency * t + np.pi/2)
            
            # Combine components
            ecg_12_lead[lead] = p_wave + qrs_complex + t_wave
            
            # Add lead-specific variations
            if lead < 6:  # Limb leads
                ecg_12_lead[lead] *= 0.8
            else:  # Precordial leads
                ecg_12_lead[lead] *= 1.2
            
            # Add noise
            ecg_12_lead[lead] += np.random.normal(0, 0.05, samples)
        
        return ecg_12_lead
    
    def display_detailed_metrics(self, performance_record: Dict):
        """Display detailed performance metrics"""
        
        st.markdown("### 📈 Detailed Performance Breakdown")
        
        timing = performance_record.get('timing', {})
        
        # Performance breakdown chart
        if timing:
            self.create_performance_breakdown_chart(timing)
        
        # Metrics table
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**⏱️ Timing Breakdown:**")
            
            breakdown_data = {
                'Stage': ['Preprocessing', 'Feature Extraction', 'Model Prediction', 'Result Formatting', 'Total'],
                'Time (s)': [
                    timing.get('preprocessing_time', 0),
                    timing.get('feature_extraction_time', 0),
                    timing.get('prediction_time', 0),
                    timing.get('formatting_time', 0),
                    timing.get('total_time', 0)
                ],
                'Target (s)': [0.5, 1.5, 0.5, 0.5, 3.0]
            }
            
            df_breakdown = pd.DataFrame(breakdown_data)
            df_breakdown['Status'] = df_breakdown.apply(
                lambda row: '✅' if row['Time (s)'] <= row['Target (s)'] else '⚠️', axis=1
            )
            
            st.dataframe(df_breakdown, use_container_width=True)
        
        with col2:
            st.markdown("**🎯 Performance Summary:**")
            
            st.write(f"**Diagnosis:** {performance_record.get('diagnosis', 'Unknown')}")
            st.write(f"**Confidence:** {performance_record.get('confidence', 0):.1%}")
            st.write(f"**Model Type:** {performance_record.get('model_type', 'Unknown')}")
            st.write(f"**Performance Grade:** {performance_record.get('performance_grade', 'Unknown')}")
            st.write(f"**Timestamp:** {performance_record.get('timestamp', 'Unknown')}")
            
            # Target status
            target_met = timing.get('target_met', False)
            if target_met:
                st.success("🎯 **TARGET MET** - <3 second response")
            else:
                st.warning("⚠️ **TARGET MISSED** - >3 second response")
    
    def create_performance_breakdown_chart(self, timing: Dict):
        """Create performance breakdown chart"""
        
        stages = ['Preprocessing', 'Feature Extraction', 'Prediction', 'Formatting']
        times = [
            timing.get('preprocessing_time', 0),
            timing.get('feature_extraction_time', 0),
            timing.get('prediction_time', 0),
            timing.get('formatting_time', 0)
        ]
        targets = [0.5, 1.5, 0.5, 0.5]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(stages))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, times, width, label='Actual', alpha=0.8)
        bars2 = ax.bar(x + width/2, targets, width, label='Target', alpha=0.6)
        
        # Color bars based on performance
        for i, (bar, time_val, target) in enumerate(zip(bars1, times, targets)):
            color = 'green' if time_val <= target else 'orange' if time_val <= target * 1.5 else 'red'
            bar.set_color(color)
        
        ax.set_xlabel('Processing Stage')
        ax.set_ylabel('Time (seconds)')
        ax.set_title('Performance Breakdown by Stage')
        ax.set_xticks(x)
        ax.set_xticklabels(stages, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.2f}s', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    def show_performance_history(self):
        """Show performance history and trends"""
        
        st.subheader("📈 Performance History & Trends")
        
        if not st.session_state.performance_history:
            st.info("No performance history available. Run some performance tests to see trends.")
            return
        
        # Convert to DataFrame
        history_df = pd.DataFrame(st.session_state.performance_history)
        
        # Performance trend chart
        if len(history_df) > 1:
            self.create_performance_trend_chart(history_df)
        
        # Recent performance table
        st.markdown("### 📋 Recent Performance Records")
        
        # Show last 10 records
        recent_df = history_df.tail(10).copy()
        recent_df['timestamp'] = recent_df['timestamp'].dt.strftime('%H:%M:%S')
        
        display_columns = ['timestamp', 'total_time', 'diagnosis', 'confidence', 'performance_grade']
        available_columns = [col for col in display_columns if col in recent_df.columns]
        
        if available_columns:
            st.dataframe(recent_df[available_columns], use_container_width=True)
    
    def create_performance_trend_chart(self, history_df: pd.DataFrame):
        """Create performance trend chart"""
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Total time trend
        ax1.plot(history_df.index, history_df['total_time'], marker='o', linewidth=2)
        ax1.axhline(y=3.0, color='red', linestyle='--', alpha=0.7, label='Target (3.0s)')
        ax1.set_title('Response Time Trend')
        ax1.set_ylabel('Total Time (seconds)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Performance grade distribution
        if 'performance_grade' in history_df.columns:
            grade_counts = history_df['performance_grade'].value_counts()
            ax2.bar(grade_counts.index, grade_counts.values, alpha=0.7)
            ax2.set_title('Performance Grade Distribution')
            ax2.set_ylabel('Count')
            ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    def show_optimization_suggestions(self):
        """Show optimization suggestions"""
        
        st.subheader("🎯 Performance Optimization Suggestions")
        
        # Analyze current performance
        suggestions = self.generate_optimization_suggestions()
        
        for category, tips in suggestions.items():
            st.markdown(f"### {category}")
            for tip in tips:
                st.write(f"• {tip}")
            st.write("")
    
    def generate_optimization_suggestions(self) -> Dict[str, List[str]]:
        """Generate performance optimization suggestions"""
        
        try:
            from app.utils.fast_prediction_pipeline import fast_pipeline
            stats = fast_pipeline.get_performance_stats()
            avg_time = stats.get('average_time', 0)
        except Exception:
            avg_time = 5.0  # Assume slow if can't get stats
        
        suggestions = {}
        
        if avg_time > 3.0:
            suggestions["🚀 Speed Optimization"] = [
                "Consider running on more powerful hardware",
                "Reduce feature extraction complexity",
                "Use model quantization for faster inference",
                "Implement feature caching for repeated patterns",
                "Use GPU acceleration if available"
            ]
        
        if avg_time > 1.0:
            suggestions["⚡ Quick Wins"] = [
                "Pre-load models into memory (already implemented)",
                "Use vectorized operations for preprocessing",
                "Reduce ECG signal length for faster processing",
                "Cache frequently used calculations",
                "Optimize feature selection to most critical features"
            ]
        
        suggestions["🏥 Clinical Optimization"] = [
            "Balance speed vs accuracy based on clinical context",
            "Use rapid screening mode for urgent cases",
            "Implement confidence-based processing depth",
            "Add interrupt capability for immediate results",
            "Consider specialized models for specific conditions"
        ]
        
        suggestions["💾 System Optimization"] = [
            "Monitor memory usage during peak loads",
            "Implement connection pooling for database access",
            "Use asynchronous processing where possible",
            "Add performance monitoring alerts",
            "Regular cleanup of temporary files and caches"
        ]
        
        return suggestions
    
    def show_performance_settings(self):
        """Show performance configuration settings"""
        
        st.subheader("⚙️ Performance Settings")
        
        # Performance targets
        st.markdown("### 🎯 Performance Targets")
        
        col1, col2 = st.columns(2)
        
        with col1:
            new_total_target = st.number_input(
                "Total Time Target (seconds):",
                min_value=1.0,
                max_value=10.0,
                value=st.session_state.performance_targets['total_time'],
                step=0.1
            )
            
            new_preprocessing_target = st.number_input(
                "Preprocessing Target (seconds):",
                min_value=0.1,
                max_value=2.0,
                value=st.session_state.performance_targets['preprocessing_time'],
                step=0.1
            )
        
        with col2:
            new_feature_target = st.number_input(
                "Feature Extraction Target (seconds):",
                min_value=0.1,
                max_value=5.0,
                value=st.session_state.performance_targets['feature_extraction_time'],
                step=0.1
            )
            
            new_prediction_target = st.number_input(
                "Prediction Target (seconds):",
                min_value=0.1,
                max_value=2.0,
                value=st.session_state.performance_targets['prediction_time'],
                step=0.1
            )
        
        if st.button("Update Targets"):
            st.session_state.performance_targets.update({
                'total_time': new_total_target,
                'preprocessing_time': new_preprocessing_target,
                'feature_extraction_time': new_feature_target,
                'prediction_time': new_prediction_target
            })
            st.success("Performance targets updated!")
        
        # Performance modes
        st.markdown("### ⚡ Performance Modes")
        
        performance_mode = st.selectbox(
            "Select Performance Mode:",
            ["Balanced", "Speed Optimized", "Accuracy Optimized", "Clinical Optimized"],
            help="Different modes optimize for different priorities"
        )
        
        mode_descriptions = {
            "Balanced": "Balanced speed and accuracy for general use",
            "Speed Optimized": "Maximum speed, minimal features, good for screening",
            "Accuracy Optimized": "Maximum accuracy, all features, slower processing",
            "Clinical Optimized": "Optimized for clinical decision support"
        }
        
        st.info(mode_descriptions[performance_mode])
        
        # Clear performance history
        st.markdown("### 🗑️ Data Management")
        
        if st.button("Clear Performance History", type="secondary"):
            st.session_state.performance_history = []
            st.success("Performance history cleared!")

# Global instance
performance_monitor = PerformanceMonitor()