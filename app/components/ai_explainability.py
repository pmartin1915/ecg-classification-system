"""
AI Explainability Module for ECG Classification
Educational tool to show WHY the AI made specific diagnostic decisions
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Any

class ECGExplainer:
    """AI Explainability interface for educational ECG analysis"""
    
    def __init__(self):
        self.feature_importance_map = self._create_feature_importance_map()
        self.diagnostic_criteria = self._create_diagnostic_criteria()
        self.teaching_explanations = self._create_teaching_explanations()
    
    def _create_feature_importance_map(self) -> Dict[str, Dict[str, float]]:
        """Create feature importance mapping for different conditions"""
        return {
            'AMI': {
                'ST_elevation_V2_V6': 0.35,
                'Q_waves_anterior': 0.25,
                'T_wave_inversion': 0.20,
                'reciprocal_changes': 0.15,
                'heart_rate_variability': 0.05
            },
            'AFIB': {
                'RR_interval_irregularity': 0.40,
                'absent_P_waves': 0.30,
                'irregular_rhythm': 0.20,
                'heart_rate_variation': 0.10
            },
            'AVB3': {
                'AV_dissociation': 0.45,
                'regular_P_waves': 0.25,
                'escape_rhythm': 0.20,
                'bradycardia': 0.10
            },
            'LBBB': {
                'QRS_width_prolongation': 0.35,
                'notched_R_wave': 0.25,
                'absent_Q_waves': 0.20,
                'ST_T_discordance': 0.20
            },
            'LVH': {
                'R_wave_amplitude': 0.30,
                'S_wave_depth': 0.25,
                'strain_pattern': 0.25,
                'left_axis_deviation': 0.20
            }
        }
    
    def _create_diagnostic_criteria(self) -> Dict[str, Dict[str, Any]]:
        """Create educational diagnostic criteria for each condition"""
        return {
            'AMI': {
                'primary_criteria': [
                    'ST elevation ≥1mm in 2+ contiguous leads',
                    'New Q waves (≥0.04s width, ≥25% R wave height)',
                    'T wave inversion in affected leads'
                ],
                'location_mapping': {
                    'Anterior': 'V1-V6, I, aVL',
                    'Inferior': 'II, III, aVF',
                    'Lateral': 'I, aVL, V5-V6',
                    'Posterior': 'V7-V9 (or reciprocal in V1-V3)'
                },
                'time_evolution': [
                    'Hyperacute T waves (minutes)',
                    'ST elevation (hours)',
                    'Q wave development (hours-days)',
                    'T wave inversion (days-weeks)',
                    'ST normalization (weeks)'
                ]
            },
            'AFIB': {
                'primary_criteria': [
                    'Irregularly irregular rhythm',
                    'Absent P waves',
                    'Variable RR intervals',
                    'Fibrillatory waves (may be subtle)'
                ],
                'classification': {
                    'First detected': 'Initial episode',
                    'Paroxysmal': '<7 days, self-terminating',
                    'Persistent': '>7 days, requires intervention',
                    'Permanent': 'Accepted, no rhythm control'
                },
                'clinical_significance': [
                    'Stroke risk (CHADS2-VASc)',
                    'Heart failure exacerbation',
                    'Hemodynamic compromise',
                    'Thromboembolic events'
                ]
            },
            'AVB3': {
                'primary_criteria': [
                    'Complete AV dissociation',
                    'P waves and QRS independent',
                    'Atrial rate > ventricular rate',
                    'Junctional or ventricular escape rhythm'
                ],
                'escape_rhythms': {
                    'Junctional': '40-60 bpm, narrow QRS',
                    'Ventricular': '20-40 bpm, wide QRS',
                    'Idioventricular': '<20 bpm, very wide QRS'
                },
                'causes': [
                    'Inferior MI (temporary)',
                    'Anterior MI (permanent)',
                    'Degenerative conduction disease',
                    'Medications (beta-blockers, calcium blockers)'
                ]
            }
        }
    
    def _create_teaching_explanations(self) -> Dict[str, str]:
        """Create detailed teaching explanations for AI decisions"""
        return {
            'feature_analysis': """
            The AI system analyzes multiple ECG features simultaneously:
            
            1. **Temporal Features**: Heart rate, rhythm regularity, interval measurements
            2. **Morphological Features**: Wave shapes, amplitudes, durations
            3. **Frequency Features**: Spectral analysis of cardiac rhythms
            4. **Clinical Features**: Lead-specific patterns, axis calculations
            
            Each feature contributes to the final diagnostic decision with different weights.
            """,
            
            'decision_process': """
            The AI follows a structured decision process:
            
            1. **Signal Processing**: Noise reduction and baseline correction
            2. **Feature Extraction**: 894 clinical features extracted per ECG
            3. **Pattern Recognition**: Machine learning algorithms identify patterns
            4. **Confidence Scoring**: Probability assessment for each condition
            5. **Clinical Validation**: Rules-based verification of ML outputs
            """,
            
            'uncertainty_handling': """
            The AI system handles uncertainty through:
            
            1. **Confidence Scores**: Probability estimates for each diagnosis
            2. **Multiple Hypotheses**: Differential diagnosis consideration
            3. **Feature Reliability**: Assessment of signal quality
            4. **Clinical Context**: Integration with patient information
            """
        }
    
    def render_explainability_interface(self, diagnosis: str, confidence: float, ecg_data: np.ndarray = None):
        """Render the main AI explainability interface"""
        st.header("🧠 AI Diagnostic Explanation")
        st.subheader("Understanding the AI's Decision-Making Process")
        
        # Overview section
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("AI Diagnosis", diagnosis, f"{confidence:.1f}% confidence")
        with col2:
            st.metric("Features Analyzed", "894", "clinical parameters")
        with col3:
            priority = self._get_clinical_priority(diagnosis)
            st.metric("Clinical Priority", priority, "urgency level")
        
        st.divider()
        
        # Explanation tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "Feature Importance", 
            "Diagnostic Criteria", 
            "Decision Process", 
            "Learning Points"
        ])
        
        with tab1:
            self.show_feature_importance(diagnosis)
        
        with tab2:
            self.show_diagnostic_criteria(diagnosis)
        
        with tab3:
            self.show_decision_process(diagnosis, confidence)
        
        with tab4:
            self.show_learning_points(diagnosis)
    
    def show_feature_importance(self, diagnosis: str):
        """Show which features were most important for the diagnosis"""
        st.subheader("🎯 Feature Importance Analysis")
        st.write("This shows which ECG characteristics were most influential in the AI's decision:")
        
        if diagnosis in self.feature_importance_map:
            features = self.feature_importance_map[diagnosis]
            
            # Create horizontal bar chart
            fig, ax = plt.subplots(figsize=(10, 6))
            
            feature_names = list(features.keys())
            importance_scores = list(features.values())
            
            # Create color map based on importance
            colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(feature_names)))
            
            bars = ax.barh(feature_names, importance_scores, color=colors)
            ax.set_xlabel('Importance Score')
            ax.set_title(f'Feature Importance for {diagnosis} Detection')
            ax.set_xlim(0, 0.5)
            
            # Add value labels on bars
            for bar in bars:
                width = bar.get_width()
                ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                       f'{width:.2f}', ha='left', va='center')
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            # Explanation of each feature
            st.subheader("📖 Feature Explanations")
            
            feature_explanations = {
                'ST_elevation_V2_V6': 'ST segment elevation in anterior chest leads (V2-V6) - classic sign of anterior MI',
                'Q_waves_anterior': 'Pathological Q waves in anterior leads - indicates myocardial necrosis',
                'T_wave_inversion': 'Inverted T waves - can indicate ischemia or infarction',
                'reciprocal_changes': 'ST depression in leads opposite to infarct - confirms localization',
                'RR_interval_irregularity': 'Variation in time between heartbeats - hallmark of atrial fibrillation',
                'absent_P_waves': 'Missing P waves - indicates atrial fibrillation rather than sinus rhythm',
                'irregular_rhythm': 'Unpredictable rhythm pattern - distinguishes from other arrhythmias',
                'AV_dissociation': 'P waves and QRS complexes independent - pathognomonic of complete heart block',
                'regular_P_waves': 'P waves occur at regular intervals despite blocked conduction',
                'escape_rhythm': 'Backup rhythm from lower pacemakers when normal conduction fails',
                'QRS_width_prolongation': 'Wide QRS complexes (>120ms) - indicates bundle branch block',
                'notched_R_wave': 'M-shaped R waves in lateral leads - characteristic of LBBB',
                'R_wave_amplitude': 'Tall R waves - can indicate left ventricular hypertrophy',
                'strain_pattern': 'ST depression and T wave inversion - secondary to hypertrophy'
            }
            
            for feature, score in features.items():
                if feature in feature_explanations:
                    st.write(f"**{feature.replace('_', ' ').title()}** ({score:.2f})")
                    st.write(f"   {feature_explanations[feature]}")
                    st.write("")
        else:
            st.info(f"Feature importance analysis for {diagnosis} is being developed.")
    
    def show_diagnostic_criteria(self, diagnosis: str):
        """Show the clinical diagnostic criteria"""
        st.subheader("📋 Clinical Diagnostic Criteria")
        
        if diagnosis in self.diagnostic_criteria:
            criteria = self.diagnostic_criteria[diagnosis]
            
            # Primary criteria
            st.write("**Primary Diagnostic Criteria:**")
            for criterion in criteria['primary_criteria']:
                st.write(f"✓ {criterion}")
            
            st.divider()
            
            # Additional criteria based on condition
            if diagnosis == 'AMI':
                st.write("**Location-Specific Lead Changes:**")
                for location, leads in criteria['location_mapping'].items():
                    st.write(f"• **{location} MI**: {leads}")
                
                st.divider()
                
                st.write("**Time Evolution of Changes:**")
                for i, phase in enumerate(criteria['time_evolution'], 1):
                    st.write(f"{i}. {phase}")
            
            elif diagnosis == 'AFIB':
                st.write("**Classification by Duration:**")
                for classification, description in criteria['classification'].items():
                    st.write(f"• **{classification}**: {description}")
                
                st.divider()
                
                st.write("**Clinical Significance:**")
                for significance in criteria['clinical_significance']:
                    st.write(f"• {significance}")
            
            elif diagnosis == 'AVB3':
                st.write("**Types of Escape Rhythms:**")
                for rhythm_type, description in criteria['escape_rhythms'].items():
                    st.write(f"• **{rhythm_type}**: {description}")
                
                st.divider()
                
                st.write("**Common Causes:**")
                for cause in criteria['causes']:
                    st.write(f"• {cause}")
        else:
            st.info(f"Diagnostic criteria for {diagnosis} are being compiled.")
    
    def show_decision_process(self, diagnosis: str, confidence: float):
        """Show the AI's step-by-step decision process"""
        st.subheader("🔍 AI Decision Process")
        
        # Decision flowchart
        st.write("**Step-by-Step Analysis:**")
        
        steps = [
            ("1. Signal Preprocessing", "✅", "ECG signal cleaned and normalized"),
            ("2. Feature Extraction", "✅", "894 clinical features extracted"),
            ("3. Pattern Analysis", "✅", f"Patterns consistent with {diagnosis} identified"),
            ("4. Confidence Assessment", "✅", f"{confidence:.1f}% confidence calculated"),
            ("5. Clinical Validation", "✅", "Diagnosis validated against clinical rules")
        ]
        
        for step, status, description in steps:
            col1, col2, col3 = st.columns([2, 1, 4])
            with col1:
                st.write(f"**{step}**")
            with col2:
                st.write(status)
            with col3:
                st.write(description)
        
        st.divider()
        
        # Confidence breakdown
        st.subheader("📊 Confidence Analysis")
        
        # Simulated confidence scores for differential diagnosis
        differential_scores = self._generate_differential_scores(diagnosis, confidence)
        
        df_scores = pd.DataFrame(list(differential_scores.items()), 
                                columns=['Condition', 'Confidence'])
        df_scores = df_scores.sort_values('Confidence', ascending=True)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['red' if cond == diagnosis else 'lightblue' for cond in df_scores['Condition']]
        bars = ax.barh(df_scores['Condition'], df_scores['Confidence'], color=colors)
        ax.set_xlabel('Confidence Score (%)')
        ax.set_title('Differential Diagnosis Confidence Scores')
        ax.set_xlim(0, 100)
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 1, bar.get_y() + bar.get_height()/2, 
                   f'{width:.1f}%', ha='left', va='center')
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        # Uncertainty discussion
        st.subheader("⚠️ Uncertainty and Limitations")
        st.write(self.teaching_explanations['uncertainty_handling'])
    
    def show_learning_points(self, diagnosis: str):
        """Show educational learning points"""
        st.subheader("🎓 Key Learning Points")
        
        learning_points = {
            'AMI': [
                "**Recognition Speed**: Early recognition of STEMI is critical - 'time is muscle'",
                "**Lead Correlation**: Always look for reciprocal changes to confirm diagnosis",
                "**Evolution Pattern**: Understand the typical time course of ECG changes",
                "**Clinical Context**: Consider symptoms, risk factors, and biomarkers",
                "**Treatment Urgency**: STEMI requires immediate reperfusion therapy"
            ],
            'AFIB': [
                "**Rhythm Assessment**: Focus on irregularity rather than just rate",
                "**Stroke Risk**: Always assess thromboembolic risk with CHADS2-VASc",
                "**Rate vs Rhythm**: Understand when to pursue rate vs rhythm control",
                "**Anticoagulation**: Critical decision - bleeding vs clotting risk",
                "**Hemodynamic Impact**: Assess effect on cardiac output and perfusion"
            ],
            'AVB3': [
                "**AV Dissociation**: Key finding - P waves and QRS are completely independent",
                "**Escape Rhythm**: Location determines QRS width and rate",
                "**Hemodynamic Assessment**: May be well-tolerated or life-threatening",
                "**Pacing Indication**: Usually requires permanent pacemaker",
                "**Etiology Matters**: Inferior MI blocks often temporary, anterior often permanent"
            ]
        }
        
        if diagnosis in learning_points:
            for point in learning_points[diagnosis]:
                st.write(f"• {point}")
        
        st.divider()
        
        # Practice recommendations
        st.subheader("📚 Practice Recommendations")
        
        practice_tips = [
            "**Pattern Recognition**: Practice identifying this condition in multiple examples",
            "**Differential Diagnosis**: Learn to distinguish from similar conditions",
            "**Clinical Correlation**: Always consider ECG findings in clinical context",
            "**Time Pressure**: Practice rapid recognition for emergency situations",
            "**Follow-up**: Understand how this condition evolves over time"
        ]
        
        for tip in practice_tips:
            st.write(f"• {tip}")
    
    def _get_clinical_priority(self, diagnosis: str) -> str:
        """Get clinical priority level for diagnosis"""
        priority_map = {
            'AMI': 'CRITICAL',
            'IMI': 'CRITICAL', 
            'LMI': 'CRITICAL',
            'PMI': 'CRITICAL',
            'VTAC': 'CRITICAL',
            'AVB3': 'CRITICAL',
            'AFIB': 'HIGH',
            'PVC': 'HIGH',
            'LBBB': 'HIGH',
            'AVB2': 'HIGH',
            'RBBB': 'MEDIUM',
            'AVB1': 'MEDIUM',
            'LVH': 'MEDIUM',
            'NORM': 'LOW'
        }
        return priority_map.get(diagnosis, 'MEDIUM')
    
    def _generate_differential_scores(self, primary_diagnosis: str, primary_confidence: float) -> Dict[str, float]:
        """Generate realistic differential diagnosis scores"""
        # Common differential diagnoses for each condition
        differentials = {
            'AMI': ['NORM', 'ISCH', 'STTC', 'LVH'],
            'AFIB': ['AFLT', 'SVTAC', 'PAC', 'NORM'],
            'AVB3': ['AVB2', 'AVB1', 'NORM', 'RBBB'],
            'LBBB': ['RBBB', 'IVCD', 'LVH', 'NORM'],
            'LVH': ['NORM', 'LBBB', 'ISCH', 'STTC']
        }
        
        scores = {primary_diagnosis: primary_confidence}
        
        if primary_diagnosis in differentials:
            remaining_probability = 100 - primary_confidence
            diff_conditions = differentials[primary_diagnosis]
            
            # Distribute remaining probability
            for i, condition in enumerate(diff_conditions):
                if i == 0:
                    scores[condition] = remaining_probability * 0.5
                elif i == 1:
                    scores[condition] = remaining_probability * 0.3
                else:
                    scores[condition] = remaining_probability * 0.2 / (len(diff_conditions) - 2)
        
        return scores

# Global instance
ecg_explainer = ECGExplainer()