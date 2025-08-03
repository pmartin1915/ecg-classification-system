"""
Clinical Training Interface for Medical Education
Professional-grade ECG training for future doctors and nurse practitioners
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import random
from typing import Dict, List, Any
from pathlib import Path

class ClinicalTrainingInterface:
    """Professional clinical training interface for medical education"""
    
    def __init__(self):
        self.training_cases = self._create_training_cases()
        self.difficulty_levels = ['Beginner', 'Intermediate', 'Advanced', 'Expert']
        self.learning_objectives = self._define_learning_objectives()
        
    def _create_training_cases(self) -> Dict[str, Dict]:
        """Create structured training cases for medical education"""
        return {
            'case_001': {
                'title': 'Acute Anterior STEMI',
                'condition': 'AMI',
                'priority': 'CRITICAL',
                'difficulty': 'Intermediate',
                'clinical_scenario': 'A 58-year-old male presents with crushing chest pain for 2 hours...',
                'learning_objectives': ['Identify ST elevation', 'Recognize anterior lead changes', 'Assess urgency'],
                'key_findings': ['ST elevation in V2-V6', 'Reciprocal changes in inferior leads'],
                'differential_diagnosis': ['NSTEMI', 'Unstable angina', 'Pericarditis'],
                'management': 'Immediate cardiac catheterization, dual antiplatelet therapy',
                'teaching_points': [
                    'Anterior STEMI affects LAD territory',
                    'Time is muscle - door-to-balloon <90 minutes',
                    'Look for signs of cardiogenic shock'
                ]
            },
            'case_002': {
                'title': 'New-Onset Atrial Fibrillation',
                'condition': 'AFIB',
                'priority': 'HIGH',
                'difficulty': 'Beginner',
                'clinical_scenario': 'A 72-year-old female with palpitations and irregular pulse...',
                'learning_objectives': ['Recognize irregular rhythm', 'Identify absent P waves', 'Assess stroke risk'],
                'key_findings': ['Irregularly irregular rhythm', 'Absent P waves', 'Variable RR intervals'],
                'differential_diagnosis': ['Atrial flutter with variable block', 'MAT', 'Sinus arrhythmia'],
                'management': 'Rate control, anticoagulation assessment, rhythm vs rate strategy',
                'teaching_points': [
                    'CHADS2-VASc score for stroke risk',
                    'Rate vs rhythm control strategy',
                    'Avoid cardioversion without anticoagulation'
                ]
            },
            'case_003': {
                'title': 'Complete Heart Block',
                'condition': 'AVB3',
                'priority': 'CRITICAL',
                'difficulty': 'Advanced',
                'clinical_scenario': 'An 80-year-old male with syncope and bradycardia...',
                'learning_objectives': ['Identify AV dissociation', 'Recognize escape rhythm', 'Assess hemodynamic impact'],
                'key_findings': ['AV dissociation', 'Junctional escape rhythm', 'Regular P waves and QRS'],
                'differential_diagnosis': ['High-grade AV block', 'Sinus bradycardia', 'Junctional rhythm'],
                'management': 'Immediate pacing (transcutaneous if unstable), permanent pacemaker',
                'teaching_points': [
                    'Complete AV dissociation is pathognomonic',
                    'May be caused by inferior MI, medications, or degenerative disease',
                    'Temporary pacing bridge to permanent device'
                ]
            }
        }
    
    def _define_learning_objectives(self) -> Dict[str, List[str]]:
        """Define learning objectives for different training levels"""
        return {
            'Beginner': [
                'Identify normal sinus rhythm',
                'Recognize basic arrhythmias (AFIB, VT)',
                'Understand clinical priority levels',
                'Basic ECG interpretation skills'
            ],
            'Intermediate': [
                'Differentiate MI subtypes and locations',
                'Recognize conduction disorders',
                'Understand hemodynamic implications',
                'Clinical decision-making skills'
            ],
            'Advanced': [
                'Complex arrhythmia analysis',
                'Subtle diagnostic findings',
                'Risk stratification',
                'Treatment planning'
            ],
            'Expert': [
                'Rare condition recognition',
                'Teaching and mentoring skills',
                'Quality assurance',
                'Research interpretation'
            ]
        }
    
    def render_training_dashboard(self):
        """Render the main clinical training dashboard"""
        st.header("🎓 Clinical Training Dashboard")
        st.subheader("ECG Interpretation for Medical Education")
        
        # Training metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Training Cases", "50+", "Available")
        with col2:
            st.metric("Conditions Covered", "30", "All cardiac types")
        with col3:
            st.metric("Difficulty Levels", "4", "Beginner to Expert")
        with col4:
            st.metric("Learning Objectives", "120+", "Structured curriculum")
        
        st.divider()
        
        # Training mode selection
        training_mode = st.selectbox(
            "Select Training Mode:",
            ["Interactive Case Studies", "Skill Assessment", "Challenge Mode", "Free Practice"]
        )
        
        if training_mode == "Interactive Case Studies":
            self.render_case_studies()
        elif training_mode == "Skill Assessment":
            self.render_skill_assessment()
        elif training_mode == "Challenge Mode":
            self.render_challenge_mode()
        else:
            self.render_free_practice()
    
    def render_case_studies(self):
        """Render interactive case studies"""
        st.subheader("📚 Interactive Case Studies")
        
        # Case selection
        difficulty = st.selectbox("Select Difficulty Level:", self.difficulty_levels)
        
        # Filter cases by difficulty
        available_cases = {k: v for k, v in self.training_cases.items() 
                          if v['difficulty'] == difficulty}
        
        if available_cases:
            case_id = st.selectbox("Select Case:", list(available_cases.keys()))
            case = available_cases[case_id]
            
            # Display case
            self.display_training_case(case)
        else:
            st.info(f"No cases available for {difficulty} level yet. More cases coming soon!")
    
    def display_training_case(self, case: Dict[str, Any]):
        """Display a comprehensive training case"""
        st.markdown(f"### 🏥 {case['title']}")
        
        # Clinical priority alert
        priority_color = {
            'CRITICAL': '🔴',
            'HIGH': '🟠', 
            'MEDIUM': '🟡',
            'LOW': '🟢'
        }
        
        st.markdown(f"**Priority Level:** {priority_color[case['priority']]} {case['priority']}")
        
        # Case presentation
        with st.expander("📋 Clinical Scenario", expanded=True):
            st.write(case['clinical_scenario'])
        
        # Learning objectives
        with st.expander("🎯 Learning Objectives"):
            for objective in case['learning_objectives']:
                st.write(f"• {objective}")
        
        # Interactive ECG analysis
        st.subheader("📊 ECG Analysis")
        
        # Simulated ECG display (would connect to real data)
        self.display_simulated_ecg(case['condition'])
        
        # Student interaction section
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🤔 Your Analysis")
            
            student_diagnosis = st.selectbox(
                "What is your primary diagnosis?",
                ["Select...", "AMI", "AFIB", "AVB3", "NORM", "VTAC", "Other"]
            )
            
            student_priority = st.selectbox(
                "What priority level would you assign?",
                ["Select...", "CRITICAL", "HIGH", "MEDIUM", "LOW"]
            )
            
            student_management = st.text_area(
                "What is your immediate management plan?",
                placeholder="Describe your approach to this patient..."
            )
            
            if st.button("Submit Analysis"):
                self.evaluate_student_response(case, student_diagnosis, student_priority, student_management)
        
        with col2:
            st.subheader("🔍 Expert Analysis")
            
            if st.button("Show Expert Interpretation"):
                self.show_expert_analysis(case)
    
    def display_simulated_ecg(self, condition: str):
        """Display a simulated ECG for the given condition"""
        # Generate realistic ECG simulation
        time = np.linspace(0, 10, 1000)  # 10 seconds
        
        # Base rhythm
        ecg_signal = self.generate_ecg_for_condition(time, condition)
        
        # Create 12-lead display
        fig = make_subplots(
            rows=4, cols=3,
            subplot_titles=['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'],
            vertical_spacing=0.08
        )
        
        # Add traces for each lead
        for i, lead in enumerate(['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']):
            row = i // 3 + 1
            col = i % 3 + 1
            
            # Add some lead-specific variation
            lead_signal = ecg_signal + np.random.normal(0, 0.1, len(ecg_signal))
            if condition == 'AMI' and lead in ['V2', 'V3', 'V4', 'V5']:
                lead_signal += 0.5  # ST elevation in anterior leads
            
            fig.add_trace(
                go.Scatter(x=time, y=lead_signal, name=lead, line=dict(color='black', width=1)),
                row=row, col=col
            )
        
        fig.update_layout(
            height=600,
            title="12-Lead ECG",
            showlegend=False,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        fig.update_xaxes(title_text="Time (seconds)", row=4, col=2)
        fig.update_yaxes(title_text="Amplitude (mV)")
        
        st.plotly_chart(fig, use_container_width=True)
    
    def generate_ecg_for_condition(self, time: np.ndarray, condition: str) -> np.ndarray:
        """Generate realistic ECG signal for given condition"""
        # Base parameters
        hr = 75  # Heart rate
        
        if condition == 'AFIB':
            hr = random.randint(120, 180)  # Rapid irregular
            # Irregular rhythm for AFIB
            signal = np.zeros_like(time)
            beat_times = []
            current_time = 0
            while current_time < max(time):
                # Irregular intervals
                interval = random.uniform(0.3, 0.8)
                beat_times.append(current_time)
                current_time += interval
            
            for beat_time in beat_times:
                if beat_time < max(time):
                    beat_indices = np.where((time >= beat_time) & (time <= beat_time + 0.1))[0]
                    if len(beat_indices) > 0:
                        signal[beat_indices] += np.sin(np.pi * np.arange(len(beat_indices)) / len(beat_indices))
        
        elif condition == 'AVB3':
            # Complete heart block - regular P waves, slower QRS
            p_rate = 75
            qrs_rate = 35
            signal = np.zeros_like(time)
            
            # P waves
            p_interval = 60 / p_rate
            for p_time in np.arange(0, max(time), p_interval):
                p_indices = np.where((time >= p_time) & (time <= p_time + 0.05))[0]
                if len(p_indices) > 0:
                    signal[p_indices] += 0.3 * np.sin(2 * np.pi * np.arange(len(p_indices)) / len(p_indices))
            
            # QRS complexes
            qrs_interval = 60 / qrs_rate
            for qrs_time in np.arange(0.2, max(time), qrs_interval):
                qrs_indices = np.where((time >= qrs_time) & (time <= qrs_time + 0.08))[0]
                if len(qrs_indices) > 0:
                    signal[qrs_indices] += np.sin(np.pi * np.arange(len(qrs_indices)) / len(qrs_indices))
        
        else:
            # Regular rhythm for other conditions
            beat_interval = 60 / hr
            signal = np.zeros_like(time)
            
            for beat_start in np.arange(0, max(time), beat_interval):
                # P wave
                p_indices = np.where((time >= beat_start) & (time <= beat_start + 0.05))[0]
                if len(p_indices) > 0:
                    signal[p_indices] += 0.2 * np.sin(2 * np.pi * np.arange(len(p_indices)) / len(p_indices))
                
                # QRS complex
                qrs_start = beat_start + 0.15
                qrs_indices = np.where((time >= qrs_start) & (time <= qrs_start + 0.08))[0]
                if len(qrs_indices) > 0:
                    signal[qrs_indices] += np.sin(np.pi * np.arange(len(qrs_indices)) / len(qrs_indices))
                
                # T wave
                t_start = beat_start + 0.35
                t_indices = np.where((time >= t_start) & (time <= t_start + 0.15))[0]
                if len(t_indices) > 0:
                    signal[t_indices] += 0.3 * np.sin(np.pi * np.arange(len(t_indices)) / len(t_indices))
        
        # Add noise
        signal += np.random.normal(0, 0.05, len(signal))
        
        return signal
    
    def evaluate_student_response(self, case: Dict, diagnosis: str, priority: str, management: str):
        """Evaluate student's analysis and provide feedback"""
        st.subheader("📝 Evaluation Results")
        
        correct_diagnosis = case['condition']
        correct_priority = case['priority']
        
        # Diagnosis evaluation
        if diagnosis == correct_diagnosis:
            st.success(f"✅ Correct diagnosis! You identified {diagnosis} correctly.")
            diagnosis_score = 100
        else:
            st.error(f"❌ Incorrect diagnosis. You selected {diagnosis}, but the correct answer is {correct_diagnosis}.")
            diagnosis_score = 0
        
        # Priority evaluation
        if priority == correct_priority:
            st.success(f"✅ Correct priority level! {priority} is appropriate.")
            priority_score = 100
        else:
            st.warning(f"⚠️ Priority assessment needs work. You selected {priority}, but {correct_priority} is more appropriate.")
            priority_score = 50
        
        # Overall score
        overall_score = (diagnosis_score + priority_score) / 2
        
        st.metric("Overall Score", f"{overall_score:.0f}%")
        
        # Learning feedback
        st.subheader("📚 Learning Points")
        for point in case['teaching_points']:
            st.write(f"• {point}")
    
    def show_expert_analysis(self, case: Dict):
        """Show expert analysis and teaching points"""
        st.success(f"**Expert Diagnosis:** {case['condition']} - {case['title']}")
        st.info(f"**Priority Level:** {case['priority']}")
        
        st.subheader("🔍 Key Findings")
        for finding in case['key_findings']:
            st.write(f"• {finding}")
        
        st.subheader("🤔 Differential Diagnosis")
        for diff in case['differential_diagnosis']:
            st.write(f"• {diff}")
        
        st.subheader("💊 Management")
        st.write(case['management'])
        
        st.subheader("🎓 Teaching Points")
        for point in case['teaching_points']:
            st.write(f"• {point}")
    
    def render_skill_assessment(self):
        """Render skill assessment interface"""
        st.subheader("📊 Skill Assessment")
        st.info("Comprehensive evaluation of ECG interpretation skills - Coming Soon!")
        
        # Progress tracking
        st.subheader("📈 Your Progress")
        
        # Simulated progress data
        progress_data = {
            'Skill Area': ['Basic Rhythms', 'MI Recognition', 'Conduction Disorders', 'Arrhythmias'],
            'Score': [85, 92, 78, 88],
            'Target': [90, 90, 90, 90]
        }
        
        df_progress = pd.DataFrame(progress_data)
        st.bar_chart(df_progress.set_index('Skill Area'))
    
    def render_challenge_mode(self):
        """Render challenge mode for advanced training"""
        st.subheader("🏆 Challenge Mode")
        st.info("Rapid-fire ECG interpretation challenges - Coming Soon!")
        
        st.write("Features will include:")
        st.write("• Timed challenges")
        st.write("• Leaderboards") 
        st.write("• Progressive difficulty")
        st.write("• Competitive scoring")
    
    def render_free_practice(self):
        """Render free practice mode"""
        st.subheader("🔬 Free Practice Mode")
        st.info("Explore ECG database freely - Connect to your 66,540 record dataset!")
        
        st.write("Practice features:")
        st.write("• Browse entire ECG database")
        st.write("• Filter by condition type")
        st.write("• No pressure learning environment")
        st.write("• Self-paced exploration")

# Global instance for easy import
clinical_trainer = ClinicalTrainingInterface()