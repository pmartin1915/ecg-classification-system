"""
Enhanced AI Explainability Module with MI-Specific Diagnostic Reasoning
Advanced educational tool that explains AI decisions using clinical reasoning principles
"""
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Any, Optional
import time

class EnhancedECGExplainer:
    """
    Advanced AI explainability with MI-specific clinical reasoning
    Designed for medical education and clinical decision support
    """
    
    def __init__(self):
        self.mi_diagnostic_criteria = self._create_comprehensive_mi_criteria()
        self.clinical_reasoning_framework = self._create_clinical_reasoning_framework()
        self.feature_clinical_mapping = self._create_feature_clinical_mapping()
        self.uncertainty_explanations = self._create_uncertainty_explanations()
        self.teaching_scenarios = self._create_teaching_scenarios()
        
    def _create_comprehensive_mi_criteria(self) -> Dict[str, Dict[str, Any]]:
        """Create comprehensive MI diagnostic criteria with clinical context"""
        return {
            'AMI_Anterior': {
                'definition': 'Anterior Myocardial Infarction - LAD Territory',
                'primary_criteria': {
                    'ST_elevation': {
                        'threshold': '≥1mm (0.1mV) in precordial leads',
                        'leads': ['V1', 'V2', 'V3', 'V4'],
                        'clinical_significance': 'Indicates acute injury to anterior wall',
                        'urgency': 'CRITICAL - Door-to-balloon <90 minutes'
                    },
                    'Q_waves': {
                        'threshold': '≥0.04s width OR ≥25% of R wave height',
                        'leads': ['V1', 'V2', 'V3', 'V4'],
                        'clinical_significance': 'Indicates myocardial necrosis',
                        'timing': 'Develops hours to days after onset'
                    },
                    'reciprocal_changes': {
                        'pattern': 'ST depression in inferior leads',
                        'leads': ['II', 'III', 'aVF'],
                        'clinical_significance': 'Confirms anterior MI diagnosis',
                        'importance': 'Increases diagnostic confidence'
                    }
                },
                'vessel_territory': {
                    'primary_vessel': 'Left Anterior Descending (LAD)',
                    'anatomy': 'Supplies anterior wall and septum',
                    'complications': ['Cardiogenic shock', 'Complete heart block', 'Ventricular arrhythmias']
                },
                'differential_diagnosis': [
                    'Old anterior MI with persistent ST elevation',
                    'Left ventricular aneurysm',
                    'Brugada syndrome (V1-V2)',
                    'Early repolarization'
                ],
                'management_priorities': [
                    'STAT cardiology consultation',
                    'Prepare for primary PCI',
                    'Dual antiplatelet therapy',
                    'Monitor for arrhythmias'
                ]
            },
            'IMI_Inferior': {
                'definition': 'Inferior Myocardial Infarction - RCA/LCX Territory',
                'primary_criteria': {
                    'ST_elevation': {
                        'threshold': '≥1mm (0.1mV) in inferior leads',
                        'leads': ['II', 'III', 'aVF'],
                        'clinical_significance': 'Indicates acute injury to inferior wall',
                        'lead_specificity': 'aVF most sensitive for inferior MI'
                    },
                    'reciprocal_changes': {
                        'pattern': 'ST depression in lateral leads',
                        'leads': ['I', 'aVL'],
                        'clinical_significance': 'Confirms inferior MI diagnosis',
                        'diagnostic_value': 'High specificity for acute MI'
                    },
                    'conduction_effects': {
                        'pattern': 'AV blocks (especially 2nd/3rd degree)',
                        'mechanism': 'RCA supplies AV node',
                        'clinical_significance': 'May require temporary pacing',
                        'reversibility': 'Often temporary with reperfusion'
                    }
                },
                'vessel_territory': {
                    'primary_vessels': ['Right Coronary Artery (RCA)', 'Left Circumflex (LCX)'],
                    'anatomy': 'Supplies inferior wall and posterior-lateral wall',
                    'complications': ['AV blocks', 'RV infarction', 'Hypotension']
                },
                'special_considerations': {
                    'RV_involvement': {
                        'assessment': 'Right-sided ECG (V4R)',
                        'significance': 'Changes management approach',
                        'treatment': 'Avoid nitrates, maintain preload'
                    },
                    'posterior_extension': {
                        'assessment': 'Tall R waves in V1-V2',
                        'significance': 'Larger infarct territory',
                        'leads': 'Consider V7-V9 for confirmation'
                    }
                }
            },
            'NSTEMI': {
                'definition': 'Non-ST Elevation Myocardial Infarction',
                'primary_criteria': {
                    'ST_depression': {
                        'threshold': '≥0.5mm horizontal or downsloping',
                        'distribution': 'Multiple leads',
                        'clinical_significance': 'Indicates subendocardial ischemia',
                        'risk_stratification': 'Higher risk with more leads involved'
                    },
                    'T_wave_changes': {
                        'pattern': 'Deep symmetric T wave inversions',
                        'leads': 'Corresponding to culprit territory',
                        'clinical_significance': 'Indicates ischemia or recent infarction',
                        'temporal_evolution': 'May precede or follow ST changes'
                    }
                },
                'risk_stratification': {
                    'high_risk_features': [
                        'ST depression ≥2mm',
                        'Multiple lead involvement',
                        'Hemodynamic instability',
                        'Elevated troponins'
                    ],
                    'TIMI_risk_score': 'Essential for risk stratification',
                    'management_timing': 'Early invasive vs conservative strategy'
                }
            }
        }
    
    def _create_clinical_reasoning_framework(self) -> Dict[str, Any]:
        """Create clinical reasoning framework for ECG interpretation"""
        return {
            'systematic_approach': {
                'step_1_rate_rhythm': {
                    'assessment': 'Heart rate and rhythm analysis',
                    'normal_values': 'Rate: 60-100 bpm, Rhythm: Regular sinus',
                    'abnormal_significance': 'May indicate arrhythmias or conduction disorders',
                    'mi_relevance': 'Arrhythmias common in acute MI'
                },
                'step_2_axis_intervals': {
                    'assessment': 'Axis deviation and interval measurements',
                    'normal_values': 'Axis: -30° to +90°, PR: 0.12-0.20s, QRS: <0.12s',
                    'abnormal_significance': 'May indicate chamber enlargement or blocks',
                    'mi_relevance': 'New conduction delays suggest large MI'
                },
                'step_3_chamber_analysis': {
                    'assessment': 'Chamber enlargement or hypertrophy',
                    'criteria': 'Voltage criteria and morphology changes',
                    'abnormal_significance': 'Indicates chronic cardiac conditions',
                    'mi_relevance': 'Pre-existing LVH affects MI presentation'
                },
                'step_4_ischemia_injury': {
                    'assessment': 'ST-T changes indicating ischemia/injury',
                    'acute_patterns': 'ST elevation, depression, T wave changes',
                    'chronic_patterns': 'Q waves, poor R wave progression',
                    'mi_relevance': 'Core diagnostic criteria for MI'
                }
            },
            'diagnostic_confidence_levels': {
                'high_confidence': {
                    'threshold': '>85%',
                    'characteristics': 'Classic findings in multiple leads',
                    'clinical_action': 'Proceed with standard protocols',
                    'examples': ['STEMI with reciprocal changes', 'Complete heart block']
                },
                'moderate_confidence': {
                    'threshold': '65-85%',
                    'characteristics': 'Some criteria met, minor inconsistencies',
                    'clinical_action': 'Correlate with symptoms and biomarkers',
                    'examples': ['Borderline ST elevation', 'Non-specific T changes']
                },
                'low_confidence': {
                    'threshold': '<65%',
                    'characteristics': 'Equivocal findings, multiple differentials',
                    'clinical_action': 'Serial ECGs and clinical monitoring',
                    'examples': ['Early repolarization vs STEMI', 'Artifact vs arrhythmia']
                }
            }
        }
    
    def _create_feature_clinical_mapping(self) -> Dict[str, Dict[str, Any]]:
        """Map AI features to clinical findings with explanations"""
        return {
            'ST_elevation_features': {
                'V2_st_elevation': {
                    'clinical_name': 'ST Elevation in V2',
                    'anatomical_correlation': 'Anterior wall (LAD territory)',
                    'diagnostic_significance': 'Acute injury pattern',
                    'threshold_interpretation': {
                        '≥2mm': 'Definite acute MI',
                        '1-2mm': 'Probable acute MI',
                        '0.5-1mm': 'Possible ischemia',
                        '<0.5mm': 'Normal variant possible'
                    },
                    'differential_diagnosis': [
                        'Acute anterior STEMI',
                        'Old MI with persistent ST elevation',
                        'Left ventricular aneurysm',
                        'Brugada pattern'
                    ]
                },
                'inferior_st_elevation': {
                    'clinical_name': 'ST Elevation in II, III, aVF',
                    'anatomical_correlation': 'Inferior wall (RCA/LCX territory)',
                    'diagnostic_significance': 'Acute inferior MI',
                    'clinical_pearls': [
                        'III > II suggests RCA occlusion',
                        'Check for RV involvement with V4R',
                        'Watch for AV blocks'
                    ]
                }
            },
            'Q_wave_features': {
                'anterior_q_waves': {
                    'clinical_name': 'Pathological Q Waves V1-V4',
                    'diagnostic_significance': 'Myocardial necrosis',
                    'timing': 'Hours to days after MI onset',
                    'permanence': 'Usually permanent marker of old MI',
                    'differential': [
                        'Old anterior MI',
                        'Hypertrophic cardiomyopathy',
                        'Left ventricular hypertrophy',
                        'Normal septal Q waves (I, aVL, V5-V6)'
                    ]
                }
            },
            'rhythm_features': {
                'irregular_rhythm': {
                    'clinical_name': 'Irregular Heart Rhythm',
                    'common_causes': [
                        'Atrial fibrillation',
                        'Atrial flutter with variable block',
                        'Multifocal atrial tachycardia',
                        'Frequent premature beats'
                    ],
                    'diagnostic_approach': 'Look for P waves and pattern',
                    'clinical_significance': 'May complicate MI management'
                }
            }
        }
    
    def _create_uncertainty_explanations(self) -> Dict[str, Dict[str, Any]]:
        """Create explanations for handling diagnostic uncertainty"""
        return {
            'confidence_interpretation': {
                'high_confidence_mi': {
                    'range': '85-100%',
                    'interpretation': 'Classic MI pattern with high diagnostic certainty',
                    'clinical_action': 'Proceed with standard MI protocols',
                    'additional_testing': 'Troponins confirmatory, not required for initial treatment'
                },
                'moderate_confidence_mi': {
                    'range': '65-84%',
                    'interpretation': 'Probable MI but some atypical features present',
                    'clinical_action': 'Correlate with symptoms and biomarkers',
                    'additional_testing': 'Serial ECGs, troponins essential'
                },
                'low_confidence_mi': {
                    'range': '50-64%',
                    'interpretation': 'Possible MI but significant uncertainty',
                    'clinical_action': 'Close monitoring and serial assessments',
                    'additional_testing': 'Comprehensive cardiac workup indicated'
                },
                'unlikely_mi': {
                    'range': '<50%',
                    'interpretation': 'MI unlikely based on ECG pattern',
                    'clinical_action': 'Consider alternative diagnoses',
                    'additional_testing': 'Troponins may still be indicated if symptoms present'
                }
            },
            'common_uncertainties': {
                'borderline_st_elevation': {
                    'scenario': 'ST elevation 0.8-1.2mm in 2 leads',
                    'uncertainty_factors': ['Lead placement', 'Baseline variations', 'Early repolarization'],
                    'resolution_approach': 'Serial ECGs, posterior leads, clinical correlation'
                },
                'old_vs_new_changes': {
                    'scenario': 'Q waves present with ST elevation',
                    'uncertainty_factors': ['Age of infarct', 'Reinfarction', 'Ventricular aneurysm'],
                    'resolution_approach': 'Compare with old ECGs, troponin kinetics'
                },
                'artifact_vs_pathology': {
                    'scenario': 'Irregular baseline or unusual morphology',
                    'uncertainty_factors': ['Patient movement', 'Electrode issues', 'Muscle artifact'],
                    'resolution_approach': 'Repeat ECG with proper technique'
                }
            }
        }
    
    def _create_teaching_scenarios(self) -> Dict[str, Dict[str, Any]]:
        """Create interactive teaching scenarios for different experience levels"""
        return {
            'beginner_scenarios': {
                'obvious_stemi': {
                    'title': 'Classic Anterior STEMI',
                    'learning_objectives': [
                        'Recognize obvious ST elevation',
                        'Identify lead groupings',
                        'Understand urgency of treatment'
                    ],
                    'key_teaching_points': [
                        'ST elevation ≥1mm in 2+ contiguous leads = STEMI',
                        'V1-V4 = anterior wall (LAD territory)',
                        'Time is muscle - immediate intervention needed'
                    ]
                }
            },
            'intermediate_scenarios': {
                'subtle_inferior_mi': {
                    'title': 'Subtle Inferior MI with Reciprocal Changes',
                    'learning_objectives': [
                        'Recognize subtle ST elevation',
                        'Identify reciprocal changes',
                        'Understand diagnostic confirmation'
                    ],
                    'key_teaching_points': [
                        'Reciprocal changes increase diagnostic confidence',
                        'aVF most sensitive for inferior MI',
                        'Consider RV involvement and AV blocks'
                    ]
                }
            },
            'advanced_scenarios': {
                'nstemi_with_complications': {
                    'title': 'NSTEMI with High-Risk Features',
                    'learning_objectives': [
                        'Risk stratify non-ST elevation ACS',
                        'Recognize high-risk features',
                        'Plan appropriate management strategy'
                    ],
                    'key_teaching_points': [
                        'ST depression extent correlates with risk',
                        'Dynamic changes indicate instability',
                        'Early invasive strategy for high-risk features'
                    ]
                }
            }
        }
    
    def render_enhanced_explainability(self, diagnosis: str, confidence: float, 
                                     feature_data: Optional[Dict] = None,
                                     user_experience_level: str = "Intermediate") -> None:
        """
        Render enhanced AI explainability interface with MI-specific reasoning
        
        Args:
            diagnosis: AI diagnosis result
            confidence: Confidence level (0-1)
            feature_data: Feature importance data from prediction
            user_experience_level: User's experience level for tailored explanations
        """
        
        st.header("🧠 Enhanced AI Diagnostic Explanation")
        st.subheader("Understanding the AI's Clinical Reasoning Process")
        
        # Experience level selector
        col1, col2, col3 = st.columns(3)
        
        with col1:
            experience_level = st.selectbox(
                "Your Experience Level:",
                ["Beginner", "Intermediate", "Advanced", "Expert"],
                index=["Beginner", "Intermediate", "Advanced", "Expert"].index(user_experience_level)
            )
        
        with col2:
            explanation_depth = st.selectbox(
                "Explanation Depth:",
                ["Quick Summary", "Standard", "Comprehensive", "Teaching Mode"],
                index=1
            )
        
        with col3:
            focus_area = st.selectbox(
                "Focus Area:",
                ["General", "MI-Specific", "Arrhythmias", "Conduction", "All Conditions"],
                index=1 if 'MI' in diagnosis.upper() else 0
            )
        
        st.divider()
        
        # Main explanation tabs
        if explanation_depth == "Teaching Mode":
            self._render_teaching_mode(diagnosis, confidence, experience_level)
        else:
            self._render_standard_explanation(diagnosis, confidence, feature_data, 
                                            experience_level, explanation_depth, focus_area)
    
    def _render_teaching_mode(self, diagnosis: str, confidence: float, experience_level: str):
        """Render interactive teaching mode"""
        
        st.subheader("🎓 Interactive Teaching Mode")
        
        # Select appropriate teaching scenario
        scenarios = self._get_scenarios_for_level(experience_level)
        
        if scenarios:
            scenario_names = list(scenarios.keys())
            selected_scenario = st.selectbox("Select Teaching Scenario:", scenario_names)
            
            if selected_scenario:
                scenario = scenarios[selected_scenario]
                
                st.markdown(f"### 📚 {scenario['title']}")
                
                # Learning objectives
                st.markdown("**🎯 Learning Objectives:**")
                for objective in scenario['learning_objectives']:
                    st.write(f"• {objective}")
                
                # Interactive questions
                self._render_interactive_questions(diagnosis, scenario)
                
                # Key teaching points
                st.markdown("**💡 Key Teaching Points:**")
                for point in scenario['key_teaching_points']:
                    st.write(f"• {point}")
        
        # Case-based learning
        st.markdown("### 📋 Case-Based Learning")
        self._render_case_based_learning(diagnosis, confidence, experience_level)
    
    def _render_standard_explanation(self, diagnosis: str, confidence: float, 
                                   feature_data: Optional[Dict], experience_level: str,
                                   explanation_depth: str, focus_area: str):
        """Render standard explanation interface"""
        
        # Explanation tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🔍 Clinical Reasoning",
            "📊 Feature Analysis", 
            "⚠️ Confidence & Uncertainty",
            "📚 Educational Context",
            "🏥 Clinical Actions"
        ])
        
        with tab1:
            self._render_clinical_reasoning(diagnosis, confidence, experience_level)
        
        with tab2:
            self._render_feature_analysis(diagnosis, feature_data, experience_level)
        
        with tab3:
            self._render_confidence_analysis(diagnosis, confidence, experience_level)
        
        with tab4:
            self._render_educational_context(diagnosis, experience_level)
        
        with tab5:
            self._render_clinical_actions(diagnosis, confidence, experience_level)
    
    def _render_clinical_reasoning(self, diagnosis: str, confidence: float, experience_level: str):
        """Render clinical reasoning explanation"""
        
        st.subheader("🔍 Clinical Reasoning Process")
        
        # Step-by-step reasoning
        if 'MI' in diagnosis.upper() or diagnosis in ['AMI', 'IMI', 'LMI', 'PMI']:
            self._render_mi_specific_reasoning(diagnosis, confidence, experience_level)
        else:
            self._render_general_reasoning(diagnosis, confidence, experience_level)
        
        # Diagnostic confidence explanation
        st.markdown("### 🎯 Diagnostic Confidence Analysis")
        
        confidence_percent = confidence * 100
        confidence_category = self._get_confidence_category(confidence_percent)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if confidence_percent >= 85:
                st.success(f"**High Confidence: {confidence_percent:.1f}%**")
                st.write("✅ Classic diagnostic pattern present")
                st.write("✅ Multiple criteria satisfied")
                st.write("✅ Minimal uncertainty factors")
            elif confidence_percent >= 65:
                st.warning(f"**Moderate Confidence: {confidence_percent:.1f}%**")
                st.write("⚠️ Most criteria present")
                st.write("⚠️ Some atypical features")
                st.write("⚠️ Clinical correlation advised")
            else:
                st.error(f"**Low Confidence: {confidence_percent:.1f}%**")
                st.write("❌ Limited diagnostic criteria")
                st.write("❌ Significant uncertainty")
                st.write("❌ Alternative diagnoses possible")
        
        with col2:
            # Confidence interpretation
            confidence_info = self.uncertainty_explanations['confidence_interpretation']
            
            for category, info in confidence_info.items():
                conf_range = info['range']
                if self._confidence_in_range(confidence_percent, conf_range):
                    st.markdown(f"**Clinical Interpretation:**")
                    st.write(f"📋 {info['interpretation']}")
                    st.write(f"🏥 **Action:** {info['clinical_action']}")
                    st.write(f"🧪 **Testing:** {info['additional_testing']}")
                    break
    
    def _render_mi_specific_reasoning(self, diagnosis: str, confidence: float, experience_level: str):
        """Render MI-specific clinical reasoning"""
        
        st.markdown("### 🫀 MI-Specific Diagnostic Reasoning")
        
        # Determine MI type
        mi_type = self._determine_mi_type(diagnosis)
        
        if mi_type in self.mi_diagnostic_criteria:
            mi_criteria = self.mi_diagnostic_criteria[mi_type]
            
            # Clinical definition
            st.markdown(f"**📋 Definition:** {mi_criteria['definition']}")
            
            # Primary diagnostic criteria
            st.markdown("**🎯 Primary Diagnostic Criteria:**")
            
            for criterion_name, criterion_data in mi_criteria['primary_criteria'].items():
                
                with st.expander(f"🔍 {criterion_name.replace('_', ' ').title()}", expanded=True):
                    
                    if isinstance(criterion_data, dict):
                        if 'threshold' in criterion_data:
                            st.write(f"**Threshold:** {criterion_data['threshold']}")
                        if 'leads' in criterion_data:
                            st.write(f"**Leads:** {', '.join(criterion_data['leads'])}")
                        if 'clinical_significance' in criterion_data:
                            st.write(f"**Clinical Significance:** {criterion_data['clinical_significance']}")
                        if 'urgency' in criterion_data:
                            st.error(f"**⚡ {criterion_data['urgency']}**")
                    else:
                        st.write(criterion_data)
            
            # Vessel territory information
            if 'vessel_territory' in mi_criteria:
                vessel_info = mi_criteria['vessel_territory']
                
                st.markdown("**🩸 Vessel Territory & Anatomy:**")
                col1, col2 = st.columns(2)
                
                with col1:
                    if 'primary_vessel' in vessel_info:
                        st.write(f"**Primary Vessel:** {vessel_info['primary_vessel']}")
                    elif 'primary_vessels' in vessel_info:
                        st.write(f"**Primary Vessels:** {', '.join(vessel_info['primary_vessels'])}")
                    
                    if 'anatomy' in vessel_info:
                        st.write(f"**Anatomy:** {vessel_info['anatomy']}")
                
                with col2:
                    if 'complications' in vessel_info:
                        st.write("**Potential Complications:**")
                        for comp in vessel_info['complications']:
                            st.write(f"• {comp}")
            
            # Management priorities
            if 'management_priorities' in mi_criteria:
                st.markdown("**🏥 Management Priorities:**")
                for i, priority in enumerate(mi_criteria['management_priorities'], 1):
                    if i == 1:
                        st.error(f"{i}. {priority}")
                    elif i == 2:
                        st.warning(f"{i}. {priority}")
                    else:
                        st.info(f"{i}. {priority}")
    
    def _render_feature_analysis(self, diagnosis: str, feature_data: Optional[Dict], experience_level: str):
        """Render feature importance analysis"""
        
        st.subheader("📊 AI Feature Analysis")
        
        if feature_data:
            # Use real feature data
            self._render_real_feature_analysis(feature_data, diagnosis, experience_level)
        else:
            # Use simulated feature analysis
            self._render_simulated_feature_analysis(diagnosis, experience_level)
    
    def _render_simulated_feature_analysis(self, diagnosis: str, experience_level: str):
        """Render simulated feature analysis for demonstration"""
        
        st.markdown("### 🎯 Key ECG Features Analyzed")
        
        # Get relevant features for diagnosis
        if diagnosis in ['AMI', 'IMI', 'LMI'] or 'MI' in diagnosis.upper():
            features = self._get_mi_features_for_display(diagnosis)
        else:
            features = self._get_general_features_for_display(diagnosis)
        
        # Feature importance visualization
        if features:
            self._create_feature_importance_chart(features, diagnosis)
            
            # Detailed feature explanations
            st.markdown("### 🔍 Feature Explanations")
            
            for feature_name, importance in features.items():
                
                with st.expander(f"📈 {feature_name} (Importance: {importance:.1%})", expanded=importance > 0.2):
                    
                    # Get clinical mapping
                    clinical_info = self._get_clinical_info_for_feature(feature_name, diagnosis)
                    
                    if clinical_info:
                        st.write(f"**Clinical Name:** {clinical_info.get('clinical_name', feature_name)}")
                        
                        if 'anatomical_correlation' in clinical_info:
                            st.write(f"**Anatomical Correlation:** {clinical_info['anatomical_correlation']}")
                        
                        if 'diagnostic_significance' in clinical_info:
                            st.write(f"**Diagnostic Significance:** {clinical_info['diagnostic_significance']}")
                        
                        if 'threshold_interpretation' in clinical_info:
                            st.write("**Threshold Interpretation:**")
                            for threshold, meaning in clinical_info['threshold_interpretation'].items():
                                st.write(f"• {threshold}: {meaning}")
                    else:
                        st.write(f"This feature contributes {importance:.1%} to the diagnostic decision.")
    
    def _render_confidence_analysis(self, diagnosis: str, confidence: float, experience_level: str):
        """Render confidence and uncertainty analysis"""
        
        st.subheader("⚠️ Confidence & Uncertainty Analysis")
        
        # Confidence breakdown
        confidence_percent = confidence * 100
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Confidence gauge
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = confidence_percent,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Diagnostic Confidence"},
                delta = {'reference': 80},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 65], 'color': "yellow"},
                        {'range': [65, 85], 'color': "orange"},
                        {'range': [85, 100], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Uncertainty factors
            st.markdown("**🔍 Uncertainty Factors:**")
            
            uncertainty_factors = self._identify_uncertainty_factors(diagnosis, confidence)
            
            if uncertainty_factors:
                for factor in uncertainty_factors:
                    st.write(f"• {factor}")
            else:
                st.success("✅ No significant uncertainty factors identified")
            
            # Recommendations for uncertainty
            st.markdown("**💡 Recommendations:**")
            recommendations = self._get_uncertainty_recommendations(diagnosis, confidence)
            for rec in recommendations:
                st.write(f"• {rec}")
    
    def _render_educational_context(self, diagnosis: str, experience_level: str):
        """Render educational context and learning opportunities"""
        
        st.subheader("📚 Educational Context")
        
        # Learning objectives based on experience level
        learning_objectives = self._get_learning_objectives(diagnosis, experience_level)
        
        if learning_objectives:
            st.markdown("**🎯 Learning Objectives:**")
            for objective in learning_objectives:
                st.write(f"• {objective}")
        
        # Related conditions
        st.markdown("**🔗 Related Conditions to Consider:**")
        related_conditions = self._get_related_conditions(diagnosis)
        
        for condition in related_conditions:
            st.write(f"• **{condition['name']}**: {condition['description']}")
        
        # Practice recommendations
        st.markdown("**📖 Practice Recommendations:**")
        practice_tips = self._get_practice_recommendations(diagnosis, experience_level)
        
        for tip in practice_tips:
            st.write(f"• {tip}")
    
    def _render_clinical_actions(self, diagnosis: str, confidence: float, experience_level: str):
        """Render clinical actions and management recommendations"""
        
        st.subheader("🏥 Clinical Actions & Management")
        
        # Immediate actions
        immediate_actions = self._get_immediate_actions(diagnosis, confidence)
        
        if immediate_actions:
            st.markdown("**⚡ Immediate Actions:**")
            for i, action in enumerate(immediate_actions, 1):
                if 'STAT' in action or 'CRITICAL' in action:
                    st.error(f"{i}. {action}")
                elif 'urgent' in action.lower():
                    st.warning(f"{i}. {action}")
                else:
                    st.info(f"{i}. {action}")
        
        # Follow-up care
        st.markdown("**📋 Follow-up Care:**")
        followup_actions = self._get_followup_actions(diagnosis, confidence)
        
        for action in followup_actions:
            st.write(f"• {action}")
        
        # Patient education points
        st.markdown("**👥 Patient Education Points:**")
        education_points = self._get_patient_education_points(diagnosis)
        
        for point in education_points:
            st.write(f"• {point}")
    
    # Helper methods for functionality
    def _get_scenarios_for_level(self, experience_level: str) -> Dict[str, Any]:
        """Get teaching scenarios appropriate for experience level"""
        level_key = f"{experience_level.lower()}_scenarios"
        return self.teaching_scenarios.get(level_key, {})
    
    def _render_interactive_questions(self, diagnosis: str, scenario: Dict[str, Any]):
        """Render interactive questions for teaching mode"""
        
        st.markdown("**❓ Interactive Questions:**")
        
        # Example questions based on diagnosis
        if 'MI' in diagnosis.upper():
            questions = [
                "What leads would you expect to see ST elevation in anterior MI?",
                "What are the reciprocal changes you would look for?",
                "What is the time-critical management priority?"
            ]
        else:
            questions = [
                "What are the key diagnostic criteria for this condition?",
                "How would you differentiate this from similar conditions?",
                "What are the clinical implications?"
            ]
        
        for i, question in enumerate(questions, 1):
            with st.expander(f"Question {i}: {question}"):
                user_answer = st.text_area(f"Your answer to question {i}:", key=f"q{i}")
                if st.button(f"Get Feedback for Q{i}", key=f"feedback{i}"):
                    feedback = self._generate_question_feedback(question, user_answer, diagnosis)
                    st.info(f"**Feedback:** {feedback}")
    
    def _render_case_based_learning(self, diagnosis: str, confidence: float, experience_level: str):
        """Render case-based learning scenarios"""
        
        if st.button("Generate Similar Case for Practice"):
            similar_case = self._generate_similar_case(diagnosis, experience_level)
            
            st.markdown("**📋 Practice Case:**")
            st.write(f"**Scenario:** {similar_case['scenario']}")
            st.write(f"**Key Features:** {', '.join(similar_case['key_features'])}")
            
            with st.expander("Reveal Diagnosis and Explanation"):
                st.write(f"**Diagnosis:** {similar_case['diagnosis']}")
                st.write(f"**Explanation:** {similar_case['explanation']}")
    
    def _determine_mi_type(self, diagnosis: str) -> str:
        """Determine MI type from diagnosis"""
        if 'AMI' in diagnosis or 'anterior' in diagnosis.lower():
            return 'AMI_Anterior'
        elif 'IMI' in diagnosis or 'inferior' in diagnosis.lower():
            return 'IMI_Inferior'
        elif diagnosis in ['AMI', 'MI_DETECTED']:
            return 'AMI_Anterior'  # Default to anterior
        else:
            return 'NSTEMI'
    
    def _get_confidence_category(self, confidence_percent: float) -> str:
        """Get confidence category from percentage"""
        if confidence_percent >= 85:
            return 'high'
        elif confidence_percent >= 65:
            return 'moderate'
        else:
            return 'low'
    
    def _confidence_in_range(self, confidence_percent: float, range_str: str) -> bool:
        """Check if confidence falls within specified range"""
        if '-' in range_str:
            low, high = range_str.split('-')
            return float(low) <= confidence_percent <= float(high.rstrip('%'))
        elif range_str.startswith('<'):
            threshold = float(range_str[1:])
            return confidence_percent < threshold
        elif range_str.startswith('>'):
            threshold = float(range_str[1:])
            return confidence_percent > threshold
        else:
            return False
    
    def _get_mi_features_for_display(self, diagnosis: str) -> Dict[str, float]:
        """Get MI-specific features for display"""
        return {
            'ST_elevation_V2_V4': 0.35,
            'Q_waves_anterior': 0.25,
            'T_wave_inversion': 0.20,
            'reciprocal_changes_inferior': 0.15,
            'heart_rate_variability': 0.05
        }
    
    def _get_general_features_for_display(self, diagnosis: str) -> Dict[str, float]:
        """Get general features for non-MI diagnoses"""
        return {
            'rhythm_irregularity': 0.30,
            'QRS_morphology': 0.25,
            'rate_characteristics': 0.20,
            'axis_deviation': 0.15,
            'interval_measurements': 0.10
        }
    
    def _create_feature_importance_chart(self, features: Dict[str, float], diagnosis: str):
        """Create feature importance visualization"""
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        feature_names = list(features.keys())
        importance_values = list(features.values())
        
        # Create horizontal bar chart
        bars = ax.barh(feature_names, importance_values, 
                      color=['red' if imp > 0.25 else 'orange' if imp > 0.15 else 'green' 
                             for imp in importance_values])
        
        ax.set_xlabel('Feature Importance')
        ax.set_title(f'Key Features for {diagnosis} Detection')
        ax.set_xlim(0, max(importance_values) * 1.1)
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{width:.1%}', ha='left', va='center')
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    def _get_clinical_info_for_feature(self, feature_name: str, diagnosis: str) -> Optional[Dict[str, Any]]:
        """Get clinical information for a specific feature"""
        
        # Search through feature clinical mapping
        for category, features in self.feature_clinical_mapping.items():
            for feat_key, feat_info in features.items():
                if feature_name.lower() in feat_key.lower() or feat_key.lower() in feature_name.lower():
                    return feat_info
        
        return None
    
    def _identify_uncertainty_factors(self, diagnosis: str, confidence: float) -> List[str]:
        """Identify factors contributing to diagnostic uncertainty"""
        
        factors = []
        
        if confidence < 0.85:
            factors.append("Confidence below high-certainty threshold")
        
        if confidence < 0.65:
            factors.append("Multiple possible diagnoses")
            factors.append("Atypical presentation")
        
        if 'MI' in diagnosis.upper() and confidence < 0.80:
            factors.extend([
                "Borderline ST elevation criteria",
                "Possible old changes vs acute findings",
                "Need for clinical correlation"
            ])
        
        return factors
    
    def _get_uncertainty_recommendations(self, diagnosis: str, confidence: float) -> List[str]:
        """Get recommendations for managing diagnostic uncertainty"""
        
        recommendations = []
        
        if confidence < 0.65:
            recommendations.extend([
                "Obtain serial ECGs for comparison",
                "Correlate with clinical symptoms",
                "Consider alternative diagnoses"
            ])
        
        if 'MI' in diagnosis.upper():
            recommendations.extend([
                "Obtain cardiac biomarkers (troponin)",
                "Compare with prior ECGs if available",
                "Consider cardiology consultation"
            ])
        
        recommendations.append("Document uncertainty in clinical notes")
        
        return recommendations
    
    def _get_learning_objectives(self, diagnosis: str, experience_level: str) -> List[str]:
        """Get learning objectives based on diagnosis and experience level"""
        
        if experience_level == "Beginner":
            return [
                "Identify basic ECG patterns",
                "Recognize normal vs abnormal findings",
                "Understand clinical urgency levels"
            ]
        elif experience_level == "Intermediate":
            return [
                "Apply systematic ECG interpretation",
                "Recognize diagnostic criteria",
                "Correlate ECG with clinical scenarios"
            ]
        else:  # Advanced/Expert
            return [
                "Differentiate subtle diagnostic findings",
                "Manage diagnostic uncertainty",
                "Integrate ECG with complex clinical presentations"
            ]
    
    def _get_related_conditions(self, diagnosis: str) -> List[Dict[str, str]]:
        """Get related conditions to consider"""
        
        if 'MI' in diagnosis.upper():
            return [
                {'name': 'Pericarditis', 'description': 'Can mimic MI with ST elevation'},
                {'name': 'Early repolarization', 'description': 'Benign cause of ST elevation'},
                {'name': 'Old MI with aneurysm', 'description': 'Persistent ST elevation'}
            ]
        elif diagnosis == 'AFIB':
            return [
                {'name': 'Atrial flutter', 'description': 'Regular atrial arrhythmia'},
                {'name': 'Multifocal atrial tachycardia', 'description': 'Irregular rhythm with P waves'},
                {'name': 'Frequent PACs', 'description': 'Can simulate atrial fibrillation'}
            ]
        else:
            return [
                {'name': 'Normal variants', 'description': 'Consider age and population norms'},
                {'name': 'Artifact', 'description': 'Technical issues affecting interpretation'},
                {'name': 'Medication effects', 'description': 'Drug-induced ECG changes'}
            ]
    
    def _get_practice_recommendations(self, diagnosis: str, experience_level: str) -> List[str]:
        """Get practice recommendations for continued learning"""
        
        recommendations = [
            "Practice with multiple examples of this condition",
            "Compare with similar conditions to understand differences",
            "Study the pathophysiology underlying ECG changes"
        ]
        
        if 'MI' in diagnosis.upper():
            recommendations.extend([
                "Practice identifying different MI territories",
                "Learn to recognize subtle vs obvious changes",
                "Understand time evolution of MI patterns"
            ])
        
        return recommendations
    
    def _get_immediate_actions(self, diagnosis: str, confidence: float) -> List[str]:
        """Get immediate clinical actions"""
        
        if 'MI' in diagnosis.upper() or diagnosis in ['AMI', 'IMI']:
            return [
                "STAT cardiology consultation - CRITICAL",
                "Prepare for emergency cardiac catheterization",
                "Administer dual antiplatelet therapy",
                "Continuous cardiac monitoring",
                "Serial ECGs every 15-30 minutes"
            ]
        elif diagnosis == 'AFIB':
            return [
                "Assess hemodynamic stability",
                "Rate control if RVR present",
                "Anticoagulation assessment",
                "Monitor for complications"
            ]
        else:
            return [
                "Clinical correlation with symptoms",
                "Consider serial monitoring",
                "Follow institutional protocols"
            ]
    
    def _get_followup_actions(self, diagnosis: str, confidence: float) -> List[str]:
        """Get follow-up care actions"""
        
        actions = [
            "Document findings and interpretation",
            "Plan appropriate follow-up timing",
            "Patient education regarding condition"
        ]
        
        if 'MI' in diagnosis.upper():
            actions.extend([
                "Cardiac rehabilitation referral",
                "Optimize medical therapy",
                "Risk factor modification counseling",
                "Follow-up echocardiogram"
            ])
        
        return actions
    
    def _get_patient_education_points(self, diagnosis: str) -> List[str]:
        """Get patient education points"""
        
        if 'MI' in diagnosis.upper():
            return [
                "Explain the nature of heart attack",
                "Discuss importance of medication compliance",
                "Review warning signs to watch for",
                "Emphasize lifestyle modifications"
            ]
        else:
            return [
                "Explain the ECG findings in simple terms",
                "Discuss any activity restrictions",
                "Review when to seek medical attention",
                "Address patient concerns and questions"
            ]
    
    def _generate_question_feedback(self, question: str, user_answer: str, diagnosis: str) -> str:
        """Generate feedback for interactive questions"""
        
        # Simple feedback generation
        if not user_answer.strip():
            return "Please provide an answer to receive feedback."
        
        if 'MI' in diagnosis.upper():
            if 'leads' in question.lower():
                return "For anterior MI, look for ST elevation in V1-V4. Good thinking about lead territories!"
            elif 'reciprocal' in question.lower():
                return "Reciprocal changes appear as ST depression in leads opposite to the infarct territory."
            else:
                return "Consider the time-critical nature of STEMI management - door-to-balloon <90 minutes."
        
        return "Good thinking! Consider reviewing the diagnostic criteria and clinical correlations."
    
    def _generate_similar_case(self, diagnosis: str, experience_level: str) -> Dict[str, Any]:
        """Generate a similar case for practice"""
        
        if 'MI' in diagnosis.upper():
            return {
                'diagnosis': 'Inferior STEMI',
                'scenario': '65-year-old with chest pain, ST elevation in II, III, aVF',
                'key_features': ['ST elevation inferior leads', 'Reciprocal changes I, aVL', 'Bradycardia'],
                'explanation': 'Classic inferior MI pattern with RCA occlusion, note the reciprocal changes confirming the diagnosis.'
            }
        else:
            return {
                'diagnosis': diagnosis,
                'scenario': f'Patient presenting with {diagnosis} pattern',
                'key_features': ['Key diagnostic criteria', 'Supporting findings'],
                'explanation': f'This case demonstrates typical {diagnosis} presentation.'
            }

# Global instance
enhanced_explainer = EnhancedECGExplainer()