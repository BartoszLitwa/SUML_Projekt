import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
from datetime import datetime

# Add the Model directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Model'))

from model_inference import OralCancerPredictor

# Page configuration
st.set_page_config(
    page_title="Oral Cancer Risk Prediction",
    page_icon="🦷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        color: #1f77b4;
        border-bottom: 2px solid #1f77b4;
        margin-bottom: 2rem;
    }
    
    .risk-card {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
    }
    
    .risk-low {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    
    .risk-moderate {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
    }
    
    .risk-high {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
    }
    
    .risk-very-high {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
    }
    
    .recommendation-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    
    .stButton > button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

class StreamlitApp:
    def __init__(self):
        self.predictor = OralCancerPredictor()
        self.load_model_status = None
        
    def load_model(self):
        """Load the prediction model"""
        if self.load_model_status is None:
            with st.spinner("Loading prediction model..."):
                self.load_model_status = self.predictor.load_model()
        return self.load_model_status
    
    def render_header(self):
        """Render the application header"""
        st.markdown("""
        <div class="main-header">
            <h1>🦷 Oral Cancer Risk Prediction System</h1>
            <p>Advanced AI-powered assessment for early oral cancer detection</p>
        </div>
        """, unsafe_allow_html=True)
    
    def render_sidebar(self):
        """Render the sidebar with information"""
        st.sidebar.markdown("## About This Application")
        st.sidebar.info("""
        This application uses machine learning to assess oral cancer risk based on various health factors.
        
        **Important Note**: This is a screening tool and should not replace professional medical consultation.
        """)
        
        st.sidebar.markdown("## Key Risk Factors")
        st.sidebar.markdown("""
        - **Tobacco Use**: Primary risk factor
        - **Alcohol Consumption**: Especially when combined with tobacco
        - **HPV Infection**: Human Papillomavirus
        - **Age**: Risk increases with age
        - **Poor Oral Hygiene**: Chronic irritation
        - **Diet**: Low fruit/vegetable intake
        """)
        
        st.sidebar.markdown("## Team Information")
        st.sidebar.markdown("""
        **Student IDs**: s25809, s24339, s24784
        
        **Project**: SUML 2023/2024
        
        **Instructor**: dr Wojciech Oronowicz-Jaśkowiak
        """)
    
    def get_user_input(self):
        """Get user input through the web interface"""
        st.markdown("## Patient Information Input")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Basic Information")
            age = st.number_input("Age", min_value=18, max_value=100, value=45, step=1)
            gender = st.selectbox("Gender", ["Male", "Female"])
            
            st.markdown("### Lifestyle Factors")
            tobacco_use = st.selectbox("Tobacco Use", ["No", "Yes"])
            alcohol_consumption = st.selectbox("Alcohol Consumption", ["No", "Yes"])
            betel_quid_use = st.selectbox("Betel Quid Use", ["No", "Yes"])
            chronic_sun_exposure = st.selectbox("Chronic Sun Exposure", ["No", "Yes"])
            
            st.markdown("### Diet and Hygiene")
            diet_intake = st.selectbox(
                "Diet (Fruits & Vegetables Intake)", 
                ["High", "Moderate", "Low"]
            )
            poor_oral_hygiene = st.selectbox("Poor Oral Hygiene", ["No", "Yes"])
        
        with col2:
            st.markdown("### Medical History")
            hpv_infection = st.selectbox("HPV Infection", ["No", "Yes"])
            family_history = st.selectbox("Family History of Cancer", ["No", "Yes"])
            compromised_immune = st.selectbox("Compromised Immune System", ["No", "Yes"])
            
            st.markdown("### Current Symptoms")
            oral_lesions = st.selectbox("Oral Lesions", ["No", "Yes"])
            unexplained_bleeding = st.selectbox("Unexplained Bleeding", ["No", "Yes"])
            difficulty_swallowing = st.selectbox("Difficulty Swallowing", ["No", "Yes"])
            white_red_patches = st.selectbox(
                "White or Red Patches in Mouth", 
                ["No", "Yes"]
            )
        
        # Compile input data
        input_data = {
            'Age': age,
            'Gender': gender,
            'Tobacco Use': tobacco_use,
            'Alcohol Consumption': alcohol_consumption,
            'HPV Infection': hpv_infection,
            'Betel Quid Use': betel_quid_use,
            'Chronic Sun Exposure': chronic_sun_exposure,
            'Poor Oral Hygiene': poor_oral_hygiene,
            'Diet (Fruits & Vegetables Intake)': diet_intake,
            'Family History of Cancer': family_history,
            'Compromised Immune System': compromised_immune,
            'Oral Lesions': oral_lesions,
            'Unexplained Bleeding': unexplained_bleeding,
            'Difficulty Swallowing': difficulty_swallowing,
            'White or Red Patches in Mouth': white_red_patches
        }
        
        return input_data
    
    def render_risk_result(self, result):
        """Render the risk assessment result"""
        if not result:
            st.error("Unable to calculate risk. Please check your inputs.")
            return
        
        risk_level = result['risk_level']
        risk_percentage = result['risk_percentage']
        
        # Color mapping for risk levels
        risk_colors = {
            'Low': '#28a745',
            'Moderate': '#ffc107', 
            'High': '#fd7e14',
            'Very High': '#dc3545'
        }
        
        # Risk level card
        risk_class = f"risk-{risk_level.lower().replace(' ', '-')}"
        st.markdown(f"""
        <div class="risk-card {risk_class}">
            <h2>Risk Assessment Result</h2>
            <h1>{risk_level} Risk</h1>
            <h3>{risk_percentage:.1f}% Risk Probability</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Risk gauge chart
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = risk_percentage,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Risk Percentage"},
            delta = {'reference': 50},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': risk_colors[risk_level]},
                'steps': [
                    {'range': [0, 20], 'color': "lightgray"},
                    {'range': [20, 50], 'color': "yellow"},
                    {'range': [50, 80], 'color': "orange"},
                    {'range': [80, 100], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    def render_recommendations(self, recommendations):
        """Render personalized recommendations"""
        st.markdown("## Personalized Recommendations")
        
        for i, recommendation in enumerate(recommendations, 1):
            st.markdown(f"""
            <div class="recommendation-box">
                <strong>{i}.</strong> {recommendation}
            </div>
            """, unsafe_allow_html=True)
    
    def render_risk_factors_chart(self, input_data):
        """Render a chart showing risk factors"""
        st.markdown("## Your Risk Factor Profile")
        
        # Identify high-risk factors
        high_risk_factors = []
        if input_data.get('Tobacco Use') == 'Yes':
            high_risk_factors.append('Tobacco Use')
        if input_data.get('Alcohol Consumption') == 'Yes':
            high_risk_factors.append('Alcohol Consumption')
        if input_data.get('HPV Infection') == 'Yes':
            high_risk_factors.append('HPV Infection')
        if input_data.get('Poor Oral Hygiene') == 'Yes':
            high_risk_factors.append('Poor Oral Hygiene')
        if input_data.get('Diet (Fruits & Vegetables Intake)') == 'Low':
            high_risk_factors.append('Low Diet Quality')
        if input_data.get('Family History of Cancer') == 'Yes':
            high_risk_factors.append('Family History')
        if input_data.get('Oral Lesions') == 'Yes':
            high_risk_factors.append('Oral Lesions')
        if input_data.get('Unexplained Bleeding') == 'Yes':
            high_risk_factors.append('Unexplained Bleeding')
        
        if high_risk_factors:
            # Create a bar chart of risk factors
            risk_factor_data = pd.DataFrame({
                'Risk Factor': high_risk_factors,
                'Present': [1] * len(high_risk_factors)
            })
            
            fig = px.bar(
                risk_factor_data, 
                y='Risk Factor', 
                x='Present',
                orientation='h',
                title="Risk Factors Present",
                color_discrete_sequence=['#ff7f0e']
            )
            fig.update_layout(
                xaxis_title="",
                yaxis_title="Risk Factors",
                showlegend=False,
                height=max(300, len(high_risk_factors) * 50)
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("Great! You have minimal identifiable risk factors.")
    
    def render_history_section(self):
        """Render section for tracking assessment history"""
        st.markdown("## Assessment History")
        
        if 'assessment_history' not in st.session_state:
            st.session_state.assessment_history = []
        
        if st.session_state.assessment_history:
            history_df = pd.DataFrame(st.session_state.assessment_history)
            
            # Plot risk over time
            fig = px.line(
                history_df, 
                x='timestamp', 
                y='risk_percentage',
                title="Risk Percentage Over Time",
                markers=True
            )
            fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Risk Percentage (%)"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Show history table
            st.dataframe(history_df[['timestamp', 'risk_level', 'risk_percentage']])
        else:
            st.info("No assessment history available. Complete an assessment to start tracking.")
    
    def save_assessment(self, input_data, result):
        """Save assessment to session history"""
        if 'assessment_history' not in st.session_state:
            st.session_state.assessment_history = []
        
        assessment_record = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'risk_level': result['risk_level'],
            'risk_percentage': result['risk_percentage'],
            'age': input_data['Age'],
            'gender': input_data['Gender']
        }
        
        st.session_state.assessment_history.append(assessment_record)
    
    def run(self):
        """Main application runner"""
        self.render_header()
        self.render_sidebar()
        
        # Check if model is loaded
        if not self.load_model():
            st.error("""
            ⚠️ **Model Loading Error**
            
            The prediction model could not be loaded. This might be because:
            1. The model hasn't been trained yet
            2. Model files are missing from the artifacts directory
            
            Please run the model training pipeline first:
            ```
            python Model/model_training.py
            ```
            """)
            return
        
        # Main application tabs
        tab1, tab2, tab3 = st.tabs(["🔍 Risk Assessment", "📊 Risk Analysis", "📋 History"])
        
        with tab1:
            # Get user input
            input_data = self.get_user_input()
            
            # Prediction button
            if st.button("🔬 Assess Risk", type="primary"):
                with st.spinner("Analyzing risk factors..."):
                    # Validate input
                    is_valid, message = self.predictor.validate_input(input_data)
                    
                    if is_valid:
                        # Make prediction
                        result = self.predictor.predict_risk(input_data)
                        
                        if result:
                            # Save assessment
                            self.save_assessment(input_data, result)
                            
                            # Display results
                            self.render_risk_result(result)
                            self.render_recommendations(result['recommendations'])
                        else:
                            st.error("Failed to make prediction. Please try again.")
                    else:
                        st.error(f"Input validation failed: {message}")
        
        with tab2:
            st.markdown("## Risk Factor Analysis")
            
            if st.button("Analyze Current Input"):
                input_data = self.get_user_input()
                self.render_risk_factors_chart(input_data)
        
        with tab3:
            self.render_history_section()
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #666; padding: 1rem;">
            <p><strong>Disclaimer:</strong> This tool is for educational and screening purposes only. 
            Always consult with healthcare professionals for medical advice.</p>
            <p>© 2024 SUML Project - Oral Cancer Risk Prediction System</p>
        </div>
        """, unsafe_allow_html=True)


def main():
    """Main function to run the Streamlit app"""
    app = StreamlitApp()
    app.run()


if __name__ == "__main__":
    main() 