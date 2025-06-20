import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys
import os
from datetime import datetime

# Add the Model directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Model'))

from model_inference import OralCancerPredictor

# Page configuration
st.set_page_config(
    page_title="Oral Cancer Risk Prediction",
    page_icon="🦷"
)

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
        st.title("🦷 Oral Cancer Risk Prediction System")
        st.write("AI-powered assessment for early oral cancer detection")
    
    def render_sidebar(self):
        """Render the sidebar with information"""
        st.sidebar.title("About")
        st.sidebar.info("This app assesses oral cancer risk using machine learning. Not a replacement for medical consultation.")
        
        st.sidebar.subheader("Key Risk Factors")
        st.sidebar.write("• Tobacco Use")
        st.sidebar.write("• Alcohol Consumption") 
        st.sidebar.write("• HPV Infection")
        st.sidebar.write("• Age")
        st.sidebar.write("• Poor Oral Hygiene")
        
        st.sidebar.subheader("Team")
        st.sidebar.write("Student IDs: s25809, s24339, s24784")
        st.sidebar.write("SUML 2023/2024")
    
    def get_user_input(self):
        """Get user input through the web interface"""
        st.subheader("Patient Information")
        
        # Basic Information
        st.write("**Basic Information**")
        age = st.number_input("Age", min_value=18, max_value=100, value=45, step=1)
        gender = st.selectbox("Gender", ["Male", "Female"])
        
        # Lifestyle Factors
        st.write("**Lifestyle Factors**")
        tobacco_use = st.selectbox("Tobacco Use", ["No", "Yes"])
        alcohol_consumption = st.selectbox("Alcohol Consumption", ["No", "Yes"])
        betel_quid_use = st.selectbox("Betel Quid Use", ["No", "Yes"])
        chronic_sun_exposure = st.selectbox("Chronic Sun Exposure", ["No", "Yes"])
        
        # Diet and Hygiene
        diet_intake = st.selectbox("Diet (Fruits & Vegetables Intake)", ["High", "Moderate", "Low"])
        poor_oral_hygiene = st.selectbox("Poor Oral Hygiene", ["No", "Yes"])
        
        # Medical History
        st.write("**Medical History**")
        hpv_infection = st.selectbox("HPV Infection", ["No", "Yes"])
        family_history = st.selectbox("Family History of Cancer", ["No", "Yes"])
        compromised_immune = st.selectbox("Compromised Immune System", ["No", "Yes"])
        
        # Current Symptoms
        st.write("**Current Symptoms**")
        oral_lesions = st.selectbox("Oral Lesions", ["No", "Yes"])
        unexplained_bleeding = st.selectbox("Unexplained Bleeding", ["No", "Yes"])
        difficulty_swallowing = st.selectbox("Difficulty Swallowing", ["No", "Yes"])
        white_red_patches = st.selectbox("White or Red Patches in Mouth", ["No", "Yes"])
        
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
        
        # Display results simply
        st.success("Risk Assessment Complete!")
        st.subheader(f"Risk Level: {risk_level}")
        st.write(f"Risk Percentage: {risk_percentage:.1f}%")
        
        # Simple progress bar
        st.progress(risk_percentage / 100)
        
        # Color-coded message
        if risk_level == "Low":
            st.success("Low risk detected")
        elif risk_level == "Moderate":
            st.warning("Moderate risk detected")
        elif risk_level == "High":
            st.error("High risk detected")
        else:
            st.error("Very high risk detected")
    
    def render_recommendations(self, recommendations):
        """Render personalized recommendations"""
        st.subheader("Recommendations")
        
        for i, recommendation in enumerate(recommendations, 1):
            st.write(f"{i}. {recommendation}")
    
    def render_risk_factors_chart(self, input_data):
        """Display risk factors"""
        st.subheader("Your Risk Factors")
        
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
            st.write("Risk factors present:")
            for factor in high_risk_factors:
                st.write(f"• {factor}")
        else:
            st.success("No major risk factors identified!")
    
    def render_history_section(self):
        """Show assessment history"""
        st.subheader("Assessment History")
        
        if 'assessment_history' not in st.session_state:
            st.session_state.assessment_history = []
        
        if st.session_state.assessment_history:
            history_df = pd.DataFrame(st.session_state.assessment_history)
            st.dataframe(history_df[['timestamp', 'risk_level', 'risk_percentage']])
        else:
            st.info("No assessment history available.")
    
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
            st.error("Model could not be loaded. Please train the model first by running: python Model/model_training.py")
            return
        
        # Main application
        # Get user input
        input_data = self.get_user_input()
        
        # Prediction button
        if st.button("Assess Risk"):
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
                    self.render_risk_factors_chart(input_data)
                else:
                    st.error("Failed to make prediction. Please try again.")
            else:
                st.error(f"Input validation failed: {message}")
        
        # Show history
        st.divider()
        self.render_history_section()
        
        # Footer
        st.divider()
        st.caption("Disclaimer: This tool is for educational purposes only. Always consult healthcare professionals for medical advice.")
        st.caption("© 2024 SUML Project - Student IDs: s25809, s24339, s24784")


def main():
    """Main function to run the Streamlit app"""
    app = StreamlitApp()
    app.run()


if __name__ == "__main__":
    main() 