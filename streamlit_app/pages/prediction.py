"""
Prediction page for Streamlit app
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from streamlit_app.utils.model_loader import load_saved_model, get_model_info
from streamlit_app.utils.preprocessing import load_preprocessor, create_input_form, preprocess_input, display_prediction_result

def show_prediction_page():
    """Display prediction page"""
    
    st.title("🔮 Loan Approval Prediction")
    st.write("Enter the loan application details below to get an instant approval prediction.")
    
    # Load model and preprocessor
    model, metadata = load_saved_model()
    preprocessor = load_preprocessor()
    
    if model is None or preprocessor is None:
        st.error("Required models not found. Please run the training pipeline first.")
        return
    
    # Display model information
    with st.expander("ℹ️ Model Information"):
        st.markdown(get_model_info(metadata))
    
    # Create input form
    input_data = create_input_form()
    
    if input_data is not None:
        try:
            # Preprocess input
            processed_data = preprocess_input(input_data, preprocessor)
            
            # Make prediction
            prediction = model.predict(processed_data)[0]
            probability = model.predict_proba(processed_data)[0][1]
            
            # Display results
            st.write("---")
            st.subheader("📊 Prediction Result")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                display_prediction_result(prediction, probability)
            
            with col2:
                # Probability gauge
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = probability * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Approval Probability (%)"},
                    delta = {'reference': 50},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 25], 'color': "lightgray"},
                            {'range': [25, 50], 'color': "gray"},
                            {'range': [50, 75], 'color': "lightgreen"},
                            {'range': [75, 100], 'color': "green"}
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
            
            # Feature Analysis
            st.write("### 📈 Input Analysis")
            
            # Display input summary
            input_df = pd.DataFrame([input_data])
            st.dataframe(input_df, use_container_width=True)
            
            # Risk factors analysis
            st.write("### ⚠️ Risk Factors Analysis")
            
            risk_factors = []
            positive_factors = []
            
            # Analyze key factors
            if input_data['Credit_History'] == 0.0:
                risk_factors.append("❌ No credit history")
            else:
                positive_factors.append("✅ Good credit history")
            
            if input_data['ApplicantIncome'] < 3000:
                risk_factors.append("❌ Low applicant income")
            elif input_data['ApplicantIncome'] > 10000:
                positive_factors.append("✅ High applicant income")
            
            if input_data['LoanAmount'] > 500:
                risk_factors.append("❌ High loan amount")
            elif input_data['LoanAmount'] < 200:
                positive_factors.append("✅ Moderate loan amount")
            
            if input_data['Self_Employed'] == 'Yes':
                risk_factors.append("⚠️ Self-employed status")
            
            # Display factors
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Positive Factors:**")
                for factor in positive_factors:
                    st.write(factor)
            
            with col2:
                st.write("**Risk Factors:**")
                for factor in risk_factors:
                    st.write(factor)
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")

if __name__ == "__main__":
    show_prediction_page()

