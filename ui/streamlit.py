import pandas as pd
import streamlit as st
import requests
import plotly.express as px
from datetime import datetime
import yaml
import psycopg2

# db_management = DatabaseManagement("postgresql+psycopg2://postgres:postgres@127.0.0.1:5432/postgres")

def get_active_models():
    conn = psycopg2.connect(
    host="127.0.0.1",
    database="postgres",
    user="postgres",
    password="postgres"
)

    cur = conn.cursor()
    df = pd.read_sql_query("""
        SELECT DISTINCT model_name, host, port
        FROM events
        WHERE id IN (
            SELECT MAX(id)
            FROM events
            GROUP BY model_name, host, port
        )
        AND status = 'active'
        AND event_type != 'shutdown'
    """, conn)
    conn.close()
    return df.to_dict('records')

st.set_page_config(layout="wide")

def get_prediction(model_info, query_text):
    host = model_info['host']
    port = model_info['port']
    api_url = f"http://{host}:{port}/predict"
    response = requests.post(api_url, json={"text": query_text})
    if response.status_code == 200:
        predictions = response.json()
        return predictions
    else:
        return None
    
def get_explanation(model_info, query_text):
    host = model_info['host']
    port = model_info['port']
    api_url = f"http://{host}:{port}/explain"
    response = requests.post(api_url, json={"text": query_text})
    if response.status_code == 200:
        predictions = response.json()
        return predictions
    else:
        return None

def submit_feedback(query_text, original_predictions, run_info, feedback_labels):
    host = run_info['host']
    port = run_info['port']
    api_url = f"http://{host}:{port}/submit_feedback"
    feedback_data = {
        "query_text": query_text,
        "original_predictions": str(original_predictions),
        "run_id": str(run_info['run_id']),
        "host": str(run_info['host']),
        "port": str(run_info['port']),
        "feedback_labels": str(feedback_labels),
        "timestamp": str(datetime.now().isoformat())
    }

    response = requests.post(api_url, json=feedback_data)
    if response.status_code == 200:
        return True
    else:
        return False

active_models = get_active_models()
model_options = [f"{model['model_name']} ({model['host']}:{model['port']})" for model in active_models]
# model_options = [f"{model['run_id']} ({model['host']}:{model['port']})" for model in active_models]
chosen_model_option = st.selectbox("Choose a deployed model", model_options)
chosen_model_info = active_models[model_options.index(chosen_model_option)]
query_text = st.text_input("Enter a sentence", "I love this movie!")

col1, col2, col3 = st.columns(3)

if 'predictions' not in st.session_state:
    st.session_state.predictions = None

if 'explanation' not in st.session_state:
    st.session_state.explanation = None

with col1:
    st.header("Prediction")
    st.session_state.predictions = get_prediction(chosen_model_info, query_text)
    if st.session_state.predictions:
        # Create a DataFrame for predictions
        predictions_df = pd.DataFrame(st.session_state.predictions)
        # Create a bar chart for predictions using Plotly
        fig = px.bar(
            predictions_df, 
            x='score', 
            y='label', 
            orientation='h', 
            text='score')
        fig.update_traces(texttemplate='%{text:.2f}', textposition='inside')
        fig.update_layout(
            uniformtext_minsize=8, 
            uniformtext_mode='hide',
            height=500,
            width=400)
        st.plotly_chart(fig)

with col2:
    st.header("Explanation")
    st.session_state.explanation = get_explanation(chosen_model_info, query_text)
    if st.session_state.explanation:
        # Create a DataFrame for explanations
        explanation_df = pd.DataFrame(
            st.session_state.explanation['explanation'], 
            columns=['word', 'importance'])
        
        # Create a new column for colors based on the sign of the importance values
        explanation_df['color'] = explanation_df['importance'].apply(lambda x: 'blue' if x > 0 else 'red')
        
        # Create a bar chart for explanations using Plotly, resembling LIME visualization
        fig = px.bar(
            explanation_df, 
            x='importance', 
            y='word', 
            orientation='h', 
            text='importance',
            color='color', 
            color_discrete_map={'red': 'red', 'blue': 'blue'})
        fig.update_traces(texttemplate='%{text:.2f}', textposition='inside')
        fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide', coloraxis_showscale=False,showlegend=False,
                          height=500, width=400)
        st.plotly_chart(fig)

if st.session_state.predictions:
    labels = [prediction["label"] for prediction in st.session_state.predictions]
else:
    labels = []

# Imports and other code remain unchanged

with col3:
    st.header("Feedback")
    if labels:
        with st.form(key="feedback_form"):
            selected_feedback_labels = st.multiselect("Select the correct labels", labels)
            
            if st.form_submit_button("Submit Feedback"):
                if st.session_state.predictions:
                    success = submit_feedback(
                        query_text, 
                        st.session_state.predictions, 
                        chosen_model_info, 
                        selected_feedback_labels)
                    if success:
                        st.success("Feedback submitted successfully!")
                    else:
                        st.error("Failed to submit feedback.")
                else:
                    st.error("Please get the prediction before submitting feedback.")
    else:
        st.warning("Please get the prediction before providing feedback.")

