import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from datetime import datetime, timedelta

class AccidentDashboard:
    def __init__(self):
        self.plots_dir = 'plots'
        os.makedirs(self.plots_dir, exist_ok=True)

    def create_sidebar(self):
        """Create sidebar with filters and controls"""
        st.sidebar.title("Filters")
        
        # Date range filter
        min_date = datetime(2020, 1, 1)
        max_date = datetime(2023, 12, 31)
        start_date = st.sidebar.date_input(
            "Start Date",
            min_date,
            min_value=min_date,
            max_value=max_date
        )
        end_date = st.sidebar.date_input(
            "End Date",
            max_date,
            min_value=min_date,
            max_value=max_date
        )
        
        # Location filter
        locations = ['All', 'Downtown', 'Suburbs', 'Highway']
        selected_location = st.sidebar.selectbox("Location", locations)
        
        # Weather condition filter
        weather_conditions = ['All', 'Clear', 'Rainy', 'Snowy', 'Foggy']
        selected_weather = st.sidebar.selectbox("Weather Condition", weather_conditions)
        
        return {
            'start_date': start_date,
            'end_date': end_date,
            'location': selected_location,
            'weather': selected_weather
        }

    def display_summary_metrics(self, df):
        """Display key metrics and statistics"""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Accidents", len(df))
        
        with col2:
            avg_severity = df['severity'].mean()
            st.metric("Average Severity", f"{avg_severity:.2f}")
        
        with col3:
            total_injuries = df['injuries'].sum()
            st.metric("Total Injuries", total_injuries)
        
        with col4:
            fatality_rate = (df['fatalities'].sum() / len(df)) * 100
            st.metric("Fatality Rate", f"{fatality_rate:.2f}%")

    def plot_accident_trends(self, df):
        """Plot accident trends over time"""
        st.subheader("Accident Trends Over Time")
        
        # Group by date and count accidents
        daily_accidents = df.groupby('date').size().reset_index(name='count')
        
        fig = px.line(
            daily_accidents,
            x='date',
            y='count',
            title='Daily Accident Count'
        )
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Number of Accidents",
            hovermode='x unified'
        )
        st.plotly_chart(fig)

    def plot_severity_distribution(self, df):
        """Plot severity distribution"""
        st.subheader("Accident Severity Distribution")
        
        severity_counts = df['severity'].value_counts().reset_index()
        severity_counts.columns = ['severity', 'count']
        
        fig = px.bar(
            severity_counts,
            x='severity',
            y='count',
            title='Accident Severity Distribution'
        )
        fig.update_layout(
            xaxis_title="Severity Level",
            yaxis_title="Number of Accidents"
        )
        st.plotly_chart(fig)

    def plot_location_heatmap(self, df):
        """Plot accident location heatmap"""
        st.subheader("Accident Location Heatmap")
        
        fig = px.density_mapbox(
            df,
            lat='latitude',
            lon='longitude',
            z='severity',
            radius=10,
            center=dict(lat=df['latitude'].mean(), lon=df['longitude'].mean()),
            zoom=10,
            mapbox_style="stamen-terrain"
        )
        st.plotly_chart(fig)

    def plot_weather_impact(self, df):
        """Plot weather impact on accidents"""
        st.subheader("Weather Impact on Accidents")
        
        weather_severity = df.groupby('weather_condition')['severity'].mean().reset_index()
        
        fig = px.bar(
            weather_severity,
            x='weather_condition',
            y='severity',
            title='Average Accident Severity by Weather Condition'
        )
        fig.update_layout(
            xaxis_title="Weather Condition",
            yaxis_title="Average Severity"
        )
        st.plotly_chart(fig)

    def plot_time_of_day_analysis(self, df):
        """Plot accident distribution by time of day"""
        st.subheader("Accident Distribution by Time of Day")
        
        df['hour'] = pd.to_datetime(df['time']).dt.hour
        hourly_accidents = df.groupby('hour').size().reset_index(name='count')
        
        fig = px.line(
            hourly_accidents,
            x='hour',
            y='count',
            title='Accidents by Hour of Day'
        )
        fig.update_layout(
            xaxis_title="Hour of Day",
            yaxis_title="Number of Accidents"
        )
        st.plotly_chart(fig)

    def display_prediction_insights(self, predictions):
        """Display model predictions and insights"""
        st.subheader("Accident Prediction Insights")
        
        # Display prediction metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Predicted Accidents (Next 24h)", predictions['next_24h']['average'])
        
        with col2:
            st.metric("Risk Level", predictions['risk_level'])
        
        # Display risk factors
        st.subheader("Key Risk Factors")
        risk_factors = predictions['risk_factors']
        for factor, impact in risk_factors.items():
            st.progress(impact, text=f"{factor}: {impact*100:.1f}% impact")

    def run_dashboard(self, df, predictions=None):
        """Run the complete dashboard"""
        st.title("Road Accident Analysis Dashboard")
        
        # Create sidebar and get filters
        filters = self.create_sidebar()
        
        # Apply filters to data
        filtered_df = df.copy()
        if filters['location'] != 'All':
            filtered_df = filtered_df[filtered_df['location'] == filters['location']]
        if filters['weather'] != 'All':
            filtered_df = filtered_df[filtered_df['weather_condition'] == filters['weather']]
        filtered_df = filtered_df[
            (filtered_df['date'] >= filters['start_date']) &
            (filtered_df['date'] <= filters['end_date'])
        ]
        
        # Display dashboard components
        self.display_summary_metrics(filtered_df)
        self.plot_accident_trends(filtered_df)
        
        col1, col2 = st.columns(2)
        with col1:
            self.plot_severity_distribution(filtered_df)
        with col2:
            self.plot_weather_impact(filtered_df)
        
        self.plot_location_heatmap(filtered_df)
        self.plot_time_of_day_analysis(filtered_df)
        
        if predictions:
            self.display_prediction_insights(predictions) 