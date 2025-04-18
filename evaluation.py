import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

class ModelEvaluator:
    def __init__(self):
        self.metrics = {}
        self.plots_dir = 'plots'
        os.makedirs(self.plots_dir, exist_ok=True)

    def calculate_metrics(self, y_true, y_pred, model_name):
        """Calculate and store evaluation metrics"""
        metrics = {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred)
        }
        self.metrics[model_name] = metrics
        return metrics

    def plot_predictions(self, y_true, y_pred, model_name, title):
        """Plot actual vs predicted values"""
        plt.figure(figsize=(10, 6))
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(title)
        plt.savefig(os.path.join(self.plots_dir, f'{model_name}_predictions.png'))
        plt.close()

    def plot_residuals(self, y_true, y_pred, model_name):
        """Plot residuals distribution"""
        residuals = y_true - y_pred
        plt.figure(figsize=(10, 6))
        sns.histplot(residuals, kde=True)
        plt.xlabel('Residuals')
        plt.title(f'Residuals Distribution - {model_name}')
        plt.savefig(os.path.join(self.plots_dir, f'{model_name}_residuals.png'))
        plt.close()

    def plot_feature_importance(self, model, feature_names, model_name):
        """Plot feature importance for tree-based models"""
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            plt.figure(figsize=(10, 6))
            sns.barplot(x=importance, y=feature_names)
            plt.title(f'Feature Importance - {model_name}')
            plt.savefig(os.path.join(self.plots_dir, f'{model_name}_feature_importance.png'))
            plt.close()

class TimeSeriesVisualizer:
    def __init__(self):
        self.plots_dir = 'plots'
        os.makedirs(self.plots_dir, exist_ok=True)

    def plot_time_series(self, df, date_column, value_column, title):
        """Create interactive time series plot"""
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df[date_column],
            y=df[value_column],
            mode='lines',
            name=value_column
        ))
        fig.update_layout(
            title=title,
            xaxis_title='Date',
            yaxis_title='Value',
            hovermode='x unified'
        )
        fig.write_html(os.path.join(self.plots_dir, f'{title.lower().replace(" ", "_")}.html'))

    def plot_forecast(self, forecast, actual=None, title='Forecast'):
        """Plot forecast with confidence intervals"""
        fig = go.Figure()
        
        # Plot actual data if provided
        if actual is not None:
            fig.add_trace(go.Scatter(
                x=actual['ds'],
                y=actual['y'],
                mode='lines',
                name='Actual'
            ))
        
        # Plot forecast
        fig.add_trace(go.Scatter(
            x=forecast['ds'],
            y=forecast['yhat'],
            mode='lines',
            name='Forecast'
        ))
        
        # Add confidence intervals
        fig.add_trace(go.Scatter(
            x=forecast['ds'],
            y=forecast['yhat_upper'],
            fill=None,
            mode='lines',
            line_color='rgba(0,100,80,0.2)',
            name='Upper Bound'
        ))
        fig.add_trace(go.Scatter(
            x=forecast['ds'],
            y=forecast['yhat_lower'],
            fill='tonexty',
            mode='lines',
            line_color='rgba(0,100,80,0.2)',
            name='Lower Bound'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Date',
            yaxis_title='Value',
            hovermode='x unified'
        )
        fig.write_html(os.path.join(self.plots_dir, f'{title.lower().replace(" ", "_")}_forecast.html'))

    def plot_seasonality(self, forecast, title='Seasonality Components'):
        """Plot seasonality components"""
        fig = make_subplots(rows=2, cols=1)
        
        # Weekly seasonality
        fig.add_trace(
            go.Scatter(x=forecast['ds'], y=forecast['weekly'], name='Weekly'),
            row=1, col=1
        )
        
        # Yearly seasonality
        fig.add_trace(
            go.Scatter(x=forecast['ds'], y=forecast['yearly'], name='Yearly'),
            row=2, col=1
        )
        
        fig.update_layout(
            title=title,
            height=800,
            showlegend=True
        )
        fig.write_html(os.path.join(self.plots_dir, 'seasonality_components.html')) 