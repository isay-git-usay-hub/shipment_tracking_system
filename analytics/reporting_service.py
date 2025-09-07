"""
Advanced Reporting Service for Maersk Shipment AI System

This module provides automated report generation, visualization creation,
and export capabilities for business intelligence and analytics.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import logging
import json
import base64
from io import BytesIO
from pathlib import Path

# Visualization imports
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio

# Data processing
from analytics_engine import get_analytics_engine, AnalysisType, TimeGranularity

logger = logging.getLogger(__name__)


class ReportType:
    EXECUTIVE_SUMMARY = "executive_summary"
    OPERATIONAL_DASHBOARD = "operational_dashboard" 
    ML_PERFORMANCE = "ml_performance"
    ROOT_CAUSE_ANALYSIS = "root_cause_analysis"
    PREDICTIVE_FORECAST = "predictive_forecast"
    KPI_SCORECARD = "kpi_scorecard"
    CUSTOM = "custom"


class ReportingService:
    """
    Advanced reporting service for comprehensive analytics and visualizations
    """
    
    def __init__(self):
        self.analytics_engine = get_analytics_engine()
        self.report_templates = {}
        self.visualization_cache = {}
        
        # Configure Plotly
        pio.templates.default = "plotly_white"
        
        # Setup report templates
        self._setup_report_templates()
    
    def _setup_report_templates(self):
        """Setup predefined report templates"""
        self.report_templates = {
            ReportType.EXECUTIVE_SUMMARY: {
                'title': 'Executive Summary Report',
                'sections': ['overview', 'kpi_performance', 'key_insights', 'recommendations'],
                'visualizations': ['kpi_dashboard', 'trend_charts', 'status_distribution']
            },
            ReportType.OPERATIONAL_DASHBOARD: {
                'title': 'Operational Dashboard Report',
                'sections': ['real_time_metrics', 'capacity_analysis', 'delay_analysis', 'asset_performance'],
                'visualizations': ['metric_cards', 'time_series', 'heatmaps', 'geographic_map']
            },
            ReportType.ML_PERFORMANCE: {
                'title': 'ML Model Performance Report',
                'sections': ['model_accuracy', 'prediction_analysis', 'feature_importance', 'model_comparison'],
                'visualizations': ['accuracy_trends', 'confusion_matrix', 'feature_plots', 'roc_curves']
            },
            ReportType.ROOT_CAUSE_ANALYSIS: {
                'title': 'Root Cause Analysis Report',
                'sections': ['problem_identification', 'causal_factors', 'impact_analysis', 'solutions'],
                'visualizations': ['pareto_charts', 'correlation_matrix', 'factor_analysis', 'timeline']
            },
            ReportType.PREDICTIVE_FORECAST: {
                'title': 'Predictive Forecast Report',
                'sections': ['forecast_overview', 'trend_analysis', 'risk_assessment', 'scenarios'],
                'visualizations': ['forecast_charts', 'confidence_intervals', 'scenario_comparison', 'risk_matrix']
            },
            ReportType.KPI_SCORECARD: {
                'title': 'KPI Performance Scorecard',
                'sections': ['kpi_summary', 'performance_trends', 'benchmark_comparison', 'action_items'],
                'visualizations': ['scorecard_table', 'gauge_charts', 'trend_sparklines', 'target_vs_actual']
            }
        }
    
    def generate_executive_summary_report(self, shipments_df: pd.DataFrame, period_days: int = 30) -> Dict[str, Any]:
        """Generate executive summary report"""
        try:
            report_data = {
                'report_type': ReportType.EXECUTIVE_SUMMARY,
                'generated_at': datetime.now().isoformat(),
                'period_days': period_days,
                'data_summary': {
                    'total_records': len(shipments_df),
                    'date_range': self._get_date_range(shipments_df)
                }
            }
            
            # Key metrics overview
            overview_metrics = self._calculate_overview_metrics(shipments_df)
            report_data['overview'] = overview_metrics
            
            # KPI performance
            kpi_performance = self.analytics_engine._evaluate_kpis(shipments_df)
            report_data['kpi_performance'] = kpi_performance
            
            # Trend analysis
            trend_analysis = self.analytics_engine.analyze_shipment_trends(shipments_df, TimeGranularity.DAILY)
            report_data['trends'] = trend_analysis.get('summary', {})
            
            # Key insights
            comprehensive_report = self.analytics_engine.generate_comprehensive_report(shipments_df)
            report_data['insights'] = comprehensive_report.insights[:5]  # Top 5 insights
            report_data['recommendations'] = comprehensive_report.recommendations[:3]  # Top 3 recommendations
            
            # Generate visualizations
            visualizations = self._create_executive_visualizations(shipments_df, overview_metrics)
            report_data['visualizations'] = visualizations
            
            # Convert numpy types to Python native types for JSON serialization
            report_data = self._sanitize_for_json(report_data)
            
            return {
                'success': True,
                'report': report_data
            }
            
        except Exception as e:
            logger.error(f"Error generating executive summary: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def generate_operational_dashboard_report(self, shipments_df: pd.DataFrame) -> Dict[str, Any]:
        """Generate operational dashboard report"""
        try:
            report_data = {
                'report_type': ReportType.OPERATIONAL_DASHBOARD,
                'generated_at': datetime.now().isoformat(),
                'data_summary': {
                    'total_records': len(shipments_df),
                    'date_range': self._get_date_range(shipments_df)
                }
            }
            
            # Real-time metrics
            real_time_metrics = self._calculate_realtime_metrics(shipments_df)
            report_data['real_time_metrics'] = real_time_metrics
            
            # Capacity analysis
            daily_data = self._prepare_daily_data(shipments_df)
            capacity_analysis = self.analytics_engine._analyze_capacity_trends(daily_data)
            report_data['capacity_analysis'] = capacity_analysis
            
            # Delay analysis
            root_cause = self.analytics_engine.perform_root_cause_analysis(shipments_df)
            report_data['delay_analysis'] = {
                'summary': root_cause.get('summary', {}),
                'top_reasons': dict(list(root_cause.get('delay_reasons', {}).get('percentages', {}).items())[:5])
            }
            
            # Asset performance
            asset_performance = self._analyze_asset_performance(shipments_df)
            report_data['asset_performance'] = asset_performance
            
            # Generate visualizations
            visualizations = self._create_operational_visualizations(shipments_df, real_time_metrics)
            report_data['visualizations'] = visualizations
            
            # Convert numpy types to Python native types for JSON serialization
            report_data = self._sanitize_for_json(report_data)
            
            return {
                'success': True,
                'report': report_data
            }
            
        except Exception as e:
            logger.error(f"Error generating operational dashboard: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def generate_predictive_forecast_report(self, shipments_df: pd.DataFrame, forecast_days: int = 14) -> Dict[str, Any]:
        """Generate predictive forecast report"""
        try:
            report_data = {
                'report_type': ReportType.PREDICTIVE_FORECAST,
                'generated_at': datetime.now().isoformat(),
                'forecast_days': forecast_days,
                'data_summary': {
                    'total_records': len(shipments_df),
                    'date_range': self._get_date_range(shipments_df)
                }
            }
            
            # Generate predictive insights
            predictive_results = self.analytics_engine.generate_predictive_insights(shipments_df)
            report_data['forecast'] = predictive_results.get('forecast', {})
            report_data['risk_assessment'] = predictive_results.get('risk_assessment', {})
            report_data['capacity_analysis'] = predictive_results.get('capacity_analysis', {})
            
            # Trend analysis for forecasting
            trend_analysis = self.analytics_engine.analyze_shipment_trends(shipments_df, TimeGranularity.DAILY)
            report_data['historical_trends'] = trend_analysis.get('summary', {})
            
            # Scenario analysis
            scenario_analysis = self._generate_scenario_analysis(shipments_df)
            report_data['scenarios'] = scenario_analysis
            
            # Generate visualizations
            visualizations = self._create_predictive_visualizations(shipments_df, predictive_results)
            report_data['visualizations'] = visualizations
            
            # Convert numpy types to Python native types for JSON serialization
            report_data = self._sanitize_for_json(report_data)
            
            return {
                'success': True,
                'report': report_data
            }
            
        except Exception as e:
            logger.error(f"Error generating predictive forecast: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def generate_kpi_scorecard_report(self, shipments_df: pd.DataFrame) -> Dict[str, Any]:
        """Generate KPI performance scorecard"""
        try:
            report_data = {
                'report_type': ReportType.KPI_SCORECARD,
                'generated_at': datetime.now().isoformat(),
                'data_summary': {
                    'total_records': len(shipments_df),
                    'date_range': self._get_date_range(shipments_df)
                }
            }
            
            # KPI evaluation
            kpi_results = self.analytics_engine._evaluate_kpis(shipments_df)
            report_data['kpi_summary'] = kpi_results
            
            # Performance trends
            performance_trends = self._calculate_kpi_trends(shipments_df)
            report_data['performance_trends'] = performance_trends
            
            # Benchmark comparison
            benchmark_comparison = self._generate_benchmark_comparison(kpi_results)
            report_data['benchmark_comparison'] = benchmark_comparison
            
            # Generate visualizations
            visualizations = self._create_kpi_visualizations(kpi_results, performance_trends)
            report_data['visualizations'] = visualizations
            
            # Convert numpy types to Python native types for JSON serialization
            report_data = self._sanitize_for_json(report_data)
            
            return {
                'success': True,
                'report': report_data
            }
            
        except Exception as e:
            logger.error(f"Error generating KPI scorecard: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _calculate_overview_metrics(self, shipments_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate key overview metrics"""
        if shipments_df.empty:
            return {}
        
        total_shipments = len(shipments_df)
        delayed_count = len(shipments_df[shipments_df['logistics_delay'] == True]) if 'logistics_delay' in shipments_df.columns else 0
        delay_rate = (delayed_count / total_shipments * 100) if total_shipments > 0 else 0
        
        return {
            'total_shipments': total_shipments,
            'on_time_shipments': total_shipments - delayed_count,
            'delayed_shipments': delayed_count,
            'delay_rate': round(delay_rate, 2),
            'on_time_rate': round(100 - delay_rate, 2),
            'avg_delay_probability': round(shipments_df['delay_probability'].mean(), 3) if 'delay_probability' in shipments_df.columns else None,
            'high_risk_shipments': len(shipments_df[shipments_df['delay_probability'] > 0.8]) if 'delay_probability' in shipments_df.columns else 0,
            'avg_waiting_time': round(shipments_df['waiting_time'].mean(), 1) if 'waiting_time' in shipments_df.columns else None
        }
    
    def _calculate_realtime_metrics(self, shipments_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate real-time operational metrics"""
        if shipments_df.empty:
            return {}
        
        # Current status distribution
        if 'shipment_status' in shipments_df.columns:
            status_counts = shipments_df['shipment_status'].value_counts()
            status_distribution = status_counts.to_dict()
        else:
            status_distribution = {}
        
        # Active shipments (not delivered)
        active_shipments = len(shipments_df[shipments_df.get('shipment_status', '') != 'Delivered']) if 'shipment_status' in shipments_df.columns else len(shipments_df)
        
        # Asset utilization
        if 'asset_id' in shipments_df.columns:
            asset_counts = shipments_df['asset_id'].value_counts()
            total_assets = len(asset_counts)
            avg_shipments_per_asset = asset_counts.mean()
        else:
            total_assets = 0
            avg_shipments_per_asset = 0
        
        return {
            'active_shipments': active_shipments,
            'status_distribution': status_distribution,
            'asset_metrics': {
                'total_assets': total_assets,
                'avg_shipments_per_asset': round(avg_shipments_per_asset, 1),
                'utilization_rate': round((active_shipments / total_assets) * 100, 1) if total_assets > 0 else 0
            },
            'current_timestamp': datetime.now().isoformat()
        }
    
    def _analyze_asset_performance(self, shipments_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze asset performance metrics"""
        if shipments_df.empty or 'asset_id' not in shipments_df.columns:
            return {}
        
        asset_metrics = []
        
        for asset_id in shipments_df['asset_id'].unique():
            asset_data = shipments_df[shipments_df['asset_id'] == asset_id]
            
            total_shipments = len(asset_data)
            delayed_shipments = len(asset_data[asset_data['logistics_delay'] == True]) if 'logistics_delay' in asset_data.columns else 0
            delay_rate = (delayed_shipments / total_shipments * 100) if total_shipments > 0 else 0
            avg_delay_prob = asset_data['delay_probability'].mean() if 'delay_probability' in asset_data.columns else 0
            
            asset_metrics.append({
                'asset_id': asset_id,
                'total_shipments': total_shipments,
                'delayed_shipments': delayed_shipments,
                'delay_rate': round(delay_rate, 2),
                'avg_delay_probability': round(avg_delay_prob, 3),
                'performance_score': round((100 - delay_rate) * (1 - avg_delay_prob), 2)
            })
        
        # Sort by performance score
        asset_metrics.sort(key=lambda x: x['performance_score'], reverse=True)
        
        return {
            'asset_performance': asset_metrics,
            'top_performers': asset_metrics[:3],
            'underperformers': [asset for asset in asset_metrics if asset['delay_rate'] > 15]
        }
    
    def _prepare_daily_data(self, shipments_df: pd.DataFrame) -> pd.DataFrame:
        """Prepare daily aggregated data"""
        if shipments_df.empty or 'timestamp' not in shipments_df.columns:
            return pd.DataFrame()
        
        shipments_df['timestamp'] = pd.to_datetime(shipments_df['timestamp'])
        daily_data = shipments_df.groupby(shipments_df['timestamp'].dt.date).agg({
            'id': 'count',
            'logistics_delay': lambda x: (x == True).sum() if 'logistics_delay' in shipments_df.columns else 0,
            'delay_probability': 'mean',
            'waiting_time': 'mean'
        }).reset_index()
        
        daily_data.columns = ['timestamp', 'id', 'logistics_delay', 'delay_probability', 'waiting_time']
        return daily_data
    
    def _generate_scenario_analysis(self, shipments_df: pd.DataFrame) -> Dict[str, Any]:
        """Generate scenario analysis for forecasting"""
        scenarios = {
            'optimistic': {
                'description': 'Best case scenario with improved operations',
                'assumptions': ['10% improvement in delay rate', 'Enhanced ML prediction accuracy', 'Optimized asset utilization'],
                'projected_metrics': {}
            },
            'realistic': {
                'description': 'Most likely scenario based on current trends',
                'assumptions': ['Current trends continue', 'Moderate operational improvements', 'Stable external conditions'],
                'projected_metrics': {}
            },
            'pessimistic': {
                'description': 'Worst case scenario with potential challenges',
                'assumptions': ['15% increase in delays due to external factors', 'Resource constraints', 'Market volatility'],
                'projected_metrics': {}
            }
        }
        
        # Calculate baseline metrics
        if not shipments_df.empty:
            current_delay_rate = (shipments_df['logistics_delay'].sum() / len(shipments_df) * 100) if 'logistics_delay' in shipments_df.columns else 0
            current_volume = len(shipments_df)
            
            # Project scenarios
            scenarios['optimistic']['projected_metrics'] = {
                'delay_rate': max(0, current_delay_rate * 0.9),  # 10% improvement
                'volume_change': 15,  # 15% increase
                'efficiency_gain': 12
            }
            
            scenarios['realistic']['projected_metrics'] = {
                'delay_rate': current_delay_rate,
                'volume_change': 5,  # 5% increase
                'efficiency_gain': 3
            }
            
            scenarios['pessimistic']['projected_metrics'] = {
                'delay_rate': min(100, current_delay_rate * 1.15),  # 15% deterioration
                'volume_change': -5,  # 5% decrease
                'efficiency_gain': -8
            }
        
        return scenarios
    
    def _calculate_kpi_trends(self, shipments_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate KPI trends over time"""
        if shipments_df.empty or 'timestamp' not in shipments_df.columns:
            return {}
        
        shipments_df['timestamp'] = pd.to_datetime(shipments_df['timestamp'])
        
        # Calculate weekly trends
        weekly_data = shipments_df.groupby(shipments_df['timestamp'].dt.isocalendar().week).agg({
            'id': 'count',
            'logistics_delay': lambda x: (x == True).mean() * 100 if 'logistics_delay' in shipments_df.columns else 0,
            'delay_probability': 'mean'
        }).reset_index()
        
        # Calculate trends
        if len(weekly_data) > 1:
            delay_trend = self.analytics_engine._calculate_trend(weekly_data['logistics_delay'].values)
            volume_trend = self.analytics_engine._calculate_trend(weekly_data['id'].values)
        else:
            delay_trend = {'direction': 'insufficient_data'}
            volume_trend = {'direction': 'insufficient_data'}
        
        return {
            'delay_rate_trend': delay_trend,
            'volume_trend': volume_trend,
            'weekly_data': weekly_data.to_dict('records')
        }
    
    def _generate_benchmark_comparison(self, kpi_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate benchmark comparison for KPIs"""
        industry_benchmarks = {
            'on_time_delivery_rate': {'excellent': 98, 'good': 95, 'average': 90, 'poor': 85},
            'delay_rate': {'excellent': 2, 'good': 5, 'average': 10, 'poor': 15},
            'customer_satisfaction': {'excellent': 4.8, 'good': 4.5, 'average': 4.0, 'poor': 3.5}
        }
        
        comparisons = {}
        
        for kpi_name, kpi_data in kpi_results.items():
            if kpi_name in industry_benchmarks:
                current_value = kpi_data.get('current_value', 0)
                benchmarks = industry_benchmarks[kpi_name]
                
                # Determine performance tier
                if kpi_name == 'delay_rate':
                    # Lower is better for delay rate
                    if current_value <= benchmarks['excellent']:
                        tier = 'excellent'
                    elif current_value <= benchmarks['good']:
                        tier = 'good'
                    elif current_value <= benchmarks['average']:
                        tier = 'average'
                    else:
                        tier = 'poor'
                else:
                    # Higher is better for other metrics
                    if current_value >= benchmarks['excellent']:
                        tier = 'excellent'
                    elif current_value >= benchmarks['good']:
                        tier = 'good'
                    elif current_value >= benchmarks['average']:
                        tier = 'average'
                    else:
                        tier = 'poor'
                
                comparisons[kpi_name] = {
                    'current_value': current_value,
                    'benchmark_tier': tier,
                    'industry_benchmarks': benchmarks,
                    'percentile_rank': self._calculate_percentile_rank(current_value, benchmarks, kpi_name == 'delay_rate')
                }
        
        return comparisons
    
    def _calculate_percentile_rank(self, value: float, benchmarks: Dict[str, float], lower_is_better: bool = False) -> int:
        """Calculate percentile rank against benchmarks"""
        benchmark_values = list(benchmarks.values())
        
        if lower_is_better:
            benchmark_values.sort()
            rank = sum(1 for bv in benchmark_values if value <= bv) / len(benchmark_values) * 100
        else:
            benchmark_values.sort(reverse=True)
            rank = sum(1 for bv in benchmark_values if value >= bv) / len(benchmark_values) * 100
        
        return min(100, max(0, int(rank)))
    
    def _get_date_range(self, shipments_df: pd.DataFrame) -> Dict[str, str]:
        """Get date range from shipment data"""
        if shipments_df.empty or 'timestamp' not in shipments_df.columns:
            return {'start': 'N/A', 'end': 'N/A'}
        
        timestamps = pd.to_datetime(shipments_df['timestamp'])
        return {
            'start': timestamps.min().strftime('%Y-%m-%d'),
            'end': timestamps.max().strftime('%Y-%m-%d')
        }
    
    # Visualization methods
    def _create_executive_visualizations(self, shipments_df: pd.DataFrame, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Create executive summary visualizations"""
        visualizations: Dict[str, Any] = {}
        
        try:
            # KPI Summary Chart (interactive JSON only to avoid static export hang)
            kpi_fig = self._create_kpi_summary_chart(metrics)
            visualizations['kpi_summary_json'] = kpi_fig.to_json()
            
            # Delay Trend Chart
            if not shipments_df.empty and 'timestamp' in shipments_df.columns:
                trend_fig = self._create_delay_trend_chart(shipments_df)
                visualizations['delay_trend_json'] = trend_fig.to_json()
            
            # Status Distribution Pie Chart
            status_fig = self._create_status_distribution_chart(shipments_df)
            visualizations['status_distribution_json'] = status_fig.to_json()
            
        except Exception as e:
            logger.error(f"Error creating executive visualizations: {e}")
        
        return visualizations
    
    def _create_operational_visualizations(self, shipments_df: pd.DataFrame, metrics: Dict[str, Any]) -> Dict[str, str]:
        """Create operational dashboard visualizations"""
        visualizations = {}
        
        try:
            # Real-time Metrics Dashboard
            metrics_fig = self._create_metrics_dashboard(metrics)
            visualizations['metrics_dashboard'] = self._fig_to_base64(metrics_fig)
            
            # Asset Performance Chart
            if 'asset_id' in shipments_df.columns:
                asset_fig = self._create_asset_performance_chart(shipments_df)
                visualizations['asset_performance'] = self._fig_to_base64(asset_fig)
            
            # Hourly Activity Heatmap
            if 'timestamp' in shipments_df.columns:
                heatmap_fig = self._create_activity_heatmap(shipments_df)
                visualizations['activity_heatmap'] = self._fig_to_base64(heatmap_fig)
            
        except Exception as e:
            logger.error(f"Error creating operational visualizations: {e}")
        
        return visualizations
    
    def _create_predictive_visualizations(self, shipments_df: pd.DataFrame, predictions: Dict[str, Any]) -> Dict[str, str]:
        """Create predictive forecast visualizations"""
        visualizations = {}
        
        try:
            # Forecast Chart
            if 'forecast' in predictions and predictions['forecast']:
                forecast_fig = self._create_forecast_chart(predictions['forecast'])
                visualizations['forecast'] = self._fig_to_base64(forecast_fig)
            
            # Risk Assessment Chart
            if 'risk_assessment' in predictions:
                risk_fig = self._create_risk_assessment_chart(predictions['risk_assessment'])
                visualizations['risk_assessment'] = self._fig_to_base64(risk_fig)
            
        except Exception as e:
            logger.error(f"Error creating predictive visualizations: {e}")
        
        return visualizations
    
    def _create_kpi_visualizations(self, kpi_results: Dict, trends: Dict) -> Dict[str, str]:
        """Create KPI scorecard visualizations"""
        visualizations = {}
        
        try:
            # KPI Scorecard
            scorecard_fig = self._create_kpi_scorecard(kpi_results)
            visualizations['kpi_scorecard'] = self._fig_to_base64(scorecard_fig)
            
            # Trend Sparklines
            if trends and 'weekly_data' in trends:
                trend_fig = self._create_trend_sparklines(trends['weekly_data'])
                visualizations['trend_sparklines'] = self._fig_to_base64(trend_fig)
            
        except Exception as e:
            logger.error(f"Error creating KPI visualizations: {e}")
        
        return visualizations
    
    # Individual chart creation methods
    def _create_kpi_summary_chart(self, metrics: Dict[str, Any]) -> go.Figure:
        """Create KPI summary chart"""
        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{"type": "indicator"}, {"type": "indicator"}],
                   [{"type": "indicator"}, {"type": "indicator"}]],
            vertical_spacing=0.25,
            horizontal_spacing=0.15
        )
        
        # Total Shipments
        fig.add_trace(go.Indicator(
            mode="number",
            value=metrics.get('total_shipments', 0),
            title={'text': "Total Shipments", 'font': {'size': 16}},
            number={'font': {'size': 40}},
            domain={'y': [0.05, 0.35]}
        ), row=1, col=1)
        
        # Delay Rate
        delay_rate = metrics.get('delay_rate', 0)
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=delay_rate,
            title={'text': "Delay Rate (%)", 'font': {'size': 16}},
            number={'font': {'size': 20}},
            gauge={
                'axis': {'range': [None, 20]},
                'bar': {'color': "red" if delay_rate > 10 else "yellow" if delay_rate > 5 else "green"},
                'steps': [{'range': [0, 5], 'color': "lightgray"},
                         {'range': [5, 10], 'color': "gray"},
                         {'range': [10, 20], 'color': "lightcoral"}],
                'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 10}
            },
            domain={'y': [0.05, 0.35]}
        ), row=1, col=2)
        
        # On-Time Rate
        on_time_rate = metrics.get('on_time_rate', 0)
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=on_time_rate,
            title={'text': "On-Time Rate (%)", 'font': {'size': 16}},
            number={'font': {'size': 20}},
            gauge={
                'axis': {'range': [80, 100]},
                'bar': {'color': "green" if on_time_rate > 95 else "yellow" if on_time_rate > 90 else "red"},
                'steps': [{'range': [80, 90], 'color': "lightcoral"},
                         {'range': [90, 95], 'color': "gray"},
                         {'range': [95, 100], 'color': "lightgreen"}],
                'threshold': {'line': {'color': "green", 'width': 4}, 'thickness': 0.75, 'value': 95}
            },
            domain={'y': [0.05, 0.35]}
        ), row=2, col=1)
        
        # High Risk Shipments
        fig.add_trace(go.Indicator(
            mode="number",
            value=metrics.get('high_risk_shipments', 0),
            title={'text': "High Risk Shipments", 'font': {'size': 16}},
            number={'font': {'size': 40}},
            domain={'y': [0.05, 0.35]}
        ), row=2, col=2)
        
        fig.update_layout(
            height=600, 
            title_text="Executive KPI Summary",
            showlegend=False,
            margin=dict(t=80, b=40, l=40, r=40)
        )
        return fig
    
    def _create_delay_trend_chart(self, shipments_df: pd.DataFrame) -> go.Figure:
        """Create delay trend chart"""
        shipments_df['timestamp'] = pd.to_datetime(shipments_df['timestamp'])
        daily_delays = shipments_df.groupby(shipments_df['timestamp'].dt.date).agg({
            'logistics_delay': lambda x: (x == True).sum() if 'logistics_delay' in shipments_df.columns else 0,
            'id': 'count'
        }).reset_index()
        
        daily_delays['delay_rate'] = (daily_delays['logistics_delay'] / daily_delays['id'] * 100).fillna(0)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=daily_delays['timestamp'],
            y=daily_delays['delay_rate'],
            mode='lines+markers',
            name='Delay Rate %',
            line=dict(color='red', width=2),
            marker=dict(size=6)
        ))
        
        fig.update_layout(
            title='Daily Delay Rate Trend',
            xaxis_title='Date',
            yaxis_title='Delay Rate (%)',
            height=400
        )
        
        return fig
    
    def _create_status_distribution_chart(self, shipments_df: pd.DataFrame) -> go.Figure:
        """Create status distribution pie chart"""
        if 'shipment_status' in shipments_df.columns:
            status_counts = shipments_df['shipment_status'].value_counts()
        else:
            # Fallback based on delay status
            if 'logistics_delay' in shipments_df.columns:
                delayed = shipments_df['logistics_delay'].sum()
                on_time = len(shipments_df) - delayed
                status_counts = pd.Series({'On Time': on_time, 'Delayed': delayed})
            else:
                status_counts = pd.Series({'Unknown': len(shipments_df)})
        
        fig = go.Figure(data=[go.Pie(
            labels=status_counts.index,
            values=status_counts.values,
            hole=.3,
            textinfo='label+percent',
            textposition='inside'
        )])
        
        fig.update_layout(
            title='Shipment Status Distribution',
            height=400
        )
        
        return fig
    
    def _create_metrics_dashboard(self, metrics: Dict[str, Any]) -> go.Figure:
        """Create metrics dashboard"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Active Shipments', 'Asset Utilization', 'Current Status', 'Performance'),
            specs=[[{"type": "indicator"}, {"type": "indicator"}],
                   [{"type": "pie"}, {"type": "bar"}]]
        )
        
        # Active Shipments
        fig.add_trace(go.Indicator(
            mode="number",
            value=metrics.get('active_shipments', 0),
            title={'text': "Active Shipments"}
        ), row=1, col=1)
        
        # Asset Utilization
        utilization = metrics.get('asset_metrics', {}).get('utilization_rate', 0)
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=utilization,
            title={'text': "Asset Utilization (%)"},
            gauge={'axis': {'range': [None, 100]}}
        ), row=1, col=2)
        
        fig.update_layout(height=600, title_text="Operational Metrics Dashboard")
        return fig
    
    def _create_asset_performance_chart(self, shipments_df: pd.DataFrame) -> go.Figure:
        """Create asset performance chart"""
        asset_performance = self._analyze_asset_performance(shipments_df)
        asset_data = asset_performance.get('asset_performance', [])
        
        if not asset_data:
            return go.Figure().add_annotation(text="No asset data available", showarrow=False)
        
        assets = [item['asset_id'] for item in asset_data]
        delay_rates = [item['delay_rate'] for item in asset_data]
        performance_scores = [item['performance_score'] for item in asset_data]
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Delay Rate by Asset', 'Performance Score by Asset'),
            specs=[[{"type": "bar"}, {"type": "bar"}]]
        )
        
        fig.add_trace(go.Bar(
            x=assets,
            y=delay_rates,
            name='Delay Rate %',
            marker_color='red'
        ), row=1, col=1)
        
        fig.add_trace(go.Bar(
            x=assets,
            y=performance_scores,
            name='Performance Score',
            marker_color='green'
        ), row=1, col=2)
        
        fig.update_layout(height=400, title_text="Asset Performance Analysis")
        return fig
    
    def _sanitize_for_json(self, obj: Any) -> Any:
        """Convert numpy/pandas types to Python native types for JSON serialization"""
        import numpy as np
        
        if isinstance(obj, dict):
            return {k: self._sanitize_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._sanitize_for_json(v) for v in obj]
        elif isinstance(obj, (np.bool_, np.bool8)):
            return bool(obj)
        elif isinstance(obj, (np.integer, np.int8, np.int16, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, 'isoformat'):
            return obj.isoformat()
        elif pd.isna(obj):
            return None
        else:
            return obj
    
    def _create_activity_heatmap(self, shipments_df: pd.DataFrame) -> go.Figure:
        """Create activity heatmap"""
        shipments_df['timestamp'] = pd.to_datetime(shipments_df['timestamp'])
        shipments_df['hour'] = shipments_df['timestamp'].dt.hour
        shipments_df['day_of_week'] = shipments_df['timestamp'].dt.day_name()
        
        heatmap_data = shipments_df.groupby(['day_of_week', 'hour']).size().reset_index(name='count')
        
        # Create pivot table for heatmap
        pivot_table = heatmap_data.pivot(index='day_of_week', columns='hour', values='count').fillna(0)
        
        fig = go.Figure(data=go.Heatmap(
            z=pivot_table.values,
            x=pivot_table.columns,
            y=pivot_table.index,
            colorscale='Viridis',
            showscale=True
        ))
        
        fig.update_layout(
            title='Shipment Activity Heatmap',
            xaxis_title='Hour of Day',
            yaxis_title='Day of Week',
            height=400
        )
        
        return fig
    
    def _create_forecast_chart(self, forecast_data: Dict[str, Any]) -> go.Figure:
        """Create forecast visualization"""
        forecasts = forecast_data.get('forecasts', [])
        if not forecasts:
            return go.Figure().add_annotation(text="No forecast data available", showarrow=False)
        
        dates = [f['date'] for f in forecasts]
        volumes = [f['predicted_volume'] for f in forecasts]
        delay_rates = [f['predicted_delay_rate'] for f in forecasts]
        confidence = [f['confidence'] for f in forecasts]
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Volume Forecast', 'Delay Rate Forecast'),
            specs=[[{"type": "scatter"}], [{"type": "scatter"}]]
        )
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=volumes,
            mode='lines+markers',
            name='Predicted Volume',
            line=dict(color='blue')
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=delay_rates,
            mode='lines+markers',
            name='Predicted Delay Rate %',
            line=dict(color='red')
        ), row=2, col=1)
        
        fig.update_layout(height=600, title_text="7-Day Forecast")
        return fig
    
    def _create_risk_assessment_chart(self, risk_data: Dict[str, Any]) -> go.Figure:
        """Create risk assessment chart"""
        risk_factors = risk_data.get('risk_factors', [])
        if not risk_factors:
            return go.Figure().add_annotation(text="No risk factors identified", showarrow=False)
        
        factors = [rf['factor'] for rf in risk_factors]
        levels = [rf['level'] for rf in risk_factors]
        
        # Map levels to numeric values
        level_map = {'low': 1, 'medium': 2, 'high': 3}
        level_values = [level_map.get(level, 0) for level in levels]
        
        fig = go.Figure(data=[go.Bar(
            x=factors,
            y=level_values,
            marker=dict(
                color=level_values,
                colorscale='RdYlGn_r',
                showscale=True,
                colorbar=dict(
                    tickvals=[1, 2, 3],
                    ticktext=['Low', 'Medium', 'High']
                )
            )
        )])
        
        fig.update_layout(
            title='Risk Factor Assessment',
            xaxis_title='Risk Factors',
            yaxis_title='Risk Level',
            height=400
        )
        
        return fig
    
    def _create_kpi_scorecard(self, kpi_results: Dict[str, Any]) -> go.Figure:
        """Create KPI scorecard visualization"""
        if not kpi_results:
            return go.Figure().add_annotation(text="No KPI data available", showarrow=False)
        
        kpi_names = list(kpi_results.keys())
        current_values = [kpi_results[kpi]['current_value'] for kpi in kpi_names]
        targets = [kpi_results[kpi]['target'] for kpi in kpi_names]
        statuses = [kpi_results[kpi]['status'] for kpi in kpi_names]
        
        # Color map for status
        status_colors = {
            'excellent': 'green',
            'good': 'lightgreen', 
            'warning': 'yellow',
            'critical': 'red'
        }
        
        colors = [status_colors.get(status, 'gray') for status in statuses]
        
        fig = go.Figure(data=[go.Bar(
            x=kpi_names,
            y=current_values,
            name='Current',
            marker_color=colors,
            text=[f'{val:.1f}' for val in current_values],
            textposition='auto'
        )])
        
        fig.add_trace(go.Scatter(
            x=kpi_names,
            y=targets,
            mode='markers',
            name='Target',
            marker=dict(size=15, symbol='diamond', color='red'),
        ))
        
        fig.update_layout(
            title='KPI Performance Scorecard',
            xaxis_title='KPIs',
            yaxis_title='Values',
            height=400
        )
        
        return fig
    
    def _create_trend_sparklines(self, weekly_data: List[Dict]) -> go.Figure:
        """Create trend sparklines"""
        weeks = [item['week'] for item in weekly_data]
        delay_rates = [item['logistics_delay'] for item in weekly_data]
        volumes = [item['id'] for item in weekly_data]
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Weekly Delay Rate Trend', 'Weekly Volume Trend'),
            specs=[[{"type": "scatter"}], [{"type": "scatter"}]]
        )
        
        fig.add_trace(go.Scatter(
            x=weeks,
            y=delay_rates,
            mode='lines',
            name='Delay Rate',
            line=dict(width=2, color='red')
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=weeks,
            y=volumes,
            mode='lines',
            name='Volume',
            line=dict(width=2, color='blue')
        ), row=2, col=1)
        
        fig.update_layout(height=400, title_text="Performance Trends")
        return fig
    
    def _fig_to_base64(self, fig: go.Figure) -> str:
        """Convert Plotly figure to base64 string"""
        try:
            img_bytes = pio.to_image(fig, format='png', engine='kaleido')
            img_base64 = base64.b64encode(img_bytes).decode()
            return f"data:image/png;base64,{img_base64}"
        except Exception as e:
            logger.error(f"Error converting figure to base64: {e}")
            return ""
    
    def export_report_to_json(self, report_data: Dict[str, Any], filename: str) -> str:
        """Export report to JSON file"""
        try:
            with open(filename, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)
            
            logger.info(f"Report exported to {filename}")
            return filename
        except Exception as e:
            logger.error(f"Error exporting report: {e}")
            return str(e)


# Global reporting service instance
_reporting_service = None

def get_reporting_service() -> ReportingService:
    """Get or create the reporting service singleton"""
    global _reporting_service
    if _reporting_service is None:
        _reporting_service = ReportingService()
    return _reporting_service


if __name__ == "__main__":
    # Example usage
    import pandas as pd
    import numpy as np
    
    # Create sample data for testing
    sample_data = {
        'id': range(1, 101),
        'timestamp': pd.date_range(start='2024-01-01', periods=100, freq='H'),
        'logistics_delay': np.random.choice([True, False], 100, p=[0.3, 0.7]),
        'delay_probability': np.random.beta(2, 5, 100),
        'waiting_time': np.random.exponential(30, 100),
        'asset_id': np.random.choice(['Truck_1', 'Truck_2', 'Truck_3'], 100),
        'shipment_status': np.random.choice(['In Transit', 'Delivered', 'Delayed'], 100)
    }
    
    df = pd.DataFrame(sample_data)
    
    # Test reporting service
    reporting = get_reporting_service()
    
    # Generate executive summary
    executive_report = reporting.generate_executive_summary_report(df)
    if executive_report['success']:
        print("Executive Summary Report Generated Successfully")
        print(f"Total Shipments: {executive_report['report']['overview']['total_shipments']}")
    
    # Generate operational dashboard
    operational_report = reporting.generate_operational_dashboard_report(df)
    if operational_report['success']:
        print("Operational Dashboard Report Generated Successfully")
        print(f"Active Shipments: {operational_report['report']['real_time_metrics']['active_shipments']}")
    
    # Export report
    filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    reporting.export_report_to_json(executive_report, filename)
    print(f"Report exported to: {filename}")
