"""
Advanced Analytics Engine for Maersk Shipment AI System

This module provides comprehensive business intelligence, trend analysis, 
predictive insights, and advanced reporting capabilities.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import logging
from dataclasses import dataclass, asdict
from enum import Enum
import json

# Statistical and ML imports
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

logger = logging.getLogger(__name__)


class AnalysisType(Enum):
    DESCRIPTIVE = "descriptive"
    DIAGNOSTIC = "diagnostic"
    PREDICTIVE = "predictive"
    PRESCRIPTIVE = "prescriptive"


class TimeGranularity(Enum):
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"


@dataclass
class AnalyticsReport:
    """Analytics report structure"""
    report_id: str
    title: str
    analysis_type: AnalysisType
    generated_at: datetime
    data_period: Tuple[datetime, datetime]
    insights: List[Dict[str, Any]]
    visualizations: List[Dict[str, Any]]
    recommendations: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    
    def to_dict(self):
        return asdict(self)


@dataclass
class BusinessInsight:
    """Business insight structure"""
    insight_id: str
    category: str
    title: str
    description: str
    impact: str
    confidence: float
    data_points: List[Dict[str, Any]]
    recommendations: List[str]
    created_at: datetime


class AdvancedAnalyticsEngine:
    """
    Advanced analytics engine providing comprehensive business intelligence
    """
    
    def __init__(self):
        self.analysis_cache = {}
        self.report_history = []
        self.kpis = {}
        
        # Configure default KPIs
        self._setup_default_kpis()
    
    def _setup_default_kpis(self):
        """Setup default Key Performance Indicators"""
        self.kpis = {
            'operational_efficiency': {
                'on_time_delivery_rate': {
                    'target': 95.0,
                    'warning_threshold': 90.0,
                    'critical_threshold': 85.0
                },
                'delay_rate': {
                    'target': 5.0,
                    'warning_threshold': 10.0,
                    'critical_threshold': 15.0
                },
                'avg_delivery_time': {
                    'target_days': 3.0,
                    'warning_threshold': 4.0,
                    'critical_threshold': 5.0
                }
            },
            'predictive_accuracy': {
                'ml_model_accuracy': {
                    'target': 85.0,
                    'warning_threshold': 80.0,
                    'critical_threshold': 75.0
                },
                'prediction_reliability': {
                    'target': 90.0,
                    'warning_threshold': 85.0,
                    'critical_threshold': 80.0
                }
            },
            'business_impact': {
                'cost_savings_from_predictions': {
                    'target_percentage': 15.0,
                    'warning_threshold': 10.0,
                    'critical_threshold': 5.0
                },
                'customer_satisfaction': {
                    'target_score': 4.5,
                    'warning_threshold': 4.0,
                    'critical_threshold': 3.5
                }
            }
        }
    
    def analyze_shipment_trends(self, shipments_df: pd.DataFrame, time_granularity: TimeGranularity = TimeGranularity.DAILY) -> Dict[str, Any]:
        """Analyze shipment trends over time"""
        try:
            if shipments_df.empty:
                return {"error": "No data available for trend analysis"}
            
            # Ensure timestamp column
            if 'timestamp' not in shipments_df.columns:
                return {"error": "Timestamp column not found"}
            
            shipments_df['timestamp'] = pd.to_datetime(shipments_df['timestamp'])
            
            # Group by time granularity
            if time_granularity == TimeGranularity.HOURLY:
                time_group = shipments_df.groupby(shipments_df['timestamp'].dt.floor('H'))
            elif time_granularity == TimeGranularity.DAILY:
                time_group = shipments_df.groupby(shipments_df['timestamp'].dt.date)
            elif time_granularity == TimeGranularity.WEEKLY:
                time_group = shipments_df.groupby(shipments_df['timestamp'].dt.isocalendar().week)
            elif time_granularity == TimeGranularity.MONTHLY:
                time_group = shipments_df.groupby(shipments_df['timestamp'].dt.to_period('M'))
            else:
                time_group = shipments_df.groupby(shipments_df['timestamp'].dt.date)
            
            # Calculate trend metrics
            trend_data = time_group.agg({
                'id': 'count',  # Total shipments
                'logistics_delay': lambda x: (x == True).sum(),  # Delayed shipments
                'delay_probability': ['mean', 'std'],
                'waiting_time': ['mean', 'median', 'max']
            }).round(2)
            
            # Flatten column names
            trend_data.columns = ['_'.join(col).strip() for col in trend_data.columns.values]
            trend_data = trend_data.reset_index()
            
            # Calculate delay rate
            trend_data['delay_rate'] = (trend_data['logistics_delay_<lambda>'] / trend_data['id_count'] * 100).round(2)
            
            # Trend analysis
            delay_rate_trend = self._calculate_trend(trend_data['delay_rate'].values)
            volume_trend = self._calculate_trend(trend_data['id_count'].values)
            
            return {
                'trend_data': trend_data.to_dict('records'),
                'summary': {
                    'total_periods': len(trend_data),
                    'avg_shipments_per_period': trend_data['id_count'].mean(),
                    'avg_delay_rate': trend_data['delay_rate'].mean(),
                    'delay_rate_trend': delay_rate_trend,
                    'volume_trend': volume_trend,
                    'peak_volume_period': trend_data.loc[trend_data['id_count'].idxmax(), trend_data.columns[0]],
                    'highest_delay_period': trend_data.loc[trend_data['delay_rate'].idxmax(), trend_data.columns[0]]
                },
                'insights': self._generate_trend_insights(trend_data, delay_rate_trend, volume_trend)
            }
            
        except Exception as e:
            logger.error(f"Error in trend analysis: {e}")
            return {"error": str(e)}
    
    def _calculate_trend(self, values: np.ndarray) -> Dict[str, Any]:
        """Calculate trend direction and strength"""
        if len(values) < 2:
            return {"direction": "insufficient_data", "strength": 0, "slope": 0}
        
        x = np.arange(len(values))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
        
        # Determine trend direction
        if abs(slope) < 0.01:
            direction = "stable"
        elif slope > 0:
            direction = "increasing"
        else:
            direction = "decreasing"
        
        # Trend strength based on correlation coefficient
        strength = abs(r_value)
        
        return {
            "direction": direction,
            "strength": strength,
            "slope": slope,
            "r_squared": r_value ** 2,
            "p_value": p_value,
            "significant": p_value < 0.05
        }
    
    def _generate_trend_insights(self, trend_data: pd.DataFrame, delay_trend: Dict, volume_trend: Dict) -> List[Dict[str, Any]]:
        """Generate insights from trend analysis"""
        insights = []
        
        # Delay rate insights
        if delay_trend['direction'] == 'increasing' and delay_trend['significant']:
            insights.append({
                'type': 'warning',
                'title': 'Increasing Delay Rate Trend',
                'description': f"Delay rates are trending upward with {delay_trend['strength']:.2f} correlation",
                'impact': 'high',
                'recommendation': 'Investigate root causes and implement preventive measures'
            })
        elif delay_trend['direction'] == 'decreasing' and delay_trend['significant']:
            insights.append({
                'type': 'positive',
                'title': 'Improving Delay Performance',
                'description': f"Delay rates are trending downward with {delay_trend['strength']:.2f} correlation",
                'impact': 'high',
                'recommendation': 'Continue current optimization strategies'
            })
        
        # Volume insights
        if volume_trend['direction'] == 'increasing' and volume_trend['significant']:
            insights.append({
                'type': 'info',
                'title': 'Growing Shipment Volume',
                'description': f"Shipment volume is increasing with {volume_trend['strength']:.2f} correlation",
                'impact': 'medium',
                'recommendation': 'Ensure capacity planning and resource allocation'
            })
        
        # Seasonal patterns
        if len(trend_data) >= 7:  # At least a week of data
            delay_variance = trend_data['delay_rate'].var()
            if delay_variance > 25:  # High variance threshold
                insights.append({
                    'type': 'info',
                    'title': 'High Delay Rate Variability',
                    'description': f"Delay rates show high variability (variance: {delay_variance:.2f})",
                    'impact': 'medium',
                    'recommendation': 'Investigate cyclical patterns and stabilize operations'
                })
        
        return insights
    
    def perform_root_cause_analysis(self, shipments_df: pd.DataFrame) -> Dict[str, Any]:
        """Perform root cause analysis for delays"""
        try:
            if shipments_df.empty:
                return {"error": "No data available for root cause analysis"}
            
            # Create a safe copy to avoid chained-assignment warnings
            delayed_shipments = shipments_df[shipments_df['logistics_delay'] == True].copy()
            
            if delayed_shipments.empty:
                return {"message": "No delayed shipments found for analysis"}
            
            # Analyze delay reasons
            if 'logistics_delay_reason' in delayed_shipments.columns:
                delay_reasons = delayed_shipments['logistics_delay_reason'].value_counts()
                delay_reason_pct = (delay_reasons / len(delayed_shipments) * 100).round(2)
            else:
                delay_reasons = pd.Series(dtype='int64')
                delay_reason_pct = pd.Series(dtype='float64')
            
            # Analyze by asset
            if 'asset_id' in delayed_shipments.columns:
                asset_delays = delayed_shipments['asset_id'].value_counts()
                asset_delay_rates = []
                
                for asset in asset_delays.index:
                    total_asset_shipments = len(shipments_df[shipments_df['asset_id'] == asset])
                    delay_rate = (asset_delays[asset] / total_asset_shipments * 100) if total_asset_shipments > 0 else 0
                    asset_delay_rates.append({
                        'asset_id': asset,
                        'total_delays': int(asset_delays[asset]),
                        'delay_rate': round(delay_rate, 2)
                    })
                
                asset_delay_rates.sort(key=lambda x: x['delay_rate'], reverse=True)
            else:
                asset_delay_rates = []
            
            # Time-based analysis
            delayed_shipments.loc[:, 'timestamp'] = pd.to_datetime(delayed_shipments['timestamp'])
            delayed_shipments.loc[:, 'hour'] = delayed_shipments['timestamp'].dt.hour
            delayed_shipments.loc[:, 'day_of_week'] = delayed_shipments['timestamp'].dt.day_name()
            
            hourly_delays = delayed_shipments['hour'].value_counts().sort_index()
            daily_delays = delayed_shipments['day_of_week'].value_counts()
            
            # Geographic analysis
            if 'latitude' in delayed_shipments.columns and 'longitude' in delayed_shipments.columns:
                # Simple geographic clustering
                coords = delayed_shipments[['latitude', 'longitude']].dropna()
                if len(coords) > 10:
                    try:
                        kmeans = KMeans(n_clusters=min(5, len(coords)//3), random_state=42, n_init=10)
                        clusters = kmeans.fit_predict(coords)
                        
                        geo_hotspots = []
                        for i in range(kmeans.n_clusters):
                            cluster_coords = coords[clusters == i]
                            center_lat = cluster_coords['latitude'].mean()
                            center_lon = cluster_coords['longitude'].mean()
                            count = len(cluster_coords)
                            
                            geo_hotspots.append({
                                'cluster_id': i,
                                'center_latitude': round(center_lat, 4),
                                'center_longitude': round(center_lon, 4),
                                'delay_count': count
                            })
                        
                        geo_hotspots.sort(key=lambda x: x['delay_count'], reverse=True)
                    except Exception as e:
                        logger.warning(f"Geographic clustering failed: {e}")
                        geo_hotspots = []
                else:
                    geo_hotspots = []
            else:
                geo_hotspots = []
            
            # Statistical analysis
            delay_stats = {
                'total_delayed': len(delayed_shipments),
                'delay_rate': round(len(delayed_shipments) / len(shipments_df) * 100, 2),
                'avg_delay_probability': round(delayed_shipments['delay_probability'].mean(), 3),
                'avg_waiting_time': round(delayed_shipments['waiting_time'].mean(), 1) if 'waiting_time' in delayed_shipments.columns else None
            }
            
            # Generate insights
            insights = self._generate_root_cause_insights(
                delay_stats, delay_reason_pct, asset_delay_rates, 
                hourly_delays, daily_delays, geo_hotspots
            )
            
            return {
                'summary': delay_stats,
                'delay_reasons': {
                    'counts': delay_reasons.to_dict(),
                    'percentages': delay_reason_pct.to_dict()
                },
                'asset_analysis': asset_delay_rates[:10],  # Top 10
                'temporal_patterns': {
                    'hourly_distribution': hourly_delays.to_dict(),
                    'daily_distribution': daily_delays.to_dict(),
                    'peak_delay_hour': int(hourly_delays.idxmax()) if len(hourly_delays) > 0 else None,
                    'worst_day': daily_delays.idxmax() if len(daily_delays) > 0 else None
                },
                'geographic_hotspots': geo_hotspots[:5],  # Top 5
                'insights': insights
            }
            
        except Exception as e:
            logger.error(f"Error in root cause analysis: {e}")
            return {"error": str(e)}
    
    def _generate_root_cause_insights(self, delay_stats, delay_reasons, asset_delays, hourly_delays, daily_delays, geo_hotspots):
        """Generate insights from root cause analysis"""
        insights = []
        
        # Overall delay rate insight
        delay_rate = delay_stats['delay_rate']
        if delay_rate > 15:
            insights.append({
                'type': 'critical',
                'title': 'High Overall Delay Rate',
                'description': f"System-wide delay rate of {delay_rate}% exceeds acceptable thresholds",
                'recommendations': ['Implement immediate corrective measures', 'Review operational procedures', 'Increase monitoring frequency']
            })
        elif delay_rate > 10:
            insights.append({
                'type': 'warning',
                'title': 'Elevated Delay Rate',
                'description': f"Delay rate of {delay_rate}% requires attention",
                'recommendations': ['Investigate primary delay causes', 'Implement preventive measures']
            })
        
        # Top delay reason insight
        if len(delay_reasons) > 0:
            top_reason = delay_reasons.index[0]
            top_percentage = delay_reasons.iloc[0]
            
            if top_percentage > 40:
                insights.append({
                    'type': 'actionable',
                    'title': f'Dominant Delay Cause: {top_reason}',
                    'description': f"{top_reason} accounts for {top_percentage}% of all delays",
                    'recommendations': [f'Focus improvement efforts on {top_reason}', 'Develop targeted mitigation strategies']
                })
        
        # Asset-specific insights
        if len(asset_delays) > 0:
            problematic_assets = [asset for asset in asset_delays if asset['delay_rate'] > 20]
            if problematic_assets:
                insights.append({
                    'type': 'warning',
                    'title': 'High-Risk Assets Identified',
                    'description': f"{len(problematic_assets)} assets have delay rates > 20%",
                    'recommendations': ['Schedule maintenance for high-risk assets', 'Review asset utilization patterns']
                })
        
        # Temporal pattern insights
        if len(hourly_delays) > 0:
            peak_hour = hourly_delays.idxmax()
            peak_count = hourly_delays.max()
            
            if peak_count > len(hourly_delays) * 0.2:  # More than 20% of delays in one hour
                insights.append({
                    'type': 'info',
                    'title': f'Peak Delay Period: {peak_hour}:00',
                    'description': f"Hour {peak_hour} accounts for {peak_count} delays",
                    'recommendations': ['Investigate operational bottlenecks during peak hours', 'Consider resource reallocation']
                })
        
        # Geographic insights
        if len(geo_hotspots) > 0:
            top_hotspot = geo_hotspots[0]
            if top_hotspot['delay_count'] > 5:
                insights.append({
                    'type': 'info',
                    'title': 'Geographic Delay Hotspot Detected',
                    'description': f"Location cluster has {top_hotspot['delay_count']} delays",
                    'recommendations': ['Investigate regional factors affecting delays', 'Consider route optimization']
                })
        
        return insights
    
    def generate_predictive_insights(self, shipments_df: pd.DataFrame, ml_predictions: Optional[Dict] = None) -> Dict[str, Any]:
        """Generate predictive insights and forecasts"""
        try:
            if shipments_df.empty:
                return {"error": "No data available for predictive analysis"}
            
            # Prepare time series data
            shipments_df['timestamp'] = pd.to_datetime(shipments_df['timestamp'])
            daily_data = shipments_df.groupby(shipments_df['timestamp'].dt.date).agg({
                'id': 'count',
                'logistics_delay': lambda x: (x == True).sum(),
                'delay_probability': 'mean',
                'waiting_time': 'mean'
            }).reset_index()
            
            daily_data['delay_rate'] = (daily_data['logistics_delay'] / daily_data['id'] * 100).round(2)
            
            # Simple forecasting using moving averages and trend
            forecast_days = 7
            predictions = self._generate_simple_forecast(daily_data, forecast_days)
            
            # Risk assessment
            risk_factors = self._assess_risk_factors(shipments_df)
            
            # Capacity analysis
            capacity_insights = self._analyze_capacity_trends(daily_data)
            
            # Generate recommendations
            recommendations = self._generate_predictive_recommendations(predictions, risk_factors, capacity_insights)
            
            return {
                'forecast': predictions,
                'risk_assessment': risk_factors,
                'capacity_analysis': capacity_insights,
                'recommendations': recommendations,
                'ml_integration': ml_predictions if ml_predictions else {'status': 'not_available'},
                'confidence_level': 'medium'  # Could be enhanced with proper ML models
            }
            
        except Exception as e:
            logger.error(f"Error in predictive insights: {e}")
            return {"error": str(e)}
    
    def _generate_simple_forecast(self, daily_data: pd.DataFrame, forecast_days: int) -> Dict[str, Any]:
        """Generate simple forecast using moving averages"""
        if len(daily_data) < 3:
            return {"error": "Insufficient historical data for forecasting"}
        
        # Calculate moving averages
        daily_data['volume_ma3'] = daily_data['id'].rolling(window=3, min_periods=1).mean()
        daily_data['delay_rate_ma3'] = daily_data['delay_rate'].rolling(window=3, min_periods=1).mean()
        
        # Simple linear trend extrapolation
        recent_volume_trend = daily_data['volume_ma3'].tail(3).values
        recent_delay_trend = daily_data['delay_rate_ma3'].tail(3).values
        
        # Generate forecasts
        forecasts = []
        last_date = daily_data['timestamp'].max()
        
        for i in range(1, forecast_days + 1):
            forecast_date = last_date + timedelta(days=i)
            
            # Simple trend projection
            volume_forecast = recent_volume_trend[-1] + (recent_volume_trend[-1] - recent_volume_trend[0]) / 3 * i
            delay_rate_forecast = recent_delay_trend[-1] + (recent_delay_trend[-1] - recent_delay_trend[0]) / 3 * i
            
            # Apply bounds
            volume_forecast = max(0, volume_forecast)
            delay_rate_forecast = max(0, min(100, delay_rate_forecast))
            
            forecasts.append({
                'date': forecast_date.strftime('%Y-%m-%d'),
                'predicted_volume': round(volume_forecast, 0),
                'predicted_delay_rate': round(delay_rate_forecast, 2),
                'confidence': max(0.3, 0.8 - 0.1 * i)  # Decreasing confidence
            })
        
        return {
            'forecasts': forecasts,
            'historical_baseline': {
                'avg_volume': round(daily_data['id'].mean(), 1),
                'avg_delay_rate': round(daily_data['delay_rate'].mean(), 2)
            },
            'forecast_summary': {
                'expected_avg_volume': round(np.mean([f['predicted_volume'] for f in forecasts]), 1),
                'expected_avg_delay_rate': round(np.mean([f['predicted_delay_rate'] for f in forecasts]), 2)
            }
        }
    
    def _assess_risk_factors(self, shipments_df: pd.DataFrame) -> Dict[str, Any]:
        """Assess various risk factors"""
        risk_factors = []
        
        # High delay probability shipments
        high_risk_count = len(shipments_df[shipments_df['delay_probability'] > 0.8])
        total_count = len(shipments_df)
        high_risk_rate = (high_risk_count / total_count * 100) if total_count > 0 else 0
        
        if high_risk_rate > 20:
            risk_factors.append({
                'factor': 'High Delay Probability Shipments',
                'level': 'high',
                'value': f"{high_risk_rate:.1f}%",
                'description': f"{high_risk_count} shipments have >80% delay probability"
            })
        
        # Asset concentration risk
        if 'asset_id' in shipments_df.columns:
            asset_distribution = shipments_df['asset_id'].value_counts()
            max_asset_share = (asset_distribution.iloc[0] / total_count * 100) if len(asset_distribution) > 0 else 0
            
            if max_asset_share > 50:
                risk_factors.append({
                    'factor': 'Asset Concentration',
                    'level': 'medium',
                    'value': f"{max_asset_share:.1f}%",
                    'description': f"Single asset handles {max_asset_share:.1f}% of shipments"
                })
        
        # Temporal concentration
        shipments_df['hour'] = pd.to_datetime(shipments_df['timestamp']).dt.hour
        hourly_dist = shipments_df['hour'].value_counts()
        max_hour_share = (hourly_dist.iloc[0] / total_count * 100) if len(hourly_dist) > 0 else 0
        
        if max_hour_share > 30:
            peak_hour = hourly_dist.index[0]
            risk_factors.append({
                'factor': 'Temporal Concentration',
                'level': 'low',
                'value': f"{max_hour_share:.1f}%",
                'description': f"Peak activity at hour {peak_hour} ({max_hour_share:.1f}% of shipments)"
            })
        
        return {
            'risk_factors': risk_factors,
            'overall_risk_level': 'high' if any(rf['level'] == 'high' for rf in risk_factors) else 
                                 'medium' if any(rf['level'] == 'medium' for rf in risk_factors) else 'low'
        }
    
    def _analyze_capacity_trends(self, daily_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze capacity and utilization trends"""
        if len(daily_data) < 7:
            return {"message": "Insufficient data for capacity analysis"}
        
        # Calculate capacity metrics
        max_daily_volume = daily_data['id'].max()
        avg_daily_volume = daily_data['id'].mean()
        utilization_rate = (avg_daily_volume / max_daily_volume * 100) if max_daily_volume > 0 else 0
        
        # Trend analysis
        recent_week = daily_data.tail(7)
        volume_trend = self._calculate_trend(recent_week['id'].values)
        
        return {
            'capacity_metrics': {
                'max_daily_volume': int(max_daily_volume),
                'avg_daily_volume': round(avg_daily_volume, 1),
                'utilization_rate': round(utilization_rate, 1)
            },
            'volume_trend': volume_trend,
            'capacity_status': 'high' if utilization_rate > 85 else 'medium' if utilization_rate > 70 else 'low'
        }
    
    def _generate_predictive_recommendations(self, predictions, risk_factors, capacity_insights):
        """Generate actionable recommendations based on predictive analysis"""
        recommendations = []
        
        # Volume-based recommendations
        if predictions and 'forecast_summary' in predictions:
            predicted_volume = predictions['forecast_summary']['expected_avg_volume']
            current_capacity = capacity_insights.get('capacity_metrics', {}).get('max_daily_volume', 100)
            
            if predicted_volume > current_capacity * 0.9:
                recommendations.append({
                    'priority': 'high',
                    'category': 'capacity_planning',
                    'title': 'Capacity Expansion Required',
                    'description': f"Predicted volume ({predicted_volume}) approaching capacity limit",
                    'actions': ['Scale up resources', 'Optimize asset utilization', 'Consider additional capacity']
                })
        
        # Risk-based recommendations
        if risk_factors['overall_risk_level'] == 'high':
            recommendations.append({
                'priority': 'high',
                'category': 'risk_mitigation',
                'title': 'High Risk Level Detected',
                'description': 'Multiple high-risk factors identified',
                'actions': ['Implement risk mitigation strategies', 'Increase monitoring frequency', 'Review operational procedures']
            })
        
        # Delay rate recommendations
        if predictions and 'forecast_summary' in predictions:
            predicted_delay_rate = predictions['forecast_summary']['expected_avg_delay_rate']
            
            if predicted_delay_rate > 10:
                recommendations.append({
                    'priority': 'medium',
                    'category': 'operational_improvement',
                    'title': 'Delay Rate Optimization Needed',
                    'description': f"Predicted delay rate of {predicted_delay_rate}% exceeds targets",
                    'actions': ['Investigate root causes', 'Implement preventive measures', 'Enhance monitoring']
                })
        
        return recommendations
    
    def generate_comprehensive_report(self, shipments_df: pd.DataFrame, analysis_types: List[AnalysisType] = None) -> AnalyticsReport:
        """Generate comprehensive analytics report"""
        if analysis_types is None:
            analysis_types = [AnalysisType.DESCRIPTIVE, AnalysisType.DIAGNOSTIC, AnalysisType.PREDICTIVE]
        
        report_id = f"analytics_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        insights = []
        visualizations = []
        recommendations = []
        
        try:
            # Determine data period
            if not shipments_df.empty and 'timestamp' in shipments_df.columns:
                shipments_df['timestamp'] = pd.to_datetime(shipments_df['timestamp'])
                data_start = shipments_df['timestamp'].min()
                data_end = shipments_df['timestamp'].max()
            else:
                data_start = datetime.now() - timedelta(days=30)
                data_end = datetime.now()
            
            # Descriptive analytics
            if AnalysisType.DESCRIPTIVE in analysis_types:
                descriptive_insights = self._generate_descriptive_insights(shipments_df)
                insights.extend(descriptive_insights.get('insights', []))
                visualizations.extend(descriptive_insights.get('visualizations', []))
            
            # Diagnostic analytics (root cause analysis)
            if AnalysisType.DIAGNOSTIC in analysis_types:
                diagnostic_results = self.perform_root_cause_analysis(shipments_df)
                if 'insights' in diagnostic_results:
                    insights.extend(diagnostic_results['insights'])
            
            # Predictive analytics
            if AnalysisType.PREDICTIVE in analysis_types:
                predictive_results = self.generate_predictive_insights(shipments_df)
                if 'recommendations' in predictive_results:
                    recommendations.extend(predictive_results['recommendations'])
            
            # Trend analysis
            trend_results = self.analyze_shipment_trends(shipments_df)
            if 'insights' in trend_results:
                insights.extend(trend_results['insights'])
            
            report = AnalyticsReport(
                report_id=report_id,
                title="Comprehensive Shipment Analytics Report",
                analysis_type=AnalysisType.DESCRIPTIVE,  # Primary type
                generated_at=datetime.now(),
                data_period=(data_start, data_end),
                insights=insights,
                visualizations=visualizations,
                recommendations=recommendations,
                metadata={
                    'data_records': len(shipments_df),
                    'analysis_types': [at.value for at in analysis_types],
                    'kpi_thresholds': self.kpis
                }
            )
            
            self.report_history.append(report)
            return report
            
        except Exception as e:
            logger.error(f"Error generating comprehensive report: {e}")
            # Return minimal report with error
            return AnalyticsReport(
                report_id=report_id,
                title="Analytics Report (Error)",
                analysis_type=AnalysisType.DESCRIPTIVE,
                generated_at=datetime.now(),
                data_period=(data_start, data_end),
                insights=[{"type": "error", "message": str(e)}],
                visualizations=[],
                recommendations=[],
                metadata={"error": str(e)}
            )
    
    def _evaluate_kpis(self, shipments_df: pd.DataFrame) -> Dict[str, Any]:
        """Evaluate current performance against KPIs"""
        if shipments_df.empty:
            return {}
        
        kpi_results = {}
        
        # On-time delivery rate
        if 'logistics_delay' in shipments_df.columns:
            on_time_rate = (1 - shipments_df['logistics_delay'].mean()) * 100
            kpi_target = self.kpis['operational_efficiency']['on_time_delivery_rate']['target']
            warning_threshold = self.kpis['operational_efficiency']['on_time_delivery_rate']['warning_threshold']
            critical_threshold = self.kpis['operational_efficiency']['on_time_delivery_rate']['critical_threshold']
            
            if on_time_rate >= kpi_target:
                status = 'excellent'
            elif on_time_rate >= warning_threshold:
                status = 'good'
            elif on_time_rate >= critical_threshold:
                status = 'warning'
            else:
                status = 'critical'
            
            kpi_results['on_time_delivery_rate'] = {
                'current_value': round(on_time_rate, 2),
                'target': kpi_target,
                'status': status,
                'variance': round(on_time_rate - kpi_target, 2)
            }
        
        # Delay rate
        if 'logistics_delay' in shipments_df.columns:
            delay_rate = shipments_df['logistics_delay'].mean() * 100
            kpi_target = self.kpis['operational_efficiency']['delay_rate']['target']
            warning_threshold = self.kpis['operational_efficiency']['delay_rate']['warning_threshold']
            critical_threshold = self.kpis['operational_efficiency']['delay_rate']['critical_threshold']
            
            if delay_rate <= kpi_target:
                status = 'excellent'
            elif delay_rate <= warning_threshold:
                status = 'good'
            elif delay_rate <= critical_threshold:
                status = 'warning'
            else:
                status = 'critical'
            
            kpi_results['delay_rate'] = {
                'current_value': round(delay_rate, 2),
                'target': kpi_target,
                'status': status,
                'variance': round(delay_rate - kpi_target, 2)
            }
        
        return kpi_results

    def perform_descriptive_analysis(self, shipments_df: pd.DataFrame) -> Dict[str, Any]:
        """Perform descriptive analysis on shipment data"""
        try:
            if shipments_df.empty:
                return {"error": "No data available for descriptive analysis"}
            
            # Basic statistics
            total_shipments = len(shipments_df)
            delayed_shipments = len(shipments_df[shipments_df['logistics_delay'] == True]) if 'logistics_delay' in shipments_df.columns else 0
            delay_rate = (delayed_shipments / total_shipments * 100) if total_shipments > 0 else 0
            
            # Time range
            if 'timestamp' in shipments_df.columns:
                shipments_df['timestamp'] = pd.to_datetime(shipments_df['timestamp'])
                date_range = {
                    'start': shipments_df['timestamp'].min().strftime('%Y-%m-%d'),
                    'end': shipments_df['timestamp'].max().strftime('%Y-%m-%d')
                }
            else:
                date_range = {'start': 'N/A', 'end': 'N/A'}
            
            # Asset distribution
            asset_distribution = {}
            if 'asset_id' in shipments_df.columns:
                asset_counts = shipments_df['asset_id'].value_counts()
                asset_distribution = asset_counts.to_dict()
            
            # Status distribution
            status_distribution = {}
            if 'shipment_status' in shipments_df.columns:
                status_counts = shipments_df['shipment_status'].value_counts()
                status_distribution = status_counts.to_dict()
            
            # Delay reasons
            delay_reasons = {}
            if 'logistics_delay_reason' in shipments_df.columns:
                reason_counts = shipments_df['logistics_delay_reason'].value_counts()
                delay_reasons = reason_counts.to_dict()
            
            # Performance metrics
            performance_metrics = {
                'total_shipments': total_shipments,
                'delayed_shipments': delayed_shipments,
                'on_time_shipments': total_shipments - delayed_shipments,
                'delay_rate_percent': round(delay_rate, 2),
                'on_time_rate_percent': round(100 - delay_rate, 2),
                'avg_delay_probability': round(shipments_df['delay_probability'].mean(), 3) if 'delay_probability' in shipments_df.columns else None,
                'avg_waiting_time_minutes': round(shipments_df['waiting_time'].mean(), 1) if 'waiting_time' in shipments_df.columns else None,
                'high_risk_shipments': len(shipments_df[shipments_df['delay_probability'] > 0.8]) if 'delay_probability' in shipments_df.columns else 0
            }
            
            # KPI evaluation
            kpi_results = self._evaluate_kpis(shipments_df)
            
            return {
                'data_summary': {
                    'total_records': total_shipments,
                    'date_range': date_range
                },
                'performance_metrics': performance_metrics,
                'distributions': {
                    'asset_distribution': asset_distribution,
                    'status_distribution': status_distribution,
                    'delay_reasons': delay_reasons
                },
                'kpi_evaluation': kpi_results,
                'insights': self._generate_descriptive_insights(shipments_df)
            }
            
        except Exception as e:
            logger.error(f"Error in descriptive analysis: {e}")
            return {"error": str(e)}

    def _generate_descriptive_insights(self, shipments_df: pd.DataFrame) -> Dict[str, Any]:
        """Generate descriptive analytics insights"""
        insights = []
        visualizations = []
        
        if shipments_df.empty:
            return {"insights": [{"type": "error", "message": "No data available"}], "visualizations": []}
        
        # Basic statistics
        total_shipments = len(shipments_df)
        delayed_shipments = len(shipments_df[shipments_df['logistics_delay'] == True]) if 'logistics_delay' in shipments_df.columns else 0
        delay_rate = (delayed_shipments / total_shipments * 100) if total_shipments > 0 else 0
        
        insights.append({
            'type': 'descriptive',
            'title': 'Shipment Overview',
            'metrics': {
                'total_shipments': total_shipments,
                'delayed_shipments': delayed_shipments,
                'delay_rate_percent': round(delay_rate, 2),
                'avg_delay_probability': round(shipments_df['delay_probability'].mean(), 3) if 'delay_probability' in shipments_df.columns else None
            }
        })
        
        # Performance against KPIs
        kpi_status = self._evaluate_kpis(shipments_df)
        if kpi_status:
            insights.append({
                'type': 'kpi_evaluation',
                'title': 'KPI Performance',
                'kpi_results': kpi_status
            })
        
        return {"insights": insights, "visualizations": visualizations}
    
    def get_report_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get analytics report history"""
        recent_reports = self.report_history[-limit:] if self.report_history else []
        return [report.to_dict() for report in recent_reports]
    
    def generate_prescriptive_insights(self, shipments_df: pd.DataFrame) -> Dict[str, Any]:
        """Generate prescriptive insights and recommendations"""
        try:
            if shipments_df.empty:
                return {"error": "No data available for prescriptive analysis"}
            
            # Get current performance metrics
            descriptive_results = self._generate_descriptive_insights(shipments_df)
            performance_metrics = descriptive_results.get('insights', [{}])[0].get('metrics', {}) if descriptive_results.get('insights') else {}
            
            # Root cause analysis for issues
            root_cause_results = self.perform_root_cause_analysis(shipments_df)
            
            # Predictive insights
            predictive_results = self.generate_predictive_insights(shipments_df)
            
            # Generate optimization opportunities
            optimization_opportunities = self._generate_optimization_opportunities(
                performance_metrics, root_cause_results, predictive_results
            )
            
            # Resource optimization suggestions
            resource_optimization = self._generate_resource_optimization(shipments_df)
            
            # KPI improvement plans
            kpi_improvements = self._generate_kpi_improvement_plans(shipments_df)
            
            return {
                "issues": self._identify_issues(performance_metrics, root_cause_results),
                "optimization_opportunities": optimization_opportunities,
                "resource_optimization": resource_optimization,
                "kpi_improvements": kpi_improvements,
                "recommendations": self._generate_prescriptive_recommendations(
                    optimization_opportunities, resource_optimization, kpi_improvements
                )
            }
            
        except Exception as e:
            logger.error(f"Error in prescriptive analysis: {e}")
            return {"error": str(e)}
    
    def _identify_issues(self, performance_metrics: Dict, root_cause_results: Dict) -> List[Dict[str, Any]]:
        """Identify key issues based on performance metrics and root cause analysis"""
        issues = []
        
        # Check delay rate
        delay_rate = performance_metrics.get('delay_rate_percent', 0)
        if delay_rate > 15:
            issues.append({
                "issue": "High Delay Rate",
                "severity": "critical",
                "description": f"Delay rate of {delay_rate}% exceeds acceptable thresholds",
                "impact": "Operational efficiency and customer satisfaction are significantly impacted"
            })
        elif delay_rate > 10:
            issues.append({
                "issue": "Elevated Delay Rate",
                "severity": "warning",
                "description": f"Delay rate of {delay_rate}% requires attention",
                "impact": "Potential negative impact on customer satisfaction"
            })
        
        # Check high-risk shipments
        high_risk_count = performance_metrics.get('high_risk_shipments', 0)
        if high_risk_count > 10:
            issues.append({
                "issue": "High Number of High-Risk Shipments",
                "severity": "warning",
                "description": f"{high_risk_count} shipments have >80% delay probability",
                "impact": "Increased operational risk and potential customer complaints"
            })
        
        # Add issues from root cause analysis
        if 'insights' in root_cause_results:
            for insight in root_cause_results['insights']:
                if insight.get('type') in ['critical', 'warning']:
                    issues.append({
                        "issue": insight.get('title', 'Unknown Issue'),
                        "severity": insight.get('type', 'info'),
                        "description": insight.get('description', ''),
                        "impact": insight.get('impact', 'Unknown')
                    })
        
        return issues
    
    def _generate_optimization_opportunities(self, performance_metrics: Dict, root_cause_results: Dict, predictive_results: Dict) -> List[Dict[str, Any]]:
        """Generate optimization opportunities"""
        opportunities = []
        
        # Based on delay analysis
        delay_rate = performance_metrics.get('delay_rate_percent', 0)
        if delay_rate > 10:
            opportunities.append({
                "category": "Delay Reduction",
                "potential_improvement": f"Reduce delay rate from {delay_rate}% to <10%",
                "implementation_effort": "medium",
                "expected_benefit": "Improved customer satisfaction and reduced costs",
                "recommendations": [
                    "Implement predictive delay alerts",
                    "Optimize routing based on traffic patterns",
                    "Improve asset maintenance schedules"
                ]
            })
        
        # Based on high-risk shipments
        high_risk_count = performance_metrics.get('high_risk_shipments', 0)
        if high_risk_count > 5:
            opportunities.append({
                "category": "Risk Mitigation",
                "potential_improvement": f"Reduce high-risk shipments from {high_risk_count} to <5",
                "implementation_effort": "low",
                "expected_benefit": "Proactive issue resolution and improved reliability",
                "recommendations": [
                    "Enhance monitoring of high-risk shipments",
                    "Implement early intervention protocols",
                    "Review asset allocation strategies"
                ]
            })
        
        # Based on asset performance (if available)
        if 'asset_analysis' in root_cause_results:
            underperforming_assets = [asset for asset in root_cause_results['asset_analysis'] if asset['delay_rate'] > 20]
            if underperforming_assets:
                opportunities.append({
                    "category": "Asset Performance Optimization",
                    "potential_improvement": f"Improve performance of {len(underperforming_assets)} underperforming assets",
                    "implementation_effort": "high",
                    "expected_benefit": "Better asset utilization and reduced delays",
                    "recommendations": [
                    "Conduct maintenance reviews for underperforming assets",
                    "Provide additional training for asset operators",
                    "Consider asset replacement for consistently underperforming units"
                ]
            })
        
        return opportunities
    
    def _generate_resource_optimization(self, shipments_df: pd.DataFrame) -> Dict[str, Any]:
        """Generate resource optimization suggestions"""
        if shipments_df.empty or 'asset_id' not in shipments_df.columns:
            return {}
        
        # Asset utilization analysis
        asset_utilization = {}
        for asset_id in shipments_df['asset_id'].unique():
            asset_data = shipments_df[shipments_df['asset_id'] == asset_id]
            utilization_rate = len(asset_data) / len(shipments_df) * 100
            asset_utilization[asset_id] = {
                "utilization_rate": round(utilization_rate, 2),
                "delay_rate": round(asset_data['logistics_delay'].mean() * 100, 2) if 'logistics_delay' in asset_data.columns else 0
            }
        
        # Identify underutilized and overutilized assets
        avg_utilization = sum(item['utilization_rate'] for item in asset_utilization.values()) / len(asset_utilization)
        underutilized = [asset for asset, data in asset_utilization.items() if data['utilization_rate'] < avg_utilization * 0.7]
        overutilized = [asset for asset, data in asset_utilization.items() if data['utilization_rate'] > avg_utilization * 1.3]
        
        return {
            "asset_utilization": asset_utilization,
            "underutilized_assets": underutilized,
            "overutilized_assets": overutilized,
            "recommendations": [
                f"Redistribute workload from overutilized assets ({', '.join(overutilized)}) to underutilized assets ({', '.join(underutilized)})" if underutilized and overutilized else "",
                "Review asset allocation strategy to balance utilization",
                "Consider adding capacity if consistently overutilized"
            ]
        }
    
    def _generate_kpi_improvement_plans(self, shipments_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate KPI improvement plans"""
        kpi_results = self._evaluate_kpis(shipments_df)
        improvement_plans = []
        
        for kpi_name, kpi_data in kpi_results.items():
            current_value = kpi_data.get('current_value', 0)
            target = kpi_data.get('target', 0)
            status = kpi_data.get('status', 'unknown')
            
            # Only create plans for underperforming KPIs
            if status in ['warning', 'critical']:
                gap = target - current_value if kpi_name != 'delay_rate' else current_value - target
                
                improvement_plans.append({
                    "kpi": kpi_name.replace('_', ' ').title(),
                    "current": current_value,
                    "target": target,
                    "gap": round(gap, 2),
                    "actions": self._get_kpi_improvement_actions(kpi_name, current_value, target)
                })
        
        return improvement_plans
    
    def _get_kpi_improvement_actions(self, kpi_name: str, current: float, target: float) -> List[str]:
        """Get specific actions for KPI improvement"""
        actions = []
        
        if kpi_name == 'on_time_delivery_rate':
            if current < target:
                actions = [
                    "Implement real-time delay prediction and alerts",
                    "Optimize routing algorithms",
                    "Enhance asset maintenance schedules",
                    "Improve coordination with port authorities"
                ]
        elif kpi_name == 'delay_rate':
            if current > target:
                actions = [
                    "Identify and address root causes of delays",
                    "Implement preventive maintenance programs",
                    "Optimize scheduling and resource allocation",
                    "Enhance communication with stakeholders"
                ]
        elif kpi_name == 'customer_satisfaction':
            if current < target:
                actions = [
                    "Improve communication with customers",
                    "Reduce delivery delays",
                    "Enhance tracking and visibility",
                    "Implement proactive issue resolution"
                ]
        
        return actions
    
    def _generate_prescriptive_recommendations(self, optimization_opportunities: List, resource_optimization: Dict, kpi_improvements: List) -> List[Dict[str, Any]]:
        """Generate overall prescriptive recommendations"""
        recommendations = []
        
        # Add recommendations from optimization opportunities
        for opportunity in optimization_opportunities:
            recommendations.append({
                "category": opportunity["category"],
                "priority": "high" if opportunity["implementation_effort"] == "low" else "medium",
                "description": opportunity["potential_improvement"],
                "actions": opportunity["recommendations"]
            })
        
        # Add resource optimization recommendations
        if 'recommendations' in resource_optimization:
            for rec in resource_optimization['recommendations']:
                if rec:  # Only add non-empty recommendations
                    recommendations.append({
                        "category": "Resource Optimization",
                        "priority": "medium",
                        "description": rec,
                        "actions": ["Implement resource redistribution plan", "Monitor utilization metrics"]
                    })
        
        # Add KPI improvement recommendations
        for kpi_improvement in kpi_improvements:
            recommendations.append({
                "category": f"KPI Improvement: {kpi_improvement['kpi']}",
                "priority": "high",
                "description": f"Close gap of {kpi_improvement['gap']} to reach target",
                "actions": kpi_improvement['actions']
            })
        
        return recommendations

    def export_insights_to_json(self, insights: List[Dict[str, Any]], filename: str) -> str:
        """Export insights to JSON file"""
        try:
            export_data = {
                'generated_at': datetime.now().isoformat(),
                'total_insights': len(insights),
                'insights': insights
            }
            
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            logger.info(f"Insights exported to {filename}")
            return filename
        except Exception as e:
            logger.error(f"Error exporting insights: {e}")
            return str(e)


# Global analytics engine instance
_analytics_engine = None

def get_analytics_engine() -> AdvancedAnalyticsEngine:
    """Get or create the analytics engine singleton"""
    global _analytics_engine
    if _analytics_engine is None:
        _analytics_engine = AdvancedAnalyticsEngine()
    return _analytics_engine


if __name__ == "__main__":
    # Example usage
    import pandas as pd
    
    # Create sample data for testing
    sample_data = {
        'id': range(1, 101),
        'timestamp': pd.date_range(start='2024-01-01', periods=100, freq='H'),
        'logistics_delay': np.random.choice([True, False], 100, p=[0.3, 0.7]),
        'delay_probability': np.random.beta(2, 5, 100),
        'waiting_time': np.random.exponential(30, 100),
        'asset_id': np.random.choice(['Truck_1', 'Truck_2', 'Truck_3'], 100)
    }
    
    df = pd.DataFrame(sample_data)
    
    # Test analytics engine
    engine = get_analytics_engine()
    
    # Generate trend analysis
    trend_results = engine.analyze_shipment_trends(df)
    print("Trend Analysis Results:")
    print(json.dumps(trend_results, indent=2, default=str))
    
    # Generate root cause analysis
    root_cause_results = engine.perform_root_cause_analysis(df)
    print("\nRoot Cause Analysis Results:")
    print(json.dumps(root_cause_results, indent=2, default=str))
    
    # Generate comprehensive report
    report = engine.generate_comprehensive_report(df)
    print(f"\nGenerated report: {report.title}")
    print(f"Insights: {len(report.insights)}")
    print(f"Recommendations: {len(report.recommendations)}")
