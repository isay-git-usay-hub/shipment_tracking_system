"""
Advanced Streamlit Dashboard for Maersk Shipment AI System
With integrated Advanced Analytics & Reporting capabilities
"""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from datetime import datetime, timedelta
import json
import base64
from io import BytesIO
from PIL import Image

# Configure page
st.set_page_config(
    page_title="🚢 Maersk AI Analytics Platform",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
API_BASE_URL = "http://localhost:8080"

def get_api_data(endpoint: str) -> dict:
    """Get data from API endpoint"""
    try:
        response = requests.get(f"{API_BASE_URL}{endpoint}")
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code}")
            return {}
    except requests.exceptions.RequestException as e:
        st.error(f"Connection Error: {e}")
        return {}

def post_api_data(endpoint: str, data: dict = None) -> dict:
    """Post data to API endpoint"""
    try:
        response = requests.post(f"{API_BASE_URL}{endpoint}", json=data or {})
        if response.status_code == 200:
            return response.json()
        else:
            msg = f"API Error: {response.status_code} - {response.text}"
            st.error(msg)
            return {"success": False, "error": msg}
    except requests.exceptions.RequestException as e:
        msg = f"Connection Error: {e}"
        st.error(msg)
        return {"success": False, "error": msg}

def check_analytics_status():
    """Check if analytics service is available"""
    try:
        response = requests.get(f"{API_BASE_URL}/analytics/status")
        return response.status_code == 200 and response.json().get('analytics_available', False)
    except:
        return False

def get_analytics_status():
    """Get analytics system status"""
    try:
        response = requests.get(f"{API_BASE_URL}/analytics/status")
        if response.status_code == 200:
            return response.json()
        return {}
    except:
        return {}

def display_base64_image(base64_string: str, caption: str = ""):
    """Display base64 encoded image"""
    if base64_string and isinstance(base64_string, str) and base64_string.startswith('data:image'):
        # Extract base64 data
        image_data = base64_string.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes))
        st.image(image, caption=caption, use_column_width=True)
    else:
        st.warning("No visualization available")


def display_plotly_json(fig_json: str, caption: str = ""):
    """Render a Plotly figure from JSON in Streamlit"""
    try:
        if fig_json:
            fig = pio.from_json(fig_json)
            if caption:
                st.caption(caption)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No visualization available")
    except Exception as e:
        st.warning(f"Visualization unavailable: {e}")

def load_sample_data():
    """Load sample data into database"""
    try:
        response = requests.post(f"{API_BASE_URL}/data/load?replace_existing=false")
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error loading data: {response.status_code}")
            return {}
    except requests.exceptions.RequestException as e:
        st.error(f"Connection Error: {e}")
        return {}

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(45deg, #1f77b4, #ff7f0e);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .analytics-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .report-container {
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        background-color: #fafafa;
    }
</style>
""", unsafe_allow_html=True)

# App header
st.markdown('<div class="analytics-header"><h1>🚢 Maersk AI Analytics Platform</h1><p>Advanced Business Intelligence & Predictive Analytics</p></div>', unsafe_allow_html=True)

# Sidebar
st.sidebar.header("🎛️ Analytics Control Panel")

# Check API health
health_data = get_api_data("/health")
if health_data:
    st.sidebar.success(f"✅ API Status: {health_data.get('status', 'Unknown')}")
else:
    st.sidebar.error("❌ API Unavailable")
    st.error("🚨 **API Connection Failed**\n\nPlease start the API server using:\n```bash\nuvicorn api.main:app --reload --host localhost --port 8080\n```")
    st.stop()

# Check Analytics service status
analytics_available = check_analytics_status()
if analytics_available:
    st.sidebar.success("🧠 Advanced Analytics: Active")
    analytics_status = get_analytics_status()
    if analytics_status.get('features'):
        features = analytics_status['features']
        st.sidebar.info(f"📊 Features: {len([k for k, v in features.items() if v])} active")
else:
    st.sidebar.warning("⚠️ Advanced Analytics: Not Available")

# Data management
st.sidebar.subheader("📊 Data Management")
data_validation = get_api_data("/data/validate")
if data_validation and data_validation.get("total_records", 0) == 0:
    st.sidebar.warning("⚠️ No data in database")
    if st.sidebar.button("🔄 Load Sample Data", key="sidebar_load_data"):
        with st.spinner("Loading sample data..."):
            result = load_sample_data()
            if result:
                st.sidebar.success(f"✅ Loaded {result.get('records_loaded', 0)} records")
                st.rerun()
else:
    records_count = data_validation.get("total_records", 0)
    st.sidebar.info(f"📈 Database: {records_count:,} records")

# Analytics Configuration
st.sidebar.subheader("🔧 Analytics Configuration")
asset_filter = st.sidebar.selectbox("🚛 Asset Filter", options=[None, "Truck_1", "Truck_2", "Truck_3"], index=0)

# Align date picker with database coverage when available
db_start_dt = None
db_end_dt = None
try:
    dr = (data_validation or {}).get("date_range") or {}
    if isinstance(dr.get("start"), str):
        db_start_dt = datetime.fromisoformat(dr["start"])  # type: ignore[arg-type]
    if isinstance(dr.get("end"), str):
        db_end_dt = datetime.fromisoformat(dr["end"])  # type: ignore[arg-type]
except Exception:
    pass

if db_start_dt and db_end_dt:
    # Default to the exact available DB window to avoid empty results
    default_start = db_start_dt
    default_end = db_end_dt
    # Initialize session default only once
    if 'selected_date_range' not in st.session_state:
        st.session_state.selected_date_range = (default_start.date(), default_end.date())
    date_range = st.sidebar.date_input(
        "📅 Date Range",
        value=st.session_state.selected_date_range,
        min_value=db_start_dt.date(),
        max_value=db_end_dt.date(),
        key='selected_date_range'
    )
else:
    # Fallback to last 30 days from now
    date_range = st.sidebar.date_input(
        "📅 Date Range",
        value=(datetime.now() - timedelta(days=30), datetime.now()),
        max_value=datetime.now()
    )

# Analytics Type Selection
analysis_types = st.sidebar.multiselect(
    "📊 Analysis Types",
    options=["descriptive", "diagnostic", "predictive", "prescriptive"],
    default=["descriptive", "diagnostic"]
)

# Refresh button
if st.sidebar.button("🔄 Refresh Analytics"):
    st.rerun()

# Main dashboard
if data_validation and data_validation.get("total_records", 0) > 0:
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Executive Dashboard", 
        "🔍 Diagnostic Analysis", 
        "🔮 Predictive Insights", 
        "💡 Recommendations", 
        "📋 Comprehensive Reports"
    ])
    
    # Prepare request data
    request_data = {
        "asset_id": asset_filter,
        "start_date": date_range[0].isoformat() if len(date_range) == 2 else None,
        "end_date": date_range[1].isoformat() if len(date_range) == 2 else None
    }
    
    with tab1:
        st.header("🎯 Executive Summary Dashboard")
        
        if analytics_available:
            # Generate executive report
            with st.spinner("Generating executive summary..."):
                exec_report = post_api_data("/reports/executive-summary", request_data)
            
            if exec_report.get('success') and 'report' in exec_report:
                report = exec_report['report']
                
                # Key Metrics Row
                st.subheader("📈 Key Performance Indicators")
                overview = report.get('overview', {})
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric(
                        "📦 Total Shipments",
                        overview.get('total_shipments', 0),
                        delta=None
                    )
                
                with col2:
                    delay_rate = overview.get('delay_rate', 0)
                    st.metric(
                        "⚠️ Delay Rate",
                        f"{delay_rate:.1f}%",
                        delta=f"{delay_rate - 15:.1f}%" if delay_rate else None,
                        delta_color="inverse"
                    )
                
                with col3:
                    on_time_rate = overview.get('on_time_rate', 0)
                    st.metric(
                        "✅ On-Time Rate",
                        f"{on_time_rate:.1f}%",
                        delta=f"{on_time_rate - 85:.1f}%" if on_time_rate else None
                    )
                
                with col4:
                    high_risk = overview.get('high_risk_shipments', 0)
                    st.metric(
                        "🚨 High Risk",
                        high_risk,
                        delta=None
                    )
                
                # Visualizations
                st.subheader("📊 Performance Visualizations")
                visualizations = report.get('visualizations', {})
                
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("KPI Performance Dashboard")
                    if visualizations.get('kpi_summary'):
                        display_base64_image(visualizations['kpi_summary'])
                    elif visualizations.get('kpi_summary_json'):
                        display_plotly_json(visualizations['kpi_summary_json'])
                    else:
                        st.warning("No visualization available")
                    
                with col2:
                    st.subheader("Shipment Status Distribution")
                    if visualizations.get('status_distribution'):
                        display_base64_image(visualizations['status_distribution'])
                    elif visualizations.get('status_distribution_json'):
                        display_plotly_json(visualizations['status_distribution_json'])
                    else:
                        st.warning("No visualization available")
                
                # Trend Analysis
                if 'delay_trend' in visualizations or 'delay_trend_json' in visualizations:
                    st.subheader("📈 Delay Rate Trends")
                    if visualizations.get('delay_trend'):
                        display_base64_image(visualizations['delay_trend'])
                    elif visualizations.get('delay_trend_json'):
                        display_plotly_json(visualizations['delay_trend_json'])
                    else:
                        st.warning("No visualization available")
                
                # Key Insights
                insights = report.get('insights', [])
                if insights:
                    st.subheader("💡 Key Insights")
                    for i, insight in enumerate(insights[:5], 1):
                        st.info(f"{i}. {insight}")
                
                # Recommendations
                recommendations = report.get('recommendations', [])
                if recommendations:
                    st.subheader("🎯 Strategic Recommendations")
                    for i, rec in enumerate(recommendations[:3], 1):
                        st.success(f"{i}. {rec}")
            
            else:
                err_msg = exec_report.get('error') if isinstance(exec_report, dict) else None
                if err_msg:
                    st.error(f"Failed to generate executive report: {err_msg}")
                    if isinstance(err_msg, str) and 'No data found for specified criteria' in err_msg:
                        dr = (data_validation or {}).get('date_range') or {}
                        tip_start = dr.get('start', 'unknown')
                        tip_end = dr.get('end', 'unknown')
                        st.info(f"Tip: Adjust the Date Range to the data window: {tip_start} to {tip_end}.")
                else:
                    st.error("Failed to generate executive report")
        
        else:
            # Fallback to basic analytics
            st.info("Advanced analytics not available. Showing basic overview.")
            analytics = get_api_data("/analytics/overview")
            
            if analytics:
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("📦 Total Shipments", analytics.get("total_shipments", 0))
                
                with col2:
                    st.metric("⚠️ Delayed Shipments", analytics.get("delayed_shipments", 0))
                
                with col3:
                    st.metric("📈 Delay Rate", f"{analytics.get('delay_rate', 0):.1f}%")
                
                with col4:
                    st.metric("⏱️ Avg Wait Time", f"{analytics.get('avg_waiting_time_minutes', 0):.0f} min")
    
    with tab2:
        st.header("🔍 Diagnostic Analysis & Root Cause Analysis")
        
        if analytics_available:
            # Generate diagnostic analysis
            with st.spinner("Performing root cause analysis..."):
                diagnostic = post_api_data("/analytics/diagnostic", request_data)
            
            if diagnostic.get('success') and 'results' in diagnostic:
                results = diagnostic['results']
                
                # Summary metrics
                st.subheader("📊 Analysis Summary")
                summary = diagnostic.get('data_summary', {})
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("📦 Total Records", summary.get('total_records', 0))
                with col2:
                    st.metric("⚠️ Delayed Records", summary.get('delayed_records', 0))
                with col3:
                    delay_pct = (summary.get('delayed_records', 0) / max(summary.get('total_records', 1), 1)) * 100
                    st.metric("📈 Delay Percentage", f"{delay_pct:.1f}%")
                
                # Delay Reasons Analysis
                if 'delay_reasons' in results:
                    st.subheader("🚨 Top Delay Reasons")
                    delay_reasons = results['delay_reasons']
                    
                    # Use percentages sub-dictionary for charting
                    percentages = delay_reasons.get('percentages', {}) if isinstance(delay_reasons, dict) else {}
                    if percentages:
                        reasons_df = pd.DataFrame(list(percentages.items()), columns=['Reason', 'Percentage']).sort_values('Percentage')
                        fig = px.bar(reasons_df, x='Percentage', y='Reason', orientation='h', title="Delay Reasons by Percentage")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No delay reasons available for the selected period.")
                
                # Asset Issues
                if 'asset_issues' in results:
                    asset_issues = results['asset_issues']
                    if 'high_risk_assets' in asset_issues:
                        st.subheader("🚛 High Risk Assets")
                        for asset in asset_issues['high_risk_assets']:
                            st.warning(f"⚠️ {asset}")
                
                # Temporal Patterns
                if 'temporal_patterns' in results:
                    temporal = results['temporal_patterns']
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if 'peak_delay_hours' in temporal:
                            st.subheader("🕐 Peak Delay Hours")
                            hours = temporal['peak_delay_hours']
                            st.info(f"High delay hours: {', '.join(map(str, hours))}")
                    
                    with col2:
                        if 'problematic_days' in temporal:
                            st.subheader("📅 Problematic Days")
                            days = temporal['problematic_days']
                            st.warning(f"Problem days: {', '.join(days)}")
                
                # Recommendations
                if 'recommendations' in results:
                    st.subheader("💡 Diagnostic Recommendations")
                    for i, rec in enumerate(results['recommendations'], 1):
                        issue = rec.get('issue', 'Unknown issue')
                        recommendation = rec.get('recommendation', 'No recommendation')
                        impact = rec.get('impact', 'Unknown impact')
                        
                        st.info(f"**{i}. {issue}**\n\n"
                               f"💡 Recommendation: {recommendation}\n\n"
                               f"📈 Expected Impact: {impact}")
            
            else:
                err_msg = diagnostic.get('error') if isinstance(diagnostic, dict) else None
                if err_msg:
                    st.error(f"Failed to generate diagnostic analysis: {err_msg}")
                else:
                    st.error("Failed to generate diagnostic analysis")
        
        else:
            st.info("Advanced diagnostic analysis requires analytics service")
    
    with tab3:
        st.header("🔮 Predictive Insights & Forecasting")
        
        if analytics_available:
            # Generate predictive analysis
            with st.spinner("Generating predictive insights..."):
                predictive = post_api_data("/analytics/predictive", request_data)
            
            if predictive.get('success') and 'results' in predictive:
                results = predictive['results']
                
                # Forecast Summary
                st.subheader("📈 Forecast Summary")
                forecast_days = predictive.get('forecast_days', 7)
                st.info(f"🔮 Forecast Period: {forecast_days} days")
                
                # Forecast Visualization
                if 'forecast' in results and 'forecasts' in results['forecast']:
                    forecasts = results['forecast']['forecasts']
                    if forecasts:
                        forecast_df = pd.DataFrame(forecasts)
                        
                        # Volume forecast
                        if 'predicted_volume' in forecast_df.columns:
                            st.subheader("📦 Volume Forecast")
                            fig = px.line(forecast_df, x='date', y='predicted_volume',
                                        title="Predicted Shipment Volume")
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Delay rate forecast
                        if 'predicted_delay_rate' in forecast_df.columns:
                            st.subheader("⚠️ Delay Rate Forecast")
                            fig = px.line(forecast_df, x='date', y='predicted_delay_rate',
                                        title="Predicted Delay Rate %")
                            st.plotly_chart(fig, use_container_width=True)
                
                # Risk Assessment
                if 'risk_assessment' in results:
                    risk = results['risk_assessment']
                    st.subheader("🚨 Risk Assessment")
                    
                    overall_risk = risk.get('overall_risk_level', 'unknown')
                    if overall_risk == 'high':
                        st.error(f"⚠️ Overall Risk Level: {overall_risk.upper()}")
                    elif overall_risk == 'medium':
                        st.warning(f"⚠️ Overall Risk Level: {overall_risk.upper()}")
                    else:
                        st.success(f"✅ Overall Risk Level: {overall_risk.upper()}")
                    
                    if 'risk_factors' in risk:
                        st.subheader("🎯 Risk Factors")
                        for factor in risk['risk_factors']:
                            factor_name = factor.get('factor', 'Unknown')
                            level = factor.get('level', 'unknown')
                            
                            if level == 'high':
                                st.error(f"🔴 {factor_name}: {level.upper()}")
                            elif level == 'medium':
                                st.warning(f"🟡 {factor_name}: {level.upper()}")
                            else:
                                st.success(f"🟢 {factor_name}: {level.upper()}")
                
                # Capacity Analysis
                if 'capacity_analysis' in results:
                    capacity = results['capacity_analysis']
                    st.subheader("⚙️ Capacity Analysis")
                    
                    # Backend returns utilization_rate inside capacity_metrics
                    utilization = capacity.get('capacity_metrics', {}).get('utilization_rate', 0)
                    st.metric("📊 Utilization Rate", f"{utilization:.1f}%")
                    
                    # Optional details
                    cm = capacity.get('capacity_metrics', {})
                    if cm:
                        st.caption(f"Max Daily Volume: {cm.get('max_daily_volume', 0)} | Avg Daily Volume: {cm.get('avg_daily_volume', 0)}")
                    
                    if 'bottleneck_assets' in capacity:
                        bottlenecks = capacity['bottleneck_assets']
                        if bottlenecks:
                            st.warning(f"🚨 Bottleneck Assets: {', '.join(bottlenecks)}")
            
            else:
                err_msg = predictive.get('error') if isinstance(predictive, dict) else None
                if err_msg:
                    st.error(f"Failed to generate predictive analysis: {err_msg}")
                else:
                    st.error("Failed to generate predictive analysis")
        
        else:
            st.info("Predictive insights require advanced analytics service")
    
    with tab4:
        st.header("💡 Prescriptive Recommendations")
        
        if analytics_available:
            # Generate prescriptive analysis
            with st.spinner("Generating optimization recommendations..."):
                prescriptive = post_api_data("/analytics/prescriptive", request_data)
            
            if prescriptive.get('success') and 'results' in prescriptive:
                results = prescriptive['results']
                
                # Summary
                st.subheader("📊 Optimization Overview")
                summary = prescriptive.get('data_summary', {})
                issues_count = summary.get('issues_identified', 0)
                
                if issues_count > 0:
                    st.warning(f"🔍 {issues_count} optimization opportunities identified")
                else:
                    st.success("✅ No major optimization issues found")
                
                # Optimization Opportunities
                if 'optimization_opportunities' in results:
                    st.subheader("🎯 Optimization Opportunities")
                    
                    for i, opp in enumerate(results['optimization_opportunities'], 1):
                        category = opp.get('category', 'General')
                        improvement = opp.get('potential_improvement', 'Unknown')
                        effort = opp.get('implementation_effort', 'unknown')
                        recommendations = opp.get('recommendations', [])
                        
                        # Color code by effort level
                        if effort.lower() == 'low':
                            st.success(f"**{i}. {category}** (Low Effort)")
                        elif effort.lower() == 'medium':
                            st.warning(f"**{i}. {category}** (Medium Effort)")
                        else:
                            st.info(f"**{i}. {category}** (High Effort)")
                        
                        st.write(f"💫 **Potential Improvement**: {improvement}")
                        
                        if recommendations:
                            st.write("**Action Items:**")
                            for rec in recommendations:
                                st.write(f"• {rec}")
                        
                        st.markdown("---")
                
                # Resource Optimization
                if 'resource_optimization' in results:
                    resource = results['resource_optimization']
                    st.subheader("⚙️ Resource Optimization")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if 'underutilized_assets' in resource:
                            underutilized = resource['underutilized_assets']
                            if underutilized:
                                st.info("📉 **Underutilized Assets**")
                                for asset in underutilized:
                                    st.write(f"• {asset}")
                    
                    with col2:
                        if 'overutilized_assets' in resource:
                            overutilized = resource['overutilized_assets']
                            if overutilized:
                                st.warning("📈 **Overutilized Assets**")
                                for asset in overutilized:
                                    st.write(f"• {asset}")
                
                # KPI Improvements
                if 'kpi_improvements' in results:
                    st.subheader("📈 KPI Improvement Plan")
                    
                    for kpi in results['kpi_improvements']:
                        kpi_name = kpi.get('kpi', 'Unknown KPI')
                        current = kpi.get('current', 0)
                        target = kpi.get('target', 0)
                        actions = kpi.get('actions', [])
                        
                        gap = target - current
                        
                        st.metric(
                            f"🎯 {kpi_name}",
                            f"{current:.1f}",
                            delta=f"Target: {target:.1f} (Gap: {gap:.1f})"
                        )
                        
                        if actions:
                            st.write("**Recommended Actions:**")
                            for action in actions:
                                st.write(f"✅ {action}")
                        
                        st.markdown("---")
            
            else:
                err_msg = prescriptive.get('error') if isinstance(prescriptive, dict) else None
                if err_msg:
                    st.error(f"Failed to generate prescriptive recommendations: {err_msg}")
                else:
                    st.error("Failed to generate prescriptive recommendations")
        
        else:
            st.info("Prescriptive recommendations require advanced analytics service")
    
    with tab5:
        st.header("📋 Comprehensive Analytics Reports")
        
        if analytics_available:
            st.subheader("📊 Report Generation")
            
            # Report type selection
            report_types = {
                "Executive Summary": "executive-summary",
                "Operational Dashboard": "operational-dashboard", 
                "Predictive Forecast": "predictive-forecast",
                "KPI Scorecard": "kpi-scorecard"
            }
            
            selected_report = st.selectbox("📋 Select Report Type", list(report_types.keys()))
            
            # Additional parameters based on report type
            params = {}
            if selected_report == "Executive Summary":
                params['period_days'] = st.slider("Period (Days)", 7, 90, 30)
            elif selected_report == "Predictive Forecast":
                params['forecast_days'] = st.slider("Forecast Days", 7, 30, 14)
            
            # Combine with base request data
            report_request = {**request_data, **params}
            
            if st.button(f"🚀 Generate {selected_report} Report"):
                with st.spinner(f"Generating {selected_report.lower()} report..."):
                    report_endpoint = f"/reports/{report_types[selected_report]}"
                    report = post_api_data(report_endpoint, report_request)
                
                if report.get('success') and 'report' in report:
                    st.success(f"✅ {selected_report} report generated successfully!")
                    
                    # Display report metadata
                    report_data = report['report']
                    st.json({
                        "Report Type": report_data.get('report_type'),
                        "Generated At": report_data.get('generated_at'),
                        "Data Summary": report_data.get('data_summary')
                    })
                    
                    # Export options
                    st.subheader("💾 Export Options")
                    
                    # JSON Export
                    json_str = json.dumps(report, indent=2)
                    st.download_button(
                        label="📄 Download JSON Report",
                        data=json_str,
                        file_name=f"{selected_report.lower()}_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
                    
                    # Display visualizations if available
                    visualizations = report_data.get('visualizations', {})
                    if visualizations:
                        st.subheader("📊 Report Visualizations")
                        for chart_name, chart_data in visualizations.items():
                            if chart_data:
                                st.subheader(f"{chart_name.replace('_', ' ').title()}")
                                if isinstance(chart_data, str) and chart_data.startswith('data:image'):
                                    display_base64_image(chart_data)
                                elif isinstance(chart_data, str) and chart_data.strip().startswith('{'):
                                    display_plotly_json(chart_data)
                                else:
                                    st.write(chart_data)
                
                else:
                    err_msg = report.get('error') if isinstance(report, dict) else None
                    if err_msg:
                        st.error(f"Failed to generate {selected_report} report: {err_msg}")
                    else:
                        st.error(f"Failed to generate {selected_report} report")
            
            # Comprehensive Analytics Report
            st.subheader("🎯 Comprehensive Analytics Report")
            st.write("Generate a complete report with all analysis types")
            
            comprehensive_types = st.multiselect(
                "Select Analysis Types",
                ["descriptive", "diagnostic", "predictive", "prescriptive"],
                default=["descriptive", "diagnostic", "predictive"]
            )
            
            if st.button("🚀 Generate Comprehensive Report"):
                if comprehensive_types:
                    with st.spinner("Generating comprehensive analytics report..."):
                        comp_request = {**request_data, "analysis_types": comprehensive_types}
                        comp_report = post_api_data("/analytics/comprehensive-report", comp_request)
                    
                    if comp_report.get('success'):
                        st.success("✅ Comprehensive report generated!")
                        st.json(comp_report)
                        
                        # Export comprehensive report
                        comp_json = json.dumps(comp_report, indent=2)
                        st.download_button(
                            label="📄 Download Comprehensive Report",
                            data=comp_json,
                            file_name=f"comprehensive_analytics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )
                    else:
                        st.error("Failed to generate comprehensive report")
                else:
                    st.warning("Please select at least one analysis type")
        
        else:
            st.info("Comprehensive reports require advanced analytics service")

else:
    st.warning("⚠️ **No Data Available**")
    st.info("Please load some data first using the sidebar controls.")
    
    if st.button("🔄 Load Sample Data", key="main_load_data"):
        with st.spinner("Loading sample data..."):
            result = load_sample_data()
            if result:
                st.success(f"✅ Loaded {result.get('records_loaded', 0)} records")
                st.rerun()

# Footer
st.markdown("---")
st.markdown("**🚢 Maersk Shipment AI System** | Advanced Analytics & Business Intelligence Platform")

# System Status Footer
if analytics_available:
    st.success("🧠 Advanced Analytics System: Active & Ready")
else:
    st.warning("⚠️ Advanced Analytics System: Not Available")
