"""
Advanced Analytics & Reporting API Router for Maersk Shipment AI System

This module provides REST API endpoints for advanced analytics and reporting functionality.
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional
import logging
import pandas as pd
from datetime import datetime, timedelta
from pydantic import BaseModel

# Import core dependencies
from core.database import get_db
from core.schemas import ShipmentFilter
from services.shipment_service import shipment_service

# Logger
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Import analytics services
try:
    import sys
    import os
    analytics_path = os.path.join(os.path.dirname(__file__), '..', '..', 'analytics')
    sys.path.insert(0, analytics_path)
    from analytics_engine import get_analytics_engine, AnalysisType, TimeGranularity
    from reporting_service import get_reporting_service, ReportType
    ANALYTICS_AVAILABLE = True
    logger.info("✅ Advanced Analytics & Reporting system available")
except ImportError as e:
    logger.warning(f"Analytics system not available: {e}")
    ANALYTICS_AVAILABLE = False
except Exception as e:
    logger.error(f"Error importing analytics system: {e}")
    ANALYTICS_AVAILABLE = False


# Pydantic models for requests
class AnalyticsRequest(BaseModel):
    asset_id: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    analysis_types: Optional[List[str]] = None
    

class ReportRequest(BaseModel):
    asset_id: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    period_days: Optional[int] = 30
    forecast_days: Optional[int] = 14
    

class ReportExportRequest(BaseModel):
    report_data: dict
    filename: Optional[str] = None
    format: str = "json"


def _sanitize_for_json(obj):
    """Convert numpy/pandas types to JSON-serializable Python types"""
    try:
        import numpy as np
    except Exception:
        np = None  # type: ignore
    
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    if np is not None and isinstance(obj, (np.bool_, getattr(np, 'bool8', bool))):  # numpy bools
        return bool(obj)
    if np is not None and isinstance(obj, (np.integer, getattr(np, 'int64', int))):
        return int(obj)
    if np is not None and isinstance(obj, (np.floating, getattr(np, 'float64', float))):
        return float(obj)
    if np is not None and hasattr(obj, 'tolist'):
        try:
            return obj.tolist()
        except Exception:
            pass
    if hasattr(obj, 'isoformat'):
        try:
            return obj.isoformat()
        except Exception:
            pass
    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass
    return obj

def apply_analytics_filters(db: Session, asset_id: str = None, start_date: str = None, end_date: str = None) -> List[dict]:
    """Apply filters and return shipment data for analytics"""
    try:
        # Parse incoming dates to datetime for server-side filtering
        start_dt = None
        end_dt = None
        if start_date:
            from dateutil.parser import parse
            start_dt = parse(start_date)
        if end_date:
            from dateutil.parser import parse
            end_dt = parse(end_date)
            # Include the entire end date if only a date (no time) was provided
            if isinstance(end_date, str) and ('T' not in end_date and ' ' not in end_date):
                end_dt = end_dt + timedelta(days=1) - timedelta(microseconds=1)

        # Get shipments from database with filters applied
        filters = ShipmentFilter(
            asset_id=asset_id,
            start_date=start_dt,
            end_date=end_dt,
            limit=10000  # Large limit for analytics
        )
        shipments = shipment_service.get_shipments(db, filters, skip=0, limit=10000)
        
        # Convert to dict format for analytics
        data = []
        for s in shipments:
            shipment_dict = {
                'id': s.id,
                'timestamp': s.timestamp,
                'asset_id': s.asset_id,
                'shipment_status': s.shipment_status.value if getattr(s, 'shipment_status', None) else None,
                'logistics_delay': bool(getattr(s, 'logistics_delay', False)),
                'delay_probability': getattr(s, 'delay_probability', None),
                'waiting_time': getattr(s, 'waiting_time', None),
                'latitude': getattr(s, 'latitude', None),
                'longitude': getattr(s, 'longitude', None),
                # Optional fields present in this model
                'inventory_level': getattr(s, 'inventory_level', None),
                'temperature': getattr(s, 'temperature', None),
                'humidity': getattr(s, 'humidity', None),
                'asset_utilization': getattr(s, 'asset_utilization', None),
                'demand_forecast': getattr(s, 'demand_forecast', None),
                'traffic_status': s.traffic_status.value if getattr(s, 'traffic_status', None) else None,
                'logistics_delay_reason': s.logistics_delay_reason.value if getattr(s, 'logistics_delay_reason', None) else None,
                'estimated_delay_hours': getattr(s, 'estimated_delay_hours', None)
            }
            data.append(shipment_dict)
        
        # Extra guard: apply Python-side filter if needed
        if (start_dt or end_dt) and data:
            filtered_data = []
            for item in data:
                item_date = item['timestamp']
                if isinstance(item_date, str):
                    from dateutil.parser import parse
                    item_date = parse(item_date)
                if start_dt and item_date < start_dt:
                    continue
                if end_dt and item_date > end_dt:
                    continue
                filtered_data.append(item)
            data = filtered_data
        
        return data
    except Exception as e:
        logger.error(f"Error applying analytics filters: {e}")
        return []


if ANALYTICS_AVAILABLE:
    
    @router.get("/status", tags=["Advanced Analytics"])
    async def get_analytics_status():
        """Get analytics system status"""
        try:
            analytics = get_analytics_engine()
            reporting = get_reporting_service()
            
            return {
                "analytics_available": True,
                "reporting_available": True,
                "supported_analysis_types": ["descriptive", "diagnostic", "predictive", "prescriptive"],
                "supported_report_types": ["executive_summary", "operational_dashboard", "predictive_forecast", "kpi_scorecard"],
                "features": {
                    "shipment_trends_analysis": True,
                    "root_cause_analysis": True,
                    "delay_forecasting": True,
                    "kpi_evaluation": True,
                    "visualization_generation": True,
                    "report_export": True
                },
                "system_info": {
                    "version": "1.0.0",
                    "timestamp": datetime.utcnow().isoformat()
                }
            }
        except Exception as e:
            logger.error(f"Error getting analytics status: {e}")
            return {"success": False, "error": str(e)}

    @router.post("/descriptive", tags=["Advanced Analytics"])
    async def analyze_descriptive(request: AnalyticsRequest, db: Session = Depends(get_db)):
        """Perform descriptive analytics on shipment data"""
        try:
            analytics = get_analytics_engine()
            
            # Get filtered data
            data = apply_analytics_filters(db, request.asset_id, request.start_date, request.end_date)
            
            if not data:
                return {"success": False, "error": "No data found for specified criteria"}
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            # Perform descriptive analysis
            result = analytics.perform_descriptive_analysis(df)
            
            response = {
                "success": True,
                "analysis_type": "descriptive",
                "data_summary": {
                    "total_records": len(df),
                    "date_range": {
                        "start": df['timestamp'].min().strftime('%Y-%m-%d') if 'timestamp' in df.columns and not df.empty else None,
                        "end": df['timestamp'].max().strftime('%Y-%m-%d') if 'timestamp' in df.columns and not df.empty else None
                    }
                },
                "results": result
            }
            return _sanitize_for_json(response)
        except Exception as e:
            logger.error(f"Descriptive analytics error: {e}")
            return {"success": False, "error": str(e)}

    @router.post("/diagnostic", tags=["Advanced Analytics"])
    async def analyze_diagnostic(request: AnalyticsRequest, db: Session = Depends(get_db)):
        """Perform root cause analysis on shipment delays"""
        try:
            analytics = get_analytics_engine()
            
            # Get filtered data
            data = apply_analytics_filters(db, request.asset_id, request.start_date, request.end_date)
            
            if not data:
                return {"success": False, "error": "No data found for specified criteria"}
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            # Perform root cause analysis
            result = analytics.perform_root_cause_analysis(df)
            
            response = {
                "success": True,
                "analysis_type": "diagnostic",
                "data_summary": {
                    "total_records": len(df),
                    "delayed_records": len(df[df.get('logistics_delay', False) == True]) if not df.empty else 0
                },
                "results": result
            }
            return _sanitize_for_json(response)
        except Exception as e:
            logger.error(f"Diagnostic analytics error: {e}")
            return {"success": False, "error": str(e)}

    @router.post("/predictive", tags=["Advanced Analytics"])
    async def analyze_predictive(request: AnalyticsRequest, db: Session = Depends(get_db)):
        """Generate predictive insights and forecasts"""
        try:
            analytics = get_analytics_engine()
            
            # Get filtered data
            data = apply_analytics_filters(db, request.asset_id, request.start_date, request.end_date)
            
            if not data:
                return {"success": False, "error": "No data found for specified criteria"}
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            # Generate predictive insights
            forecast_days = 7  # Default forecast period (used for metadata only)
            # The analytics engine handles its own forecast horizon internally
            result = analytics.generate_predictive_insights(df)
            
            response = {
                "success": True,
                "analysis_type": "predictive",
                "forecast_days": forecast_days,
                "data_summary": {
                    "total_records": len(df),
                    "training_period": {
                        "start": df['timestamp'].min().strftime('%Y-%m-%d') if 'timestamp' in df.columns and not df.empty else None,
                        "end": df['timestamp'].max().strftime('%Y-%m-%d') if 'timestamp' in df.columns and not df.empty else None
                    }
                },
                "results": result
            }
            return _sanitize_for_json(response)
        except Exception as e:
            logger.error(f"Predictive analytics error: {e}")
            return {"success": False, "error": str(e)}

    @router.post("/prescriptive", tags=["Advanced Analytics"])
    async def analyze_prescriptive(request: AnalyticsRequest, db: Session = Depends(get_db)):
        """Generate prescriptive recommendations and actionable insights"""
        try:
            analytics = get_analytics_engine()
            
            # Get filtered data
            data = apply_analytics_filters(db, request.asset_id, request.start_date, request.end_date)
            
            if not data:
                return {"success": False, "error": "No data found for specified criteria"}
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            # Generate prescriptive insights
            result = analytics.generate_prescriptive_insights(df)
            
            response = {
                "success": True,
                "analysis_type": "prescriptive",
                "data_summary": {
                    "total_records": len(df),
                    "issues_identified": len(result.get('issues', []))
                },
                "results": result
            }
            return _sanitize_for_json(response)
        except Exception as e:
            logger.error(f"Prescriptive analytics error: {e}")
            return {"success": False, "error": str(e)}

    @router.post("/comprehensive-report", tags=["Advanced Analytics"])
    async def generate_comprehensive_analytics_report(request: AnalyticsRequest, db: Session = Depends(get_db)):
        """Generate comprehensive analytics report with all analysis types"""
        try:
            analytics = get_analytics_engine()
            
            # Get filtered data
            data = apply_analytics_filters(db, request.asset_id, request.start_date, request.end_date)
            
            if not data:
                return {"success": False, "error": "No data found for specified criteria"}
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            # Determine analysis types
            analysis_types = request.analysis_types or ['descriptive', 'diagnostic', 'predictive', 'prescriptive']
            
            # Generate comprehensive report
            report = analytics.generate_comprehensive_report(df, analysis_types=analysis_types)
            
            response = {
                "success": True,
                "report_type": "comprehensive",
                "analysis_types": analysis_types,
                "generated_at": datetime.utcnow().isoformat(),
                "data_summary": {
                    "total_records": len(df),
                    "period": {
                        "start": df['timestamp'].min().strftime('%Y-%m-%d') if 'timestamp' in df.columns and not df.empty else None,
                        "end": df['timestamp'].max().strftime('%Y-%m-%d') if 'timestamp' in df.columns and not df.empty else None
                    }
                },
                "report": report.__dict__ if hasattr(report, '__dict__') else report
            }
            return _sanitize_for_json(response)
        except Exception as e:
            logger.error(f"Comprehensive analytics report error: {e}")
            return {"success": False, "error": str(e)}

else:
    # Placeholder endpoints when analytics system is not available
    @router.get("/status", tags=["Advanced Analytics"])
    async def analytics_status():
        """Analytics system status"""
        return {
            "available": False,
            "message": "Advanced Analytics system not available. Check dependencies and configuration."
        }


# Reporting Router
reporting_router = APIRouter()

if ANALYTICS_AVAILABLE:
    
    @reporting_router.post("/executive-summary", tags=["Advanced Reports"])
    async def generate_executive_summary_report(request: ReportRequest, db: Session = Depends(get_db)):
        """Generate executive summary report with visualizations"""
        try:
            reporting = get_reporting_service()
            
            # Get filtered data
            data = apply_analytics_filters(db, request.asset_id, request.start_date, request.end_date)
            
            if not data:
                return {"success": False, "error": "No data found for specified criteria"}
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            # Generate report
            result = reporting.generate_executive_summary_report(df, period_days=request.period_days)
            
            return result
        except Exception as e:
            logger.error(f"Executive summary report error: {e}")
            return {"success": False, "error": str(e)}

    @reporting_router.post("/operational-dashboard", tags=["Advanced Reports"])
    async def generate_operational_dashboard_report(request: ReportRequest, db: Session = Depends(get_db)):
        """Generate operational dashboard report"""
        try:
            reporting = get_reporting_service()
            
            # Get filtered data
            data = apply_analytics_filters(db, request.asset_id, request.start_date, request.end_date)
            
            if not data:
                return {"success": False, "error": "No data found for specified criteria"}
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            # Generate report
            result = reporting.generate_operational_dashboard_report(df)
            
            return result
        except Exception as e:
            logger.error(f"Operational dashboard report error: {e}")
            return {"success": False, "error": str(e)}

    @reporting_router.post("/predictive-forecast", tags=["Advanced Reports"])
    async def generate_predictive_forecast_report(request: ReportRequest, db: Session = Depends(get_db)):
        """Generate predictive forecast report"""
        try:
            reporting = get_reporting_service()
            
            # Get filtered data
            data = apply_analytics_filters(db, request.asset_id, request.start_date, request.end_date)
            
            if not data:
                return {"success": False, "error": "No data found for specified criteria"}
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            # Generate report
            result = reporting.generate_predictive_forecast_report(df, forecast_days=request.forecast_days)
            
            return result
        except Exception as e:
            logger.error(f"Predictive forecast report error: {e}")
            return {"success": False, "error": str(e)}

    @reporting_router.post("/kpi-scorecard", tags=["Advanced Reports"])
    async def generate_kpi_scorecard_report(request: ReportRequest, db: Session = Depends(get_db)):
        """Generate KPI performance scorecard"""
        try:
            reporting = get_reporting_service()
            
            # Get filtered data
            data = apply_analytics_filters(db, request.asset_id, request.start_date, request.end_date)
            
            if not data:
                return {"success": False, "error": "No data found for specified criteria"}
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            # Generate report
            result = reporting.generate_kpi_scorecard_report(df)
            
            return result
        except Exception as e:
            logger.error(f"KPI scorecard report error: {e}")
            return {"success": False, "error": str(e)}

    @reporting_router.post("/export", tags=["Advanced Reports"])
    async def export_report(request: ReportExportRequest):
        """Export report data to file"""
        try:
            reporting = get_reporting_service()
            
            filename = request.filename
            if not filename:
                timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                filename = f"report_{timestamp}.{request.format}"
            
            # Export report
            if request.format.lower() == 'json':
                result_path = reporting.export_report_to_json(request.report_data, filename)
            else:
                return {"success": False, "error": "Unsupported export format. Currently only JSON is supported."}
            
            return {
                "success": True,
                "exported_file": result_path,
                "format": request.format
            }
        except Exception as e:
            logger.error(f"Report export error: {e}")
            return {"success": False, "error": str(e)}

    @reporting_router.get("/status", tags=["Advanced Reports"])
    async def get_reporting_status():
        """Get reporting system status"""
        try:
            reporting = get_reporting_service()
            
            return {
                "reporting_available": True,
                "supported_report_types": ["executive_summary", "operational_dashboard", "predictive_forecast", "kpi_scorecard"],
                "supported_formats": ["json"],
                "features": {
                    "visualization_generation": True,
                    "interactive_charts": True,
                    "report_export": True,
                    "automated_insights": True
                },
                "system_info": {
                    "version": "1.0.0",
                    "timestamp": datetime.utcnow().isoformat()
                }
            }
        except Exception as e:
            logger.error(f"Error getting reporting status: {e}")
            return {"success": False, "error": str(e)}

else:
    # Placeholder endpoints when reporting system is not available
    @reporting_router.get("/status", tags=["Advanced Reports"])
    async def reporting_status():
        """Reporting system status"""
        return {
            "available": False,
            "message": "Advanced Reporting system not available. Check dependencies and configuration."
        }
