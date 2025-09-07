# 🚢 Maersk AI Analytics Platform

### Advanced Business Intelligence & Predictive Analytics System for Supply Chain Optimization

A comprehensive enterprise-grade analytics platform designed for Maersk's logistics operations, featuring real-time shipment tracking, AI-powered delay predictions, and advanced business intelligence capabilities.

---

## 🌟 Key Features

### Core Capabilities
- **Real-time Shipment Tracking**: Monitor shipments with location tracking, status updates, and delay detection.
- **4-Tier Analytics Engine**: 
  - **Descriptive Analytics**: KPI monitoring and performance metrics.
  - **Diagnostic Analytics**: Root cause analysis for delays and bottlenecks.
  - **Predictive Analytics**: 7-day forecasting with risk assessment.
  - **Prescriptive Analytics**: Optimization recommendations and resource allocation.
- **Interactive Dashboard**: Executive summaries, KPI gauges, trend visualizations, and comprehensive reporting.
- **RESTful API**: 20+ endpoints for data management, analytics, and reporting.

### Technical Highlights
- **Scalable Architecture**: Microservices design with modular components.
- **Advanced Visualizations**: Interactive Plotly charts with real-time updates.
- **Intelligent Insights**: Automated insight generation with actionable recommendations.
- **Performance Optimized**: Database indexing, async processing, and efficient data pipelines.

---

## 🛠️ Tech Stack

### Backend
- **Framework**: FastAPI
- **Database**: SQLAlchemy ORM with SQLite/PostgreSQL
- **Analytics**: Pandas, NumPy, Scikit-learn
- **API Design**: RESTful architecture with Pydantic validation

### Frontend
- **Dashboard**: Streamlit
- **Visualizations**: Plotly
- **UI Components**: Custom CSS, responsive design

### Infrastructure
- **Caching**: Redis
- **Task Queue**: Celery
- **Monitoring**: Structured logging
- **Deployment**: Docker-ready

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Virtual environment (recommended)
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/maersk_shipment_ai_system.git
   cd maersk_shipment_ai_system
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   # Windows
   .\venv\Scripts\activate
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**
   ```bash
   # Copy example env file
   cp .env.example .env
   # Edit .env with your settings
   ```

### Running the Application

1. **Start the API server**
   ```bash
   uvicorn api.main:app --reload --host localhost --port 8080
   ```
   API will be available at: http://localhost:8080
   - Swagger Docs: http://localhost:8080/docs
   - ReDoc: http://localhost:8080/redoc

2. **Launch the Dashboard** (in a new terminal)
   ```bash
   streamlit run dashboard/main.py
   ```
   Dashboard will open at: http://localhost:8501

---

## 📁 Project Structure

```
maersk_shipment_ai_system/
├── api/                    # FastAPI application
│   ├── main.py            # Main API entry point
│   └── routers/           # API endpoints
│       ├── analytics.py   # Analytics endpoints
│       ├── shipments.py   # Shipment CRUD
│       └── health.py      # Health checks
├── analytics/             # Analytics engine
│   ├── analytics_engine.py    # Core analytics logic
│   └── reporting_service.py   # Report generation
├── core/                  # Core modules
│   ├── models/           # Database models
│   ├── schemas/          # Pydantic schemas
│   └── database/         # Database configuration
├── dashboard/            # Streamlit dashboard
│   └── main.py          # Dashboard application
├── services/            # Business services
│   ├── shipment_service.py    # Shipment logic
│   └── data_service.py        # Data management
├── data/                # Data files
├── ml/                  # Machine Learning models and services
├── tests/               # Automated tests
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

---

## 📈 API Endpoints

### Core Endpoints
- `GET /health` - System health check
- `GET /analytics/status` - Analytics service status
- `GET /data/validate` - Validate database integrity

### Shipment Management
- `GET /shipments` - List shipments with filters
- `POST /shipments` - Create new shipment
- `GET /shipments/{id}` - Get shipment details
- `PUT /shipments/{id}` - Update shipment
- `DELETE /shipments/{id}` - Delete shipment

### Analytics & Reporting
- `POST /analytics/descriptive` - Descriptive analytics
- `POST /analytics/diagnostic` - Root cause analysis
- `POST /analytics/predictive` - Predictive insights
- `POST /analytics/prescriptive` - Optimization recommendations
- `POST /reports/executive-summary` - Executive report
- `POST /reports/operational-dashboard` - Operations report
- `POST /reports/predictive-forecast` - Forecast report
- `POST /reports/kpi-scorecard` - KPI report

### Data Management
- `POST /data/load` - Load sample data
- `GET /data/summary` - Dataset summary
- `POST /data/load-real` - Load CSV data

---

## 🤖 Model Card - Delay Prediction Model

### Model Overview
- **Model Type**: Random Forest Classifier
- **Task**: Binary classification (Delayed vs On-Time)
- **Training Date**: September 2025
- **Dataset Size**: 1,000 shipments
- **Train/Test Split**: 80/20 stratified

### Features Used
The model uses **29 engineered features** from legitimate operational data:

#### Geographic Features
- Latitude, Longitude, Distance from origin
- Regional encoding (lat/lon regions)

#### Temporal Features  
- Hour, Day of week, Month, Quarter
- Cyclical time encodings (sin/cos transformations)

#### Operational Features
- Distance traveled
- Fuel efficiency
- Waiting time
- Traffic conditions (encoded)
- Weather conditions (encoded)

#### Asset Features
- Asset ID encoding
- Historical asset performance metrics
- Asset-specific delay statistics

### Excluded Features (Target Leakage Prevention)
⚠️ The following columns are explicitly **excluded** to prevent data leakage:
- `Shipment_Status` - directly indicates if delayed
- `Logistics_Delay_Reason` - only known after delay occurs
- `status` - redundant with target variable
- `delay_category` - derived from target

### Model Performance

#### Without Target Leakage (Production-Ready)
- **Accuracy**: 73.0%
- **Precision**: 0.0% (needs more diverse training data)
- **Recall**: 0.0% (model is conservative)
- **F1 Score**: 0.0% (requires rebalancing)
- **ROC AUC**: 0.453 (near random, due to synthetic data)
- **Cross-Validation ROC AUC**: 0.495 ± 0.078

#### Top Feature Importance
1. Fuel efficiency (8.10%)
2. Latitude (7.11%)
3. Longitude (6.94%)
4. Distance from origin (6.55%)
5. Distance traveled (6.42%)

### Validation Methodology
- **Strategy**: Stratified K-Fold Cross-Validation (k=5)
- **Metrics**: ROC AUC (primary), Accuracy, Precision, Recall, F1
- **Test Set**: Hold-out 20% for final evaluation

### Limitations & Considerations

#### Current Limitations
1. **Synthetic Data**: Current model trained on generated delay patterns
2. **Low Predictive Power**: ROC AUC ~0.5 indicates random performance
3. **Class Imbalance**: Model struggles with minority class (delays)
4. **Feature Engineering**: Requires real-world validation

#### Production Readiness Checklist
- ✅ Target leakage removed
- ✅ Cross-validation implemented
- ✅ Feature importance tracked
- ✅ Model versioning in place
- ⚠️ Needs real data for meaningful predictions
- ⚠️ Requires hyperparameter tuning
- ⚠️ Should implement model calibration

### Recommended Improvements
1. **Data Quality**: Train on real historical shipment data
2. **Feature Engineering**: Add route-specific features, seasonal patterns
3. **Model Selection**: Try XGBoost or ensemble methods
4. **Hyperparameter Tuning**: Use GridSearchCV or Bayesian optimization
5. **Calibration**: Apply Platt scaling or isotonic regression
6. **Monitoring**: Implement drift detection and performance tracking

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new feature branch (`git checkout -b feature/your-feature`).
3. Make your changes and commit them (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a pull request.

---

## 📝 License

This project is proprietary software developed for Maersk logistics operations.

---

**Built with ❤️ for Maersk Supply Chain Excellence**
