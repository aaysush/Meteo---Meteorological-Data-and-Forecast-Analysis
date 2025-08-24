# Meteorological Data Forecasting and Insights
https://meteo-front.onrender.com/

**NOTE** : when entering Co-ordinates enter only int part
eg. 18.9582, 72.8321   
**NOT** the symbols ,letter or directional indicators eg. 18.9582°N, 72.8321°E


A comprehensive weather forecasting system that provides real-time meteorological data analysis and predictions using machine learning algorithms. The system features a FastAPI backend for data processing and a Streamlit frontend for interactive visualization.

## 🌟 Features

### Real-time Weather Data
- **Temperature Analysis**: Current temperature and apparent temperature (feels-like) readings
- **Atmospheric Conditions**: Dewpoint, humidity, and vapor pressure deficit calculations
- **Wind Information**: Speed, direction analysis with cyclical encoding
- **Precipitation Data**: Rainfall measurements and probability forecasting
- **Atmospheric Pressure**: Surface and sea-level pressure monitoring
- **Visibility & UV Index**: Environmental safety parameters
- **Cloud Coverage**: Average cloud cover percentage

### Machine Learning Predictions
- **Temperature Forecasting**: 5-hour ahead predictions using Random Forest Regressor
- **Humidity Predictions**: Relative humidity forecasting for optimal comfort planning
- **Rainfall Probability**: XGBoost-powered precipitation chance predictions
- **Iterative Prediction**: Sequential hourly forecasts with increasing accuracy

## 🏗️ Architecture

### Backend (FastAPI)
- RESTful API for weather data processing
- Asynchronous request handling
- Integration with WeatherAPI for historical data
- Machine learning pipeline for predictions

### Frontend (Streamlit)
- Interactive web interface
- Real-time data visualization
- Multi-tab analysis dashboard
- Geographic coordinate input system

## 📊 Technical Stack

### Core Libraries
- **Data Processing**: pandas, numpy
- **Machine Learning**: scikit-learn, XGBoost
- **API Framework**: FastAPI
- **Frontend**: Streamlit
- **Visualization**: seaborn, matplotlib
- **Weather Data**: WeatherAPI integration

### Machine Learning Models
- **Random Forest Regressor**: Temperature and humidity predictions
- **XGBoost Regressor**: Rainfall probability forecasting
- **Feature Engineering**: Cyclical encoding for temporal and directional data

## 🚀 Installation

### Prerequisites
- Python 3.8+
- WeatherAPI key (sign up at [WeatherAPI.com](https://www.weatherapi.com/))

### Backend Setup
```bash
# Clone the repository
git clone <repository-url>
cd weather-forecasting-project

# Install dependencies
pip install -r requirements.txt

# Set your WeatherAPI key
# Replace API_KEY in the backend code with your actual key

# Run FastAPI server
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

### Frontend Setup
```bash
# Run Streamlit application
streamlit run main_page.py
```

## 📋 Requirements

Create a `requirements.txt` file with the following dependencies:

```txt
pandas>=1.5.0
numpy>=1.21.0
requests>=2.28.0
scikit-learn>=1.1.0
xgboost>=1.6.0
fastapi>=0.68.0
uvicorn>=0.15.0
streamlit>=1.25.0
seaborn>=0.11.0
matplotlib>=3.5.0
pytz>=2022.1
pydantic>=1.8.0
```

## 🔧 API Usage

### Endpoint: `/process-data/`
**Method**: POST

**Request Body**:
```json
{
    "latitude": 19.0760,
    "longitude": 72.8777
}
```

**Response**: JSON array containing processed weather data with predictions

**Example**:
```bash
curl -X POST "http://127.0.0.1:8000/process-data/" \
     -H "Content-Type: application/json" \
     -d '{"latitude": 19.0760, "longitude": 72.8777}'
```

## 📱 Frontend Interface

### Main Page
- Project overview and feature descriptions
- Introduction to forecasting capabilities

### Analysis & Prediction Page
- **Sidebar**: Coordinate input (latitude/longitude)
- **Current Tab**: Real-time weather conditions display
- **Forecast Tab**: 5-hour ahead predictions for temperature, humidity, and rainfall
- **Weekly Analysis Tab**: Historical data visualization and trend analysis

## 🧠 Machine Learning Pipeline

### 1. Data Collection
- Fetches 7 days of historical weather data from WeatherAPI
- Processes hourly data points for comprehensive analysis

### 2. Feature Engineering
- **Temporal Features**: Hour encoding using sine/cosine transformation
- **Wind Direction**: Conversion to x,y components for cyclical data
- **Weather Conditions**: One-hot encoding for categorical data
- **Future Features**: Shifted target variables for time series prediction

### 3. Prediction Models

#### Temperature Forecasting
- **Model**: Random Forest Regressor
- **Features**: Correlated meteorological variables (correlation > 0.25)
- **Process**: Iterative prediction with previously predicted values as features

#### Humidity Forecasting
- **Model**: Random Forest Regressor
- **Target**: Relative humidity levels
- **Applications**: Comfort index and agricultural planning

#### Rainfall Probability
- **Model**: XGBoost Regressor
- **Output**: Percentage chance of precipitation
- **Range**: Capped at 100% maximum probability

## 📊 Data Processing

### Weather Data Features
| Original Column | Description | Units |
|----------------|-------------|-------|
| temperature_2m | Temperature at 2m above surface | °C |
| relative_humidity_2m | Relative humidity at 2m | % |
| dewpoint_2m | Dew point temperature | °C |
| apparent_temperature | Feels-like temperature | °C |
| wind_speed_10m | Wind speed at 10m height | km/h |
| wind_direction | Wind direction | Degrees |
| cloud_cover_avg | Average cloud coverage | % |
| surface_pressure | Atmospheric pressure at surface | hPa |
| rainfall | Precipitation amount | mm |
| uv_index | UV radiation index | Index |
| vapour_pressure_deficit | VPD for agricultural applications | kPa |

### Prediction Targets
- **Temperature**: 1h, 2h, 3h, 4h, 5h ahead
- **Humidity**: 1h, 2h, 3h, 4h, 5h ahead  
- **Rainfall Chance**: 1h, 2h, 3h, 4h, 5h ahead

## 🌍 Geographic Coverage

The system supports global weather data fetching using latitude and longitude coordinates. Tested locations include:
- Mumbai, India (19.0760°N, 72.8777°E)
- Delhi, India (28.6139°N, 77.2090°E)

## ⚠️ Important Notes

### API Limitations
- WeatherAPI key required for data fetching
- Rate limits apply based on your WeatherAPI plan
- Historical data limited to past 7 days for processing

### Model Constraints
- Predictions based on historical patterns
- Accuracy may vary with extreme weather conditions
- Model retraining recommended for different geographic regions

## 🔐 Configuration

### WeatherAPI Setup
1. Sign up at [WeatherAPI.com](https://www.weatherapi.com/)
2. Generate your API key
3. Replace `API_KEY` constant in the backend code
4. Ensure your plan supports historical weather data access

### Timezone Configuration
- Default timezone: Asia/Kolkata (Indian Standard Time)
- Modify `LOCAL_TZ` variable for different regions

## 🚀 Deployment

### Local Development
```bash
# Terminal 1 - Backend
uvicorn main:app --reload --host 127.0.0.1 --port 8000

# Terminal 2 - Frontend
streamlit run main_page.py
```

### Production Deployment
- FastAPI: Deploy using Docker, Heroku, or cloud services
- Streamlit: Use Streamlit Cloud, Docker, or cloud platforms
- Database: Consider adding persistent storage for historical data

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

For questions, suggestions, or support, please open an issue in the repository or contact the development team.


---

**Built by Aayush Pandey**
