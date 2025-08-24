# Meteorological Data Forecasting and Insights
[https://meteo-front.onrender.com/](https://meteo-front.onrender.com/)

<img width="718" height="475" alt="{1886C309-D2BE-476A-90DE-0E6F53540E56}" src="https://github.com/user-attachments/assets/c667bbe3-3cf5-4f8c-8271-6b496448e896" />




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
# System Architecture Design
## Meteorological Data Forecasting and Insights Platform

### Architecture Overview

The Meteorological Data Forecasting and Insights Platform follows a modern microservices architecture pattern with clear separation of concerns between presentation, application, and data layers. The system is designed for scalability, reliability, and optimal performance across distributed cloud infrastructure.

---

## 🏛️ High-Level Architecture

### System Components

#### **Frontend Layer (Presentation Tier)**
- **Platform**: Streamlit Web Application
- **Deployment**: Render Cloud Platform
- **Responsibilities**: 
  - User interface and experience
  - Coordinate input collection (latitude/longitude)
  - Data visualization and analytics dashboard
  - Real-time weather data presentation

#### **Application Layer (Business Logic Tier)**
- **Platform**: AWS EC2 Instance
- **Web Server**: Nginx (Reverse Proxy)
- **API Framework**: FastAPI (Containerized)
- **Container Runtime**: Docker
- **Responsibilities**:
  - Request routing and load balancing
  - Business logic processing
  - Machine learning model execution
  - Data transformation and analysis

#### **External Services Layer**
- **Weather Data Provider**: WeatherAPI.com
- **Purpose**: Historical and real-time meteorological data source

---

## 🔄 Data Flow Architecture

### Request-Response Cycle

```
User Input → Streamlit Frontend → HTTP Request → AWS EC2 → Nginx → Docker Container → FastAPI → ML Processing → WeatherAPI → Response Processing → User Display
```

### Detailed Flow Description

1. **User Interaction**
   - User accesses Streamlit web application on Render
   - Enters geographical coordinates (latitude, longitude)
   - Initiates weather analysis request

2. **Frontend Processing**
   - Streamlit captures user input
   - Validates coordinate parameters
   - Sends HTTP POST request to backend API endpoint

3. **Load Balancing & Routing**
   - Request reaches AWS EC2 instance
   - Nginx reverse proxy receives incoming request
   - Routes traffic to appropriate Docker container
   - Implements load balancing and SSL termination

4. **Backend Processing**
   - Docker container hosts FastAPI application
   - FastAPI validates request payload
   - Initiates asynchronous processing pipeline

5. **Machine Learning Pipeline**
   - Feature engineering and data preprocessing
   - Model inference using trained algorithms:
     - Random Forest for temperature predictions
     - XGBoost for rainfall probability
   - Iterative forecasting for 5-hour predictions

6. **External Data Integration**
   - API calls to WeatherAPI.com for historical data
   - Data aggregation and cleansing
   - Feature extraction and transformation

7. **Response Generation**
   - Processed predictions compiled into JSON response
   - Data validation and formatting
   - Response sent back through the architecture stack

8. **Frontend Rendering**
   - Streamlit receives processed data
   - Dynamic visualization generation
   - Interactive dashboard updates
   - User-friendly data presentation

---

## 🏗️ Infrastructure Architecture

### Cloud Infrastructure Components

#### **Frontend Infrastructure (Render)**
- **Service Type**: Static Site Hosting
- **Auto-scaling**: Managed by Render
- **CDN**: Built-in content delivery
- **SSL/TLS**: Automated certificate management

#### **Backend Infrastructure (AWS EC2)**
- **Instance Type**: Optimized for ML workloads
- **Operating System**: Linux-based distribution
- **Security Groups**: Configured firewall rules
- **Elastic IP**: Static IP address assignment

#### **Container Architecture**
```dockerfile
# Simplified Docker Architecture
FROM python:3.9-slim
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . /app
WORKDIR /app
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### **Nginx Configuration**
- **Port Binding**: Listens on port 80/443
- **Proxy Pass**: Forwards to Docker container port 8000
- **Load Balancing**: Round-robin distribution
- **Caching**: Static content caching
- **Compression**: Gzip compression enabled

---

## 🔧 Technical Stack

### Frontend Stack
| Component | Technology | Purpose |
|-----------|------------|---------|
| Framework | Streamlit | Interactive web applications |
| Deployment | Render | Cloud hosting platform |
| Language | Python | Application development |
| Libraries | pandas, numpy, seaborn | Data manipulation and visualization |

### Backend Stack
| Component | Technology | Purpose |
|-----------|------------|---------|
| API Framework | FastAPI | RESTful API development |
| Web Server | Nginx | Reverse proxy and load balancer |
| Containerization | Docker | Application packaging |
| Cloud Platform | AWS EC2 | Scalable compute infrastructure |
| ML Libraries | scikit-learn, XGBoost | Machine learning algorithms |

### External Dependencies
| Service | Provider | Purpose |
|---------|----------|---------|
| Weather Data API | WeatherAPI.com | Historical meteorological data |
| DNS Management | Cloud Provider | Domain name resolution |
| SSL Certificates | Let's Encrypt/Cloud Provider | Secure communications |

---

## 🛡️ Security Architecture

### Security Measures

#### **Network Security**
- **AWS Security Groups**: Firewall rules restricting inbound/outbound traffic

#### **Application Security**
- **Input Validation**: Coordinate parameter validation and sanitization
- **CORS Configuration**: Cross-origin resource sharing controls
- **Error Handling**: Secure error messages without sensitive data exposure(every key is in proper environment variable)

#### **Infrastructure Security**
- **Container Isolation**: Docker container security boundaries
- **Access Control**: SSH key-based authentication for EC2 access

---

## 📈 Scalability Design

### Horizontal Scaling Capabilities

#### **Frontend Scaling**
- **Render Auto-scaling**: Automatic traffic-based scaling
- **CDN Distribution**: Global content delivery network
- **Caching Strategy**: Browser and CDN caching optimization

#### **Backend Scaling**
- **Docker Swarm/Kubernetes**: Container orchestration ready
- **AWS Auto Scaling Groups**: EC2 instance auto-scaling
- **Load Balancer Integration**: Application Load Balancer support(future aspect)

#### **Database Scaling** (Future Enhancement)
- **Read Replicas**: Distributed read operations

---


---

## 🚀 Deployment Strategy


#### **Frontend Deployment**
1. Code commit to repository
2. Render automatic build trigger
3. Dependency installation
4. Application deployment
5. Health check validation

#### **Backend Deployment**
1. Docker image building
2. Container registry push
3. EC2 deployment automation
4. Nginx configuration update
5. Service restart and validation

### Environment Management
- **Development**: Local Docker environment
- **Staging**: Separate AWS environment
- **Production**: Full production infrastructure

---

## 📊 Architecture Benefits

### Performance Advantages
- **Asynchronous Processing**: Non-blocking request handling
- **Containerization**: Consistent deployment environments
- **Load Balancing**: Distributed traffic management
- **Caching**: Reduced API call overhead

### Reliability Features
- **Fault Isolation**: Container-based error containment
- **Auto-recovery**: Automatic service restart capabilities

---

This architecture design ensures a robust, scalable, and maintainable weather forecasting platform capable of handling growing user demands while maintaining high performance and reliability standards.

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
