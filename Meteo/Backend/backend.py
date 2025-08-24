import pandas as pd
import numpy as np
import requests
import math
from datetime import datetime, timedelta
import pytz
from sklearn.preprocessing import  OneHotEncoder
import re 
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
import matplotlib.pyplot as plt
 
import xgboost as xgb # we used it for xgb regressor

#---------------------------------------------------------------------------------------------

# Constants
API_KEY = "6fbf86cc0b0640b4a6983900253105"
LOCAL_TZ = pytz.timezone('Asia/Kolkata')  # Define local timezone


# Utility Functions
def get_current_local_time():
    """Get the current local time."""
    current_utc_time = datetime.utcnow().replace(tzinfo=pytz.utc)
    return current_utc_time.astimezone(LOCAL_TZ)

def calculate_vpd(temp_c, relative_humidity):
    """Calculate vapor pressure deficit."""
    svp = 0.6108 * math.exp((17.27 * temp_c) / (temp_c + 237.3))  # Saturated vapor pressure
    avp = (relative_humidity / 100) * svp  # Actual vapor pressure
    vpd = svp - avp  # Vapor pressure deficit
    return round(vpd, 3)

def fetch_weather_data(lat, lon):
    """Fetch weather data for the past 7 days for a given location."""
    data_training = []
    current_local_time = get_current_local_time()

    for day_delta in range(7):
        date = current_local_time.date() - timedelta(days=day_delta)
        date_str = date.strftime("%Y-%m-%d")
        url = f"http://api.weatherapi.com/v1/history.json?key={API_KEY}&q={lat},{lon}&dt={date_str}"
        
        print(f"Fetching data for {date_str} from URL: {url}")
        response = requests.get(url)

        if response.status_code != 200:
            print(f"Error fetching data for {date_str}: {response.status_code} - {response.text}")
            continue

        weather_json = response.json()
        forecast_day = weather_json.get("forecast", {}).get("forecastday", [])
        if not forecast_day:
            print(f"No forecast data available for {date_str}.")
            continue

        hours = forecast_day[0].get("hour", [])
        for hour in hours:
            temp_c = hour.get("temp_c")
            rh = hour.get("humidity")
            if temp_c is None or rh is None:
                continue
            vpd = calculate_vpd(temp_c, rh)

            data_training.append({
                "datetime": hour["time"],
                "temperature_2m": temp_c,
                "dewpoint_2m": hour.get("dewpoint_c"),
                "apparent_temperature": hour.get("feelslike_c"),
                "wind_speed_10m": hour.get("wind_kph"),
                "wind_direction": hour.get("wind_dir"),
                "cloud_cover_avg": hour.get("cloud"),
                "surface_pressure": hour.get("pressure_mb"),
                "sealevel_pressure": hour.get("pressure_mb"),
                "rainfall": hour.get("precip_mm"),
                "snowfall": hour.get("snow", 0),
                "relative_humidity_2m": rh,
                "visibility": hour.get("vis_km"),
                "uv_index": hour.get("uv"),
                "chance_of_rain": hour.get("chance_of_rain", 0),
                "weather_condition": hour.get("condition", {}).get("text", "Unknown"),
                "vapour_pressure_deficit": vpd
            })

    if not data_training:
        print("No valid weather data collected. Returning an empty DataFrame.")
        return pd.DataFrame()

    df = pd.DataFrame(data_training)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values(by="datetime")
    
    # Ensure only past data is included
    current_local_time_dt = pd.to_datetime(current_local_time.strftime('%Y-%m-%d %H:%M:%S'))
    df = df[df["datetime"] <= current_local_time_dt]

    print('FETCH WEATHER DATA RUN COMPLETE')
    return df


def hour_conversion(df):
    df['hour'] = df['datetime'].dt.hour
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df.drop(['hour'], axis=1, inplace=True)

    print('HOUUR CONVERSION RUN COMPLETE')
    return df

    # Data Preparation Functions

def add_future_features(df):
    """Add future hour features for prediction."""
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)
    for h in range(1, 6):
        df[f'temperature_2m_{h}h'] = df['temperature_2m'].shift(-h)
        df[f'relative_humidity_2m_{h}h'] = df['relative_humidity_2m'].shift(-h)
        df[f'chance_of_rain_{h}h'] = df['chance_of_rain'].shift(-h)

        print('ADD FUTURE RUN COMPLETE')
    return df

def add_wind_features(df):
    """Add wind direction as components and encode weather conditions."""
    wind_direction_to_degrees = {
        'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5, 'E': 90,
        'ESE': 112.5, 'SE': 135, 'SSE': 157.5, 'S': 180,
        'SSW': 202.5, 'SW': 225, 'WSW': 247.5, 'W': 270,
        'WNW': 292.5, 'NW': 315, 'NNW': 337.5
    }
    df['wind_degrees'] = df['wind_direction'].map(wind_direction_to_degrees)
    df['wind_x'] = np.cos(np.radians(df['wind_degrees']))
    df['wind_y'] = np.sin(np.radians(df['wind_degrees']))
    df.drop(columns=['wind_direction'], inplace=True)

    print('ADD WINDD RUN COMPLETE')
    return df


def encode_weather_conditions(df):
    """One-hot encode weather conditions."""
    # Initialize the OneHotEncoder with specified options
    encoder = OneHotEncoder(sparse_output=False, drop='first')
    # Fit and transform the 'weather_condition' column
    encoded = encoder.fit_transform(df[['weather_condition']])
    # Create a DataFrame for the encoded features
    df_encoded = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(['weather_condition']))
    # Concatenate the original DataFrame with the encoded features
    df = pd.concat([df, df_encoded], axis=1)
    # Drop the original 'weather_condition' column
    df.drop(columns=['weather_condition'], inplace=True)

    print('ENCODE WEATHER RUN COMPLETE')
    return df

def main(latitude,longitude):
    lat, lon = latitude,longitude  # Example coordinates (Mumbai) i.e we did testing with coordinates of Mumbai and Delhi
 
    
    # Fetch the data from the user 
    weather_data = fetch_weather_data(lat, lon)
       
    # Add features and transformations
    weather_data = add_future_features(weather_data)
    weather_data = add_wind_features(weather_data)
    weather_data = hour_conversion(weather_data)  # Call hour conversion here
    weather_data = encode_weather_conditions(weather_data)

    # Save the processed data to CSV
    weather_data.to_csv('new.csv', index=False, encoding='utf-8')

    return weather_data

#df = main()

#------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------
#TEMPERATURE

target_cols_temperature = [
    "temperature_2m_1h", "temperature_2m_2h",
    "temperature_2m_3h", "temperature_2m_4h", "temperature_2m_5h"
]
 

def plot_temperature_correlation(df):
    """
    Plots a correlation heatmap for features related to temperature.
     ----------------
     we removed this heatmaps as we removes the maps as the 
     GUI operations must be done on min thred but the real opertions are done on non-main thread this violets the ruleof tiner library 
     so we are removing the ploting of matpplotlib graphs 
     ------------
    Args:
        df: DataFrame containing numerical features and 'temperature_2m'.

    Returns:
        List of features with high correlation to 'temperature_2m'.
    
     plt.figure(figsize=(12, 8))
    sns.heatmap(
        corr_matrix[['temperature_2m']].sort_values(by='temperature_2m', ascending=False),
        annot=True,
        cmap='coolwarm',
        fmt='.2f'
    )
    plt.title("Feature Correlation with Temperature")
    plt.show()
    
    """
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    corr_matrix = df[numeric_cols].corr()
    
   
    

    threshold = 0.25
    correlated_features = corr_matrix[
        abs(corr_matrix['temperature_2m']) > threshold
    ]['temperature_2m'].index.tolist()

    print(correlated_features)
    print('RUNNN plot_temperature_correlation')

    return correlated_features


def prepare_features_and_targets(df, correlated_features, target_cols_temperature):
    """
    Prepares the features (X) and targets (Y) DataFrames.

    Args:
        df: Original DataFrame.
        feature_cols: List of feature column names.
        target_cols: List of target column names.

    Returns:
        Tuple of features (X) and targets (Y) DataFrames.
    """

    print(f'\nCORRELATED FEATURES ARE {correlated_features}\n')
    exclude_columns = [s for s in correlated_features if re.search(r'\d+h$', s)]
    filtered_features = [s for s in correlated_features if s not in exclude_columns]

    X = df[filtered_features]
    Y = df[target_cols_temperature]
    

    print(f'\n  X and  Y ARE {X.columns} \n AND {Y.columns}\n')

    print('\n  we r printing  X & Y')
    print(X,Y)

    print('RUNNN PREPAARE FEATUERS AND TARGETS')
    return target_cols_temperature,filtered_features
 

def iterative_fill(df, target_cols_temperature, feature_cols):
    """
    for my own note :
    df: pandas DataFrame containing original features and shifted target columns.
    target_cols: list of target columns to predict in order, e.g. ['temp_1h', 'temp_2h', ..., 'temp_5h']
    feature_cols: list of feature columns to use for training and prediction.

    Returns:
    df_filled: DataFrame with missing values in target_cols filled iteratively.
    """

    df_filled = df.copy()

    for i, target_col in enumerate(target_cols_temperature):
        print(f"\nPredicting and filling missing values for: {target_cols_temperature}")

        # Select rows where target_col is NOT null for training
        train_df = df_filled[df_filled[target_col].notna()]

        # Select rows where target_col IS null for prediction
        predict_df = df_filled[df_filled[target_col].isna()]

        if predict_df.empty:
            print(f"No missing values to fill for {target_col}")
            continue

        # Features for training: all features + previously predicted target columns (up to current)
        current_features = feature_cols + target_cols_temperature[:i]

        
        train_df = train_df.dropna(subset=current_features + [target_col])

        if train_df.empty:
            print(f"No sufficient data to train model for {target_col}")
            continue

        X_train = train_df[current_features]
        y_train = train_df[target_col]
 
        predict_df = predict_df.dropna(subset=current_features)

        if predict_df.empty:
            print(f"No rows with enough features to predict for {target_col}")
            continue

        X_predict = predict_df[current_features]

        #her we Train the model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

       
        y_pred = model.predict(X_predict)

        # Fill predicted values back into the DataFrame
        df_filled.loc[X_predict.index, target_col] = y_pred

        print(f"Filled {len(y_pred)} missing values for {target_col}")

    return df_filled

 
def main_temperature_analysis(df):
    """
    Main function to run all steps for temperature analysis.

    Arguments given here are :
        df: Input DataFrame.
        feature_cols: List of features.
        target_cols: List of target columns.

    Returns:
        Final DataFrame with temperature predictions filled.
    """
    #df = encode_wind_direction(df) as the og wind directions are not suited for the model here we need numeric and cyclic 
    correlated_features = plot_temperature_correlation(df)
     
    Y,X = prepare_features_and_targets(df, correlated_features, target_cols_temperature)

    
    print(f'TARGET COLUMNS {Y} , FEATURE COLUMNS{X}')
    df_filled = iterative_fill(df,Y,X)

    print('RUNNN MAIN TEMOPERATURE ANALYSIS')
    return df_filled
 


#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
#HUMIDTY


# Define humidity target columns
target_cols_humidity = [
    "relative_humidity_2m_1h", "relative_humidity_2m_2h",
    "relative_humidity_2m_3h", "relative_humidity_2m_4h", "relative_humidity_2m_5h"
]
# Removed because of Conflict between FastApi and this 
# also we will be using stremlit in built charts as they are kinda more clean
def plot_humidity_correlation(df):
    """
    Plots a correlation heatmap for features related to humidity.

    Arguments:
        df: DataFrame containing numerical features and 'relative_humidity_2m'.

    Returns:
        List of features with high correlation to 'relative_humidity_2m'.

        
    # Plot correlation heatmap for 'relative_humidity_2m'
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        correlation_matrix[['relative_humidity_2m']].sort_values(by='relative_humidity_2m', ascending=False),
        annot=True,
        cmap='coolwarm',
        fmt='.2f'
    )
    plt.title("Feature Correlation with Relative Humidity")
    plt.show()

    
    
    """


    
    filtered_columns = [col for col in df.columns if not any(col.endswith(suffix) for suffix in ['1h', '2h', '3h', '4h', '5h'])]
    
    # Calculate and Filter correlations greater than the threshold
    correlation_matrix = df[filtered_columns].corr().fillna(0)

    
    threshold = 0.25
    correlated_features = correlation_matrix[
        abs(correlation_matrix['relative_humidity_2m']) > threshold
    ]['relative_humidity_2m'].index.tolist()
 

    
    print(f'\nCORRELATED FRATURES ARE {correlated_features}\n')
    print('RUNNN plot_humidity_correlation')




    return correlated_features 

def iterative_fill_humidity(df, target_cols, feature_cols):
    """
    Fills missing values in target_cols iteratively using RandomForestRegressor.

    Args:
        df: DataFrame containing original features and shifted target columns.
        target_cols: List of target columns to predict in order.
        feature_cols: List of feature columns to use for training and prediction.

    Returns:
        df_filled: DataFrame with missing values in target_cols filled iteratively.
    """
    df_filled = df.copy()

    for i, target_col in enumerate(target_cols):
        print(f"\nPredicting and filling missing values for: {target_col}")

        train_df = df_filled[df_filled[target_col].notna()]
        predict_df = df_filled[df_filled[target_col].isna()]

        if predict_df.empty:
            print(f"No missing values to fill for {target_col}")
            continue

        current_features = feature_cols + target_cols[:i]
        train_df = train_df.dropna(subset=current_features + [target_col])

        if train_df.empty:
            print(f"No sufficient data to train model for {target_col}")
            continue

        X_train = train_df[current_features]
        y_train = train_df[target_col]

        predict_df = predict_df.dropna(subset=current_features)

        if predict_df.empty:
            print(f"No rows with enough features to predict for {target_col}")
            continue

        X_predict = predict_df[current_features]
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_predict)
        df_filled.loc[X_predict.index, target_col] = y_pred
        print(f"Filled {len(y_pred)} missing values for {target_col}")

    return df_filled

def main_humidity(df):
    df_filled_humidity = iterative_fill_humidity(df, target_cols_humidity, plot_humidity_correlation(df))
    return df_filled_humidity


#---------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------
#RAINNFALL PROBABBILTY

chance_of_rain_cols = [
    "chance_of_rain_1h",
    "chance_of_rain_2h",
    "chance_of_rain_3h",
    "chance_of_rain_4h",
    "chance_of_rain_5h"
]  

def analyze_chance_of_rain_correlation(df):
    """
    Analyzes and plots correlations with the target column and identifies highly correlated features.

    Args:
        df: DataFrame containing numerical features and the target column.
        target_column: The column to analyze correlations for.
        threshold: Minimum absolute correlation value to filter for.

    Returns:
        List of features with high correlation to the target column.


        
    # Plot heatmap for the target column
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        correlation_matrix[[target_column]].sort_values(by=target_column, ascending=False),
        annot=True,
        cmap='coolwarm',
        fmt='.2f'
    )
    plt.title(f"Feature Correlation with {target_column.capitalize()}")
    plt.show()
    """

    target_column='chance_of_rain'
    threshold=0.25
     

    # Filter columns to exclude specific suffixes
    exclude_suffixes = ['1h', '2h', '3h', '4h', '5h']
    filtered_columns = [
        col for col in df.columns 
        if not any(col.endswith(suffix) for suffix in exclude_suffixes) and col != target_column
    ]

    # Calculate correlation matrix for filtered columns and get the ones with high coorelation 
    correlation_matrix = df[filtered_columns + [target_column]].corr().fillna(0)


    correlated_features = correlation_matrix[
        abs(correlation_matrix[target_column]) > threshold
    ][target_column].index.tolist()
 
    
    print(f'\nCORRELATED FEATURES ARE {correlated_features}\n')
    print('RUNNN analyze_chance_of_rain_correlation')

    return correlated_features

def iterative_fill_xgboost(df, target_cols, correlated_features):
    """
    Iteratively fills missing values using XGBRegressor.

    Args:
        df: DataFrame with missing values.
        target_cols: List of target columns to fill.
        correlated_features: List of correlated features to use for prediction.

    Returns:
        DataFrame with missing values filled.
    """
    
    

    df_filled = df.copy()

    for target_col in target_cols:
        print(f"Predicting and filling missing values for: {target_col}")

        # Ensure the target column exists in the DataFrame
        if target_col not in df.columns:
            print(f"Column {target_col} not found in DataFrame.")
            continue

        # Filter rows where the target column is not missing
        train_df = df_filled[df_filled[target_col].notna()]
        test_df = df_filled[df_filled[target_col].isna()]

        # Ensure no empty training set
        if train_df.empty:
            print(f"No non-missing values available for {target_col}. Skipping.")
            continue

        # Exclude datetime and non-numeric columns
        features = [col for col in correlated_features if col in df.columns and col != target_col]
        features = [col for col in features if df[col].dtype in ['int64', 'float64', 'bool']]

        # Prepare train and test data
        X_train = train_df[features]
        y_train = train_df[target_col]
        X_test = test_df[features]

        # Handle empty features or test sets
        if X_train.empty or X_test.empty:
            print(f"Insufficient data for filling {target_col}. Skipping.")
            continue

        # Initialize and fit the XGBRegressor model
        model = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
            enable_categorical=False  # Add `enable_categorical=True` if categorical data exists
        )
        model.fit(X_train, y_train)

        # Predict missing values
        y_pred = model.predict(X_test)

        # Fill missing values
        df_filled.loc[test_df.index, target_col] = y_pred

        print(f"Filled {len(y_pred)} missing values for {target_col}")

        for col in target_cols:
         if col in df_filled.columns:
            df_filled[col] = df_filled[col].clip(upper=100)
            print(f"Capped values in column '{col}' at 100.")

    return df_filled

def main_rain_chance(df):
    df_filled_rain = iterative_fill_xgboost(df, chance_of_rain_cols, analyze_chance_of_rain_correlation(df))
    return df_filled_rain



#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#COMBINE RESULTS
from fastapi.responses import JSONResponse
from fastapi import FastAPI
from pydantic import BaseModel
import asyncio

app = FastAPI()

def super_call(latitude, longitude):
    # Heavy computation or data processing
    df = main(latitude, longitude)
    
    # Step 1: Separate non-numeric 'datetime' column
    datetime_column = df['datetime']  # we will remove it for now and use it later (probably i guess)
    df_numeric = df.drop(columns=['datetime'])  
    
    # Step 2: Perform the processing on numeric data
    df_filled_temp = main_temperature_analysis(df_numeric)
    df_filled_humi = main_humidity(df_filled_temp)
    df_filled_chance = main_rain_chance(df_filled_humi)
    
    # Step 3: Combine processed data with the original 'datetime' column
    df_processed = pd.concat([datetime_column, df_filled_chance], axis=1)
    
    print(df_processed.head())  
    
    return df_processed


# FAST API ZONE ---------starts


# Define the schema for the request body
class Coordinates(BaseModel):
    latitude: float
    longitude: float


@app.post("/process-data/")
async def process_data(data: Coordinates):
    # Run the blocking `super_call` in a separate thread
    df_final = await asyncio.to_thread(super_call, data.latitude, data.longitude)
    print(df_final.columns)

    #Convert the datetime column to string format before returning as this is causing us mannnnnnnnny errors 
    df_final["datetime"] = df_final["datetime"].astype(str)

    # Send the JSON response
    return JSONResponse(content=df_final.to_dict(orient="records"))
    
     


 

