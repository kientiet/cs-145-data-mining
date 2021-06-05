# <codecell> Loading libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense, Dropout
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# <codecell> Loading data
train_vaccine = pd.read_csv('train_vaccine.csv')
train_trendency = pd.read_csv('train_trendency.csv')
test = pd.read_csv('test.csv')
starting_forecast_date = pd.to_datetime('2021-04-01', format='%Y-%m-%d')
forecast_period = pd.date_range(starting_forecast_date, periods = 30, freq='1d').tolist()

# <codecell> Convert date to correct format
train_trendency['Date'] = pd.to_datetime(train_trendency['Date'])
train_vaccine['date'] = pd.to_datetime(train_vaccine['date'])
test['Date'] = pd.to_datetime(test['Date'])

# <codecell> Chaging date column in train_vaccine to Date
renamed_vaccine = train_vaccine.rename(columns={'date': 'Date', 'location':'Province_State'})

# <codecell> Dropping 'Recovered' and 'Active' since there are too many missing data
trendency = train_trendency.drop(columns = ['Recovered', 'Active'])

# <codecell> Merging train_trendency and train_vaccine
merged_trendency_vaccine = pd.merge(trendency, renamed_vaccine, on = ['Province_State', 'Date'])
merged_df = merged_trendency_vaccine.drop(columns = ['Unnamed: 0_y', 'Unnamed: 0_x'])

# <codecell> Correlation matrix for an example state Alabama
example = merged_df[merged_df['Province_State'] == 'California']
corrMatrix = example.corr()
sns.heatmap(corrMatrix, annot = True)
plt.show()
## we can see that all the variables are highly correlated with each other so it would 
## not be helpful to include others variables in our model

# <codecell> Forecasting function
def forecast_function(state, attribute, validate, plotting):
    state_filtered = train_trendency[train_trendency['Province_State'] == state]
    df = pd.concat([state_filtered[['Date', attribute]], state_filtered[attribute].shift()], axis = 1)
    df.columns = ['Date', attribute, 'Lagged']
    df['New_Cases'] = df[attribute] - df['Lagged']
    df.dropna(axis = 0, inplace = True)
    df.reset_index(drop=True, inplace = True)
    
    #-----PLOTTING-----#
    if(plotting):
        plt.figure(figsize=(10,6))
        plt.plot(df['Date'], df[attribute], label = 'Cumulative', color = 'blue')
        plt.plot(df['Date'], df['New_Cases'], label = 'New Cases', color = 'red')
        plt.legend(loc='upper right')
        plt.show()
        
        plt.figure(figsize=(10,6))
        plt.plot(df['Date'], df['New_Cases'], label = 'New Cases', color = 'red')
        plt.legend(loc = 'upper right')
        plt.show()
    
    # Scaling data
    scaler = StandardScaler()
    scaler = scaler.fit(df['New_Cases'].to_frame())
    scaled_df = scaler.transform(df['New_Cases'].to_frame())
    
    # Reshaping data
    X = []
    Y = []
    
    obs_point = 15
    for i in range(obs_point, len(scaled_df)):
        X.append(scaled_df[i - obs_point: i, :])
        Y.append(scaled_df[i:i + 1, 0])
        
    X = np.array(X)
    Y = np.array(Y)
    
    # Splitting Train and Test sets
    test_ratio = 3/10
    test_size = int(len(X) * test_ratio)
    train_size = len(X) - test_size
    trainX, trainY = X[0:train_size, :, :], Y[0:train_size]
    testX, testY = X[train_size: len(X), :, :], Y[train_size: len(X)]
    
    # Model Validation
    if (validate):
        # Initializing LSTM
        model = Sequential()
        model.add(LSTM(64, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True))
        model.add(LSTM(32, activation='relu', return_sequences=True))
        model.add(LSTM(32, activation='relu', return_sequences=False))
        model.add(Dense(32))
        model.add(Dropout(0.2))
        model.add(Dense(trainY.shape[1]))
        model.compile(optimizer='adam', loss='mse')
        model.summary()
        
        # Fitting model
        history = model.fit(trainX, trainY, epochs=76, batch_size=30, validation_split=0.2, verbose=1)
        
        # Model Validation
        y_pred_train_scaled = model.predict(trainX)
        y_pred_train = np.floor(scaler.inverse_transform(y_pred_train_scaled))
        y_pred_test_scaled = model.predict(testX)
        y_pred_test = np.floor(scaler.inverse_transform(y_pred_test_scaled))
    
    # Reinitializing LSTM
    model = Sequential()
    model.add(LSTM(64, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True))
    model.add(LSTM(32, activation='relu', return_sequences=True))
    model.add(LSTM(32, activation='relu', return_sequences=False))
    model.add(Dense(32))
    model.add(Dropout(0.2))
    model.add(Dense(trainY.shape[1]))
    model.compile(optimizer='adam', loss='mse')
    model.summary()
    
    # Forecating
    ## Retraining model
    history = model.fit(X, Y, epochs=76, batch_size=30, validation_split=0.2, verbose=1)
    
    ## Initialization
    days_to_forecast = 30
    check = []
    forecastX = X[-1]
    forecastY = Y[-1]
    forecast_scaled = []
    ## Forecasting
    for i in range(days_to_forecast):
        forecastX = np.roll(forecastX, -1)
        check.append(forecastX)
        forecastX[-1] = forecastY
        forecastY = model.predict(np.array([forecastX]))
        forecast_scaled.append(forecastY.flatten())
    
    forecast = np.floor(scaler.inverse_transform(forecast_scaled))
    forecast = np.where(forecast < 0, 0, forecast)
    ## Validation
    y_pred_final_train_scaled = model.predict(X)
    y_pred_final_train = np.floor(scaler.inverse_transform(y_pred_final_train_scaled))
    
    #-----PLOTTING-----#
    if (plotting):
        starting_train_date = df['Date'][0] + pd.DateOffset(days=obs_point)
        train_period = pd.date_range(starting_train_date, periods = len(trainX), freq='1d').tolist()
        starting_test_date = starting_train_date + pd.DateOffset(days=len(trainX))
        test_period = pd.date_range(starting_test_date, periods = len(testX), freq='1d').tolist()
        final_train_period = pd.date_range(starting_train_date, periods = len(X), freq='1d').tolist()
        
        plt.figure(figsize=(10,6))
        plt.plot(df['Date'], df['New_Cases'], label = 'Actual', color = 'black')
        plt.plot(train_period, y_pred_train, label = 'Train', color = 'blue')
        plt.plot(test_period, y_pred_test, label = 'Test', color = 'red')
        plt.plot(forecast_period, forecast, label = 'Forecast', color = 'green')
        plt.plot(final_train_period, y_pred_final_train, label = 'Training for forecasting', color = 'yellow')
        plt.legend(loc = 'upper right')
        plt.show()
    
    # Merging to Submission
    cumulative = df[attribute].iloc[-1]
    submission = []
    for forecast_case in forecast:
        cumulative += int(forecast_case)
        submission.append(cumulative)
    
    return np.array(submission)


#temp = forecast_function('Alabama', 'Deaths', True, True)


# <codecell> Making Submission
attribute_to_forecast = ['Confirmed','Deaths']
states = train_trendency['Province_State'].unique().tolist()
submission_df = pd.DataFrame(columns = ['Province_State', 'Date', 'Confirmed', 'Deaths'])
for state in states:
    state_submission = pd.DataFrame({'Province_State': [state]*30, 'Date': forecast_period})
    for attribute in attribute_to_forecast:
        state_submission[attribute] = forecast_function(state, attribute, False, False)
    submission_df = pd.concat([submission_df, state_submission])

# <codecell> Death
final_df = pd.merge(test.drop(columns = ['Unnamed: 0', 'Confirmed', 'Deaths']), submission_df, on = ['Province_State', 'Date'])
final_df.index.name = 'ID'
submission = final_df.drop(columns = ['Province_State', 'Date'])

# <codecell> Final clean
submission.to_csv('sondang_v1.csv')







    









