# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf

# df.to_csv('df_sample.csv', index=True)


# %% [markdown]
# ## Description of the dataset

# %%
"""
Column Description:

DateTime:               String, defines date and time of sample
Holiday:                String, gives name of holiday if day is a bank holiday
HolidayFlag:            integer, 1 if day is a bank holiday, zero otherwise
DayOfWeek:              integer (0-6), 0 monday, day of week
WeekOfYear:             integer, running week within year of this date
Day integer:            day of the date
Month integer:          month of the date
Year integer:           year of the date
PeriodOfDay integer:    denotes half hour period of day (0-47)
ForecastWindProduction: the forecasted wind production for this period
SystemLoadEA:           the national load forecast for this period
SMPEA:                  the price forecast for this period
ORKTemperature:         the actual temperature measured at Cork airport
ORKWindspeed:           the actual windspeed measured at Cork airport
CO2Intensity:           the actual CO2 intensity in (g/kWh) for the electricity produced
ActualWindProduction:   the actual wind energy production for this period
SystemLoadEP2:          the actual national system load for this period
SMPEP2:                 the actual price of this time period, the value to be forecasted

"""
elec_df = pd.read_csv('https://raw.githubusercontent.com/Akamemz/Time-Series-Term-Project/main/Term%20Project/electricity_prices.csv', low_memory=False)
# elec_df = pd.read_csv('electricity_prices.csv')

df = elec_df.copy()
df.info()

# %%
# Format 'DateTime' feature into date and set indexes of data frame as formatted DateTime
df['DateTime'] = pd.to_datetime(df['DateTime'], format='%d/%m/%Y %H:%M')
df.set_index('DateTime', inplace=True)
# Conversion of features from object type to numerical
df['ForecastWindProduction'] = pd.to_numeric(df['ForecastWindProduction'], errors='coerce')
df['SystemLoadEA'] = pd.to_numeric(df['SystemLoadEA'], errors='coerce')
df['SMPEA'] = pd.to_numeric(df['SMPEA'], errors='coerce')
df['ORKTemperature'] = pd.to_numeric(df['ORKTemperature'], errors='coerce')
df['ORKWindspeed'] = pd.to_numeric(df['ORKWindspeed'], errors='coerce')
df['CO2Intensity'] = pd.to_numeric(df['CO2Intensity'], errors='coerce')
df['ActualWindProduction'] = pd.to_numeric(df['ActualWindProduction'], errors='coerce')
df['SystemLoadEP2'] = pd.to_numeric(df['SystemLoadEP2'], errors='coerce')
df['SMPEP2'] = pd.to_numeric(df['SMPEP2'], errors='coerce')


# For now we can consider to drop 'Holiday' feature as we have 'HolidayFlag' as it simply indicates
# wether a day is holiday or not 
print(df['Holiday'].unique())

# Also we do not need columns 'Day', 'Month', 'Year', 'WeekOfYear', 'PeriodOfDay' as they do not
# carry any important information as of now
df = df.drop(['Day', 'DayOfWeek', 'Month', 'Year', 'WeekOfYear', 'PeriodOfDay', 'Holiday'], axis=1)


# %%
# Check for missing values in the entire dataset
missing_values = df.isnull().sum()

print("Missing values in the entire dataset:")
print(missing_values)

# %%
# Look up missing values by row location
# For ORKTemperature and ORKWindspeed will need to locate from 2011-12-24 23:00:00':'2011-12-25 07:00:00
# becasue there are big gaps
missing_smpep2 = df[df['ORKTemperature'].isna()]
print(missing_smpep2)

# %%
# Using this code we will be able to see exatly the rows of missing values of a given feature 
# df.loc['2013-03-31 00:00:00':'2013-03-31 23:30:00', 'SystemLoadEA']

# %%
resampled_data = df.resample('2H').mean()

# %%
resampled_data.info()

#Look up missing values location based on indexes
missing_smpep3 = resampled_data[resampled_data['ORKTemperature'].isna()]

#%%
# =========================================
# Filling out ORKTemperature
# =========================================

# Your provided indexes
indexes_to_fill = [
    '2011-12-25 02:00:00', '2011-12-25 04:00:00', '2011-12-25 08:00:00',
    '2011-12-25 10:00:00', '2011-12-25 14:00:00', '2011-12-25 16:00:00',
    '2011-12-25 20:00:00', '2011-12-25 22:00:00']

resampled_data.loc[indexes_to_fill, 'ORKTemperature'] = 5.28592698
print(resampled_data.loc['2011-12-25 00:00:00':'2011-12-25 22:00:00', 'ORKTemperature'])

# Filling out ORKTemperature for 2011-12-26 02:00:00
# Your provided indexes
indexes_to_fill = ['2011-12-26 02:00:00']

# Filling missing values in 'ORKTemperature' column with 7.2610694 based on indexes
resampled_data.loc[indexes_to_fill, 'ORKTemperature'] = 7.2610694
print(resampled_data.loc['2011-12-26 00:00:00':'2011-12-26 22:00:00', 'ORKTemperature'])

# Filling out ORKTemperature for 2012-04-19 16:00, 20:00, 22:00
# Your provided indexes
indexes_to_fill = ['2012-04-19 16:00:00', '2012-04-19 20:00:00', '2012-04-19 22:00:00']

resampled_data.loc[indexes_to_fill, 'ORKTemperature'] = 7.09574397
print(resampled_data.loc['2012-04-19 00:00:00':'2012-04-19 22:00:00', 'ORKTemperature'])

# Filling out ORKTemperature for 2012-12-17 10:00, 12:00, 14:00
# Your provided indexes
indexes_to_fill = ['2012-12-17 10:00:00', '2012-12-17 12:00:00', '2012-12-17 14:00:00']

resampled_data.loc[indexes_to_fill, 'ORKTemperature'] = 6.3469697
print(resampled_data.loc['2012-12-17 08:00:00':'2012-12-17 22:00:00', 'ORKTemperature'])

# Filling out ORKTemperature for 2012-12-25
# Your provided indexes
indexes_to_fill = [
    '2012-12-25 08:00:00', '2012-12-25 10:00:00', '2012-12-25 14:00:00',
    '2012-12-25 16:00:00', '2012-12-25 20:00:00', '2012-12-25 22:00:00']


resampled_data.loc[indexes_to_fill, 'ORKTemperature'] = 7.9162682
print(resampled_data.loc['2012-12-25 8:00:00':'2012-12-25 22:00:00', 'ORKTemperature'])

# =========================================
# Filling out ORKWindspeed for 2011-12-25
# =========================================

# Your provided indexes
indexes_to_fill = [
    '2011-12-25 02:00:00', '2011-12-25 04:00:00', '2011-12-25 08:00:00',
    '2011-12-25 10:00:00', '2011-12-25 14:00:00', '2011-12-25 16:00:00',
    '2011-12-25 20:00:00', '2011-12-25 22:00:00']


resampled_data.loc[indexes_to_fill, 'ORKWindspeed'] = 18.8642205
print(resampled_data.loc['2011-12-25 00:00:00':'2011-12-25 22:00:00', 'ORKWindspeed'])

# Filling out ORKWindspeed for 2011-12-26 02:00:00
# Your provided indexes
indexes_to_fill = ['2011-12-26 02:00:00']

# Filling missing values in 'ORKWindspeed' column with 23.0460827 based on indexes
resampled_data.loc[indexes_to_fill, 'ORKWindspeed'] = 23.0460827
print(resampled_data.loc['2011-12-26 00:00:00':'2011-12-26 22:00:00', 'ORKWindspeed'])

# Filling out ORKWindspeed for 2012-04-19 16:00, 20:00, 22:00
# Your provided indexes
indexes_to_fill = ['2012-04-19 16:00:00', '2012-04-19 20:00:00', '2012-04-19 22:00:00']

resampled_data.loc[indexes_to_fill, 'ORKWindspeed'] = 33.8796336
print(resampled_data.loc['2012-04-19 00:00:00':'2012-04-19 22:00:00', 'ORKWindspeed'])

# Filling out ORKTemperature for 2012-12-17 10:00, 12:00, 14:00
# Your provided indexes
indexes_to_fill = ['2012-12-17 10:00:00', '2012-12-17 12:00:00', '2012-12-17 14:00:00']

resampled_data.loc[indexes_to_fill, 'ORKWindspeed'] = 18.2460086
print(resampled_data.loc['2012-12-17 08:00:00':'2012-12-17 22:00:00', 'ORKWindspeed'])

# Filling out ORKWindspeed for 2012-12-25
# Your provided indexes
indexes_to_fill = [
    '2012-12-25 08:00:00', '2012-12-25 10:00:00', '2012-12-25 14:00:00',
    '2012-12-25 16:00:00', '2012-12-25 20:00:00', '2012-12-25 22:00:00']


resampled_data.loc[indexes_to_fill, 'ORKWindspeed'] = 20.6442546
print(resampled_data.loc['2012-12-25 8:00:00':'2012-12-25 22:00:00', 'ORKWindspeed'])

# %%
# Check for missing values in the entire dataset
missing_values = resampled_data.isnull().sum()

print("Missing values in the entire dataset:")
print(missing_values)


# %%
# Double cheking Data set information
resampled_data.info()

# Converting HolidayFlag to category
resampled_data['HolidayFlag'] = resampled_data['HolidayFlag'].astype('category')
resampled_data.info()

# %%
# Pearson’s Correlation matrix
plt.figure(figsize=(13, 7))
sns.heatmap(resampled_data[['ForecastWindProduction', 'SystemLoadEA', 'SMPEA', 'ORKTemperature', 'ORKWindspeed', 'CO2Intensity', 'ActualWindProduction',
                          'SystemLoadEP2', 'SMPEP2']].corr(), annot=True)
plt.xticks(rotation=45)
plt.title('Features Correlation matrix', fontdict={'fontsize': 15}, pad=10)
plt.tight_layout()
plt.show()


# %%
# Time Series Plot
plt.figure(figsize=(15, 6))
resampled_data['SMPEP2'].plot()
#sns.lineplot(data=resampled_data, x=resampled_data.index, y=resampled_data['SMPEP2'])
plt.title('Price of electricity (¢/kWh)')
plt.tight_layout()
plt.grid()
plt.show()

# %% [markdown]
# ## Stationarity

# %%
# ACF of Dependent variable

def ACF_PACF_Plot(y, lags):
    acf = sm.tsa.stattools.acf(y, nlags=lags)
    pacf = sm.tsa.stattools.pacf(y, nlags=lags)
    fig = plt.figure(figsize=(11, 5))
    plt.subplot(211)
    plt.title('ACF/PACF of the raw data')
    plot_acf(y, ax=plt.gca(), lags=lags)
    plt.subplot(212)
    plot_pacf(y, ax=plt.gca(), lags=lags)
    fig.tight_layout(pad=3)
    plt.show()

print(ACF_PACF_Plot(resampled_data['SMPEP2'], lags=100))

# %%
# Plots for Rolling mean and Rolling variance 
def Cal_rolling_mean_var(df, variable):

    # Create a subplot
    fig, axs = plt.subplots(2, 1, figsize=(11, 5))

    # Initialize empty lists to store rolling means and variances
    rolling_means = []
    rolling_variances = []

    # Loop the number of observations in the dataset
    for i in range(1, len(df) + 1):

        # Calculation for the rolling mean
        rolling_mean = df[variable].head(i).mean()
        rolling_means.append(rolling_mean)

        # Calculation for the rolling variance
        rolling_variance = df[variable].head(i).var()
        rolling_variances.append(rolling_variance)

    # Plot for rolling mean
    axs[0].plot(rolling_means)
    axs[0].set_title(f"Rolling Mean -{variable}")
    axs[0].set(xlabel='Sample', ylabel='Magnitude')

    # Plot for rolling variance
    axs[1].plot(rolling_variances, label='Varying variance')
    axs[1].set_title(f"Rolling Variance -{variable}")
    axs[1].set(xlabel='Sample', ylabel='Magnitude')
    axs[1].legend(loc='lower right')

    plt.tight_layout()
    plt.show()


print(Cal_rolling_mean_var(resampled_data, 'SMPEP2'))

# %%
# Chech for stationarity using ADF test

def ADF_Cal(x):
     result = adfuller(x)
     print("ADF Statistic: %f" %result[0])
     print('p-value: %f' % result[1])
     print('Critical Values:')
     for key, value in result[4].items():
         print('\t%s: %.3f' % (key, value))

print('*==========================*')
print('ADF test for SMPEP2:\n')
print(ADF_Cal(resampled_data['SMPEP2']))
print('*==========================*')

# Chech for stationarity using KPSS test

def kpss_test(timeseries):
     print('Results of KPSS Test:')
     kpsstest = kpss(timeseries, regression='c', nlags="auto")
     kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic','p-value','Lags Used'])
     for key,value in kpsstest[3].items(): kpss_output['Critical Value (%s)'%key] = value
     print (kpss_output)

print('*==========================*')
print('KPSS test for SMPEP2:\n')
print(kpss_test(resampled_data['SMPEP2']))
print('*==========================*')

# Resluts show that according to ADF test dependent variabel is stationary where as to KPSS its not.
# In conclusion we can say that my dependent feature is Stationary according to ADF test and Rolling mean & Rolling variavnce.

# %% [markdown]
# ## Time series Decomposition
from statsmodels.tsa.seasonal import STL
# %%

# Time series Decomposition
STL = STL(resampled_data['SMPEP2'], period=12) # also 24(daily) & 168(weekly)
res = STL.fit()
plt.figure(figsize=(13, 8))
res.plot()
plt.tight_layout()
plt.show()


T = res.trend
S = res.seasonal
R = res.resid


def str_trend_seasonal(T, S, R):
    F = np.maximum(0, 1 - np.var(np.array(R)) / np.var(np.array(T + R)))
    print(f'The strength of trend for this data set is {100 * F:.2f}%')

    FS = np.maximum(0, 1 - np.var(np.array(R)) / np.var(np.array(S + R)))
    print(f'The strength of seasonality for this data set is {100 * FS:.2f}%')


str_trend_seasonal(T, S, R)

# %%
# Getting rid of seasonality
df_clean = resampled_data[['SMPEP2']].copy()

#%%
# ===============================================================
# Subtract the seasonal component from SMPEP2 and create a new column 'SMPEP2' in df_clean
df_clean['SMPEP2'] = resampled_data['SMPEP2'] - res.seasonal
df_clean['SMPEP2_trend'] = resampled_data['SMPEP2'] - res.trend
#%%
plt.figure(figsize=(13, 6))
plt.plot(resampled_data['SMPEP2'], label='Original')
plt.plot(df_clean['SMPEP2'], label='Seasonally Adj')
plt.title('Seasonally adjusted data Vs Original')
plt.xlabel('Date')
plt.ylabel('SMPEP2')
plt.legend(loc='upper left')
plt.tight_layout()
plt.grid()
plt.show()

plt.figure(figsize=(13, 6))
plt.plot(resampled_data['SMPEP2'], label='Original')
plt.plot(df_clean['SMPEP2_trend'], label='Trend Adj')
plt.title('Trend adjusted data Vs Original')
plt.xlabel('Date')
plt.ylabel('SMPEP2')
plt.legend(loc='upper left')
plt.tight_layout()
plt.grid()
plt.show()

# %%
# print(ACF_PACF_Plot(df_clean['SMPEP2'], lags=100))
# Note for self. Try different lags like 160 to see hidden patterns
# ADF_Cal(df_clean['SMPEP2'])
# Cal_rolling_mean_var(df_clean, 'SMPEP2')




# %% [markdown]
# ## Feature selection/dimensionality reduction
# ##            &&
# ## Holt-Winters method


# %%
# Standardizing dataset

# Standardizing dataset
resampled_data_numOnly = resampled_data.drop(columns=['HolidayFlag'])
resampled_data_std = (resampled_data_numOnly - resampled_data_numOnly.mean()) / resampled_data_numOnly.std()


resampled_data_std['HolidayFlag'] = resampled_data['HolidayFlag']


# %%

Y = resampled_data_std['SMPEP2']
X = resampled_data_std.drop('SMPEP2', axis=1)
X = sm.add_constant(X)
X_train, X_test, y_train, y_test = train_test_split(X, Y, shuffle=False, test_size=0.2)
print(f"The shapes of X_train:{X_train.shape} and y_train:{y_train.shape}")
print(f"The shapes of X_test:{X_test.shape} and y_test:{y_test.shape}")



#%%
# SVD decomposition

# a. Perform SVD analysis on the original feature space and write down your observation if co-linearity exists
U, S, V = np.linalg.svd(X)
print('Singular values:\n', S)
print('\nThis disparity in magnitude suggests that the first few variables or components might be highly influential '
      'or capture a substantial amount of variance in the data compared to the rest.')

print()
# b. Calculate the condition number and write down your observation if co-linearity exists.
print(f'Condition number of X is = {round(np.linalg.cond(X), 3)}')
print("The output value of condition number of X is acceptable and might not severely affect the model's stability or coefficient estimates")

# c. If collinearity exist, how many features will you remove to avoid the co-linearity?
print('\nThe decreasing pattern of the singular values suggests decreasing importance or strength. It seems there are '
      '\npotentially 8 significant dimensions or directions that capture most of the variance in dataset'
      '\nas the last value is close to 0')

# %%
# Fit the OLS model
model = sm.OLS(y_train, X_train).fit()

X_vif = X_train
X_vif_test = X_test
X_train_copy = X_train
X_test_copy = X_test


# %%
# Backward stepwise regression feature selection

# 1st iteration
model = sm.OLS(y_train, X_train).fit()
print(model.summary())


y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print("Mean Squared Error (MSE):", round(mse, 3))
print("Root Mean Squared Error (RMSE):", round(rmse, 3))



# %%

# 2nd iteration
X_train_copy = X_train_copy.drop('ORKWindspeed', axis=1)
X_test_copy = X_test_copy.drop('ORKWindspeed', axis=1)
model = sm.OLS(y_train, X_train_copy).fit()
print(model.summary())

y_pred = model.predict(X_test_copy)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print("Mean Squared Error (MSE):", round(mse, 3))
print("Root Mean Squared Error (RMSE):", round(rmse, 3))


#%%

# 3rd iteration
X_train_copy = X_train_copy.drop('ORKTemperature', axis=1)
X_test_copy = X_test_copy.drop('ORKTemperature', axis=1)
model = sm.OLS(y_train, X_train_copy).fit()
print(model.summary())

y_pred = model.predict(X_test_copy)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print("Mean Squared Error (MSE):", round(mse, 3))
print("Root Mean Squared Error (RMSE):", round(rmse, 3))


#%%
#4th iteration
X_train_copy = X_train_copy.drop('CO2Intensity', axis=1)
X_test_copy = X_test_copy.drop('CO2Intensity', axis=1)
model = sm.OLS(y_train, X_train_copy).fit()
print(model.summary())


y_pred = model.predict(X_test_copy)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print("Mean Squared Error (MSE):", round(mse, 3))
print("Root Mean Squared Error (RMSE):", round(rmse, 3))

U, S, V = np.linalg.svd(X_train_copy)
print('Singular values:\n', S)
print(f'Condition number of X is = {round(np.linalg.cond(X_train_copy), 3)}')

# %%
# VIF dataframe
vif_data = pd.DataFrame()
vif_data['feature'] = X_vif.columns
# calculating VIF for each feature
vif_data['VIF'] = [variance_inflation_factor(X_vif.values, i) for i in range(len(X_vif.columns))]

print(vif_data)
model_vif = sm.OLS(y_train, X_vif).fit()

print('From the VIF definition we know that if feature has VIF value more that 10 than its a clear indicator of '
      '\nmulticollinearity.')

print()


# %%
# 1st iteration
print(model_vif.summary())

y_pred = model_vif.predict(X_vif_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print("Mean Squared Error (MSE):", round(mse, 3))
print("Root Mean Squared Error (RMSE):", round(rmse, 3))



# %%

# 2nd iteration
X_vif = X_vif.drop('SystemLoadEA', axis=1)
X_vif_test = X_vif_test.drop('SystemLoadEA', axis=1)
model_vif = sm.OLS(y_train, X_vif).fit()
print(model_vif.summary())


y_pred = model_vif.predict(X_vif_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print("Mean Squared Error (MSE):", round(mse, 3))
print("Root Mean Squared Error (RMSE):", round(rmse, 3))




# %%
# Recalculate VIF score based on new model
# VIF dataframe
vif_data = pd.DataFrame()
vif_data['feature'] = X_vif.columns
# calculating VIF for each feature
vif_data['VIF'] = [variance_inflation_factor(X_vif.values, i) for i in range(len(X_vif.columns))]

print(vif_data)



# %%

# 3rd iteration
X_vif = X_vif.drop('ActualWindProduction', axis=1)
X_vif_test = X_vif_test.drop('ActualWindProduction', axis=1)
model_vif = sm.OLS(y_train, X_vif).fit()
print(model_vif.summary())

y_pred = model_vif.predict(X_vif_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print("Mean Squared Error (MSE):", round(mse, 3))
print("Root Mean Squared Error (RMSE):", round(rmse, 3))

U, S, V = np.linalg.svd(X_vif_test)
print('Singular values:\n', S)
print(f'Condition number of X is = {round(np.linalg.cond(X_vif_test), 3)}')




# %%
# Recalculate VIF score based on new model
# VIF dataframe
vif_data = pd.DataFrame()
vif_data['feature'] = X_vif.columns
# calculating VIF for each feature
vif_data['VIF'] = [variance_inflation_factor(X_vif.values, i) for i in range(len(X_vif.columns))]

print(vif_data)
model_vif = sm.OLS(y_train, X_vif).fit()
print()


# %% [markdown]
# ## Base-models

# %%
# ==============================
# Split Data into train and Test
# ==============================

Y = resampled_data['SMPEP2']
X = resampled_data.drop('SMPEP2', axis=1)
X = sm.add_constant(X)
X_train, X_test, y_train, y_test = train_test_split(X, Y, shuffle=False, test_size=0.2)

y_train_ord = pd.DataFrame(y_train, columns=['SMPEP2'])
y_test_ord = pd.DataFrame(y_test, columns=['SMPEP2'])

# Cheking for stationarity of train and test datasets
print('Train')
print(Cal_rolling_mean_var(y_train_ord, 'SMPEP2'))
print(ADF_Cal(y_train_ord))
print(kpss_test(y_train_ord))
print('Test')
print(Cal_rolling_mean_var(y_test_ord, 'SMPEP2'))
print(ADF_Cal(y_test_ord))
print(kpss_test(y_test_ord))

#%%
# ==================
# Average
# ==================

# Calculations for Train
y_pred_train = [round(np.mean(y_train[:i]), 2) if i != 0 else None for i in range(len(y_train))]
e_train = [round(y_train[i] - y_pred_train[i], 2) if y_pred_train[i] is not None else None for i in range(len(y_train))]
e2_train = [round(e ** 2, 2) if e is not None else None for e in e_train]

# Calculations for Test
mean_y_train = round(np.mean(y_train), 2)
y_pred_test = [mean_y_train for _ in y_test]
e_test = [round(y_test[i] - mean_y_train, 2) for i in range(len(y_test))]
e2_test = [round(e ** 2, 2) for e in e_test]

# MSE
mse_train = round(np.mean(e2_train[2:]), 2)
mse_test = round(np.mean(e2_test), 2)
rmse_train = np.sqrt(mse_train)
rmse_test = np.sqrt(mse_test)
print("MSE of train data Average", mse_train)
print("MSE of test data Average", mse_test)
print("RMSE of train data Average:", round(rmse_train, 2))
print("RMSE of test data Average:", round(rmse_test, 2))


# Variance
var_train = round(np.var(e_train[2:]), 2)
var_test = round(np.var(e_test), 2)
print("Variance of train data Average", var_train)
print("Variance of test data Average", var_test)


#%%
# ==================
# Naïve Method
# ==================

# Calculations for Train Naïve
y_pred_train_naive = [y_train[i - 1] if i != 0 else None for i in range(len(y_train))]
e_train_naive = [round(y_train[i] - y_pred_train_naive[i], 2) if y_pred_train_naive[i] is not None else None for i in range(len(y_train))]
e2_train_naive = [round(e ** 2, 2) if e is not None else None for e in e_train_naive]
last_y_train = y_train[-1]

# Calculations for Test Naïve
y_pred_test_naive = [last_y_train] * len(y_test)
e_test_naive = [round(y_test[i] - last_y_train, 2) for i in range(len(y_test))]
e2_test_naive = [round(e ** 2, 2) for e in e_test_naive]

# MSE Naïve
mse_train_naive = round(np.mean(e2_train_naive[2:]), 2)
mse_test_naive = round(np.mean(e2_test_naive), 2)
rmse_train_naive = np.sqrt(mse_train_naive)
rmse_test_naive = np.sqrt(mse_test_naive)
print("MSE of train data Naïve", mse_train_naive)
print("MSE of test data Naïve", mse_test_naive)
print("RMSE of train data Naïve:", round(rmse_train_naive, 2))
print("RMSE of test data Naïve:", round(rmse_test_naive, 2))

# Variance Naïve
var_train_naive = round(np.var(e_train_naive[2:]), 2)
var_test_naive = round(np.var(e_test_naive), 2)
print("Variance of train data Naïve", var_train_naive)
print("Variance of test data Naïve", var_test_naive)


#%%
# ==================
# Drift Method
# ==================

# Calculations for Train Drift
y_pred_train_drift = [y_train[i - 1] + (y_train[i - 1] - y_train[0]) / (i - 1) if i > 1 else None for i in range(len(y_train))]
e_train_drift = [round(y_train[i] - y_pred_train_drift[i], 2) if y_pred_train_drift[i] is not None else None for i in range(len(y_train))]
e2_train_drift = [round(e ** 2, 2) if e is not None else None for e in e_train_drift]

# Calculations for Test Drift
last_y_train = y_train[-1]
y_pred_test_drift = [last_y_train + (i + 1) * (last_y_train - y_train[0]) / (len(y_train) - 1) for i in range(len(y_test))]
e_test_drift = [round(y_test[i] - y_pred_test_drift[i], 2) for i in range(len(y_test))]
e2_test_drift = [round(e ** 2, 2) for e in e_test_drift]


# MSE Drift
mse_train_drift = round(np.mean(e2_train_drift[2:]), 2)
mse_test_drift = round(np.mean(e2_test_drift), 2)
rmse_train_drift = np.sqrt(mse_test_drift)
rmse_test_drift = np.sqrt(mse_test_drift)
print("MSE of train data Drift", mse_train_drift)
print("MSE of test data Drift", mse_test_drift)
print("RMSE of train data Drift:", round(rmse_train_drift, 2))
print("RMSE of test data Drift:", round(rmse_test_drift, 2))

# Variance Drift
var_train_drift = round(np.var(e_train_drift[2:]), 2)
var_test_drift = round(np.var(e_test_drift), 2)
print("Variance of train data Drift", var_train_drift)
print("Variance of test data Drift", var_test_drift)


#%%
# ============================
# Simple Exponential Smoothing
# ============================
def calculate_ses_predictions(data, alpha):
    predictions = [data[0]]
    for i in range(1, len(data)):
        prediction = alpha * data[i - 1] + (1 - alpha) * predictions[-1]
        predictions.append(prediction)
    return predictions

def calculate_errors(true_values, predicted_values):
    errors = [round(true - pred, 2) for true, pred in zip(true_values, predicted_values)]
    squared_errors = [round(error ** 2, 2) for error in errors]
    return errors, squared_errors

alpha = 0.5

# Calculations for Train SES
y_pred_train_ses = calculate_ses_predictions(y_train, alpha)
e_train_ses, e2_train_ses = calculate_errors(y_train, y_pred_train_ses)

# Calculations for Test SES
last_y_train = y_train[-1]
y_pred_test_ses = [
    round(alpha * last_y_train + (1 - alpha) * y_pred_train_ses[-1], 2)
    for _ in y_test]

e_test_ses, e2_test_ses = calculate_errors(y_test, y_pred_test_ses)


# MSE SES
mse_train_ses = round(np.mean(e2_train_ses[2:]), 2)
mse_test_ses = round(np.mean(e2_test_ses), 2)
rmse_train_ses = np.sqrt(mse_test_drift)
rmse_test_ses = np.sqrt(mse_test_drift)
print("MSE of train data SES", mse_train_ses)
print("MSE of test data SES", mse_test_ses)
print("RMSE of train data SES:", round(rmse_train_ses, 2))
print("RMSE of test data SES:", round(rmse_test_ses, 2))

# Variance SES
var_train_ses = round(np.var(e_train_ses[2:]), 2)
var_test_ses = round(np.var(e_test_ses), 2)
print("Variance of train data SES", var_train_ses)
print("Variance of test data SES", var_test_ses)


plt.figure(figsize=(13, 6))
plt.plot(y_train.index, y_train, label="Training set")
plt.plot(y_test.index, y_test, label="Test set")
plt.plot(y_test.index, y_pred_test, label="Average Forecast", color="red")
plt.plot(y_test.index, y_pred_test_naive, label="Naïve Forecast", color="green")
plt.plot(y_test.index, y_pred_test_drift, label="Drift Forecast", color="lightblue")
plt.plot(y_test.index, y_pred_test_ses, label="SES Forecast", color="brown")
plt.title("Training set, Test set, and h-step Forecast")
plt.xlabel("Date")
plt.ylabel("SMPEP2")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

#%%
import statsmodels.tsa.holtwinters as ets

y_train_plus1 = y_train + 0.5
y_test_plus1 = y_test + 0.5
holtt = ets.ExponentialSmoothing(y_train_plus1, trend='multiplicative', damped_trend=True, seasonal='mul').fit()
holtf_pred = holtt.forecast(steps=len(y_test_plus1))
holtf_pred = holtf_pred - 1

mse_holt_mul = mean_squared_error(y_test, holtf_pred)
rmse_holt_mul = np.sqrt(mse_holt_mul)
residuals = y_test - holtf_pred
forecast_variance = np.var(residuals)
print(f"Holt Winters model MSE mul: {round(mse_holt_mul, 3)}")
print(f"Holt Winters model RMSE mul: {round(rmse_holt_mul, 3)}")
print(f"Holt Winters model Variance mul: {round(forecast_variance, 3)}")

plt.figure(figsize=(15, 6))
# plt.plot(y_train, label="Train")
# To plot only Test and Forecast comment out line above
plt.plot(y_test, label="Test")
plt.plot(holtf_pred, label="Forecast")

plt.legend(loc='upper left')
plt.title('Holt-Winters Forecast for SMPEP2 (Multiplicative model)')
plt.xlabel('Time')
plt.ylabel('SMPEP2')
plt.grid()
plt.tight_layout()
plt.show()

#%%
model = ExponentialSmoothing(y_train, seasonal='add', seasonal_periods=12).fit()
predictions = model.forecast(len(y_test))


# Plotting
plt.figure(figsize=(15, 6))
# plt.plot(y_train.index, y_train, label='Train')
# To plot only Test and Forecast comment out line above
plt.plot(y_test.index, y_test, label='Test')
plt.plot(y_test.index, predictions, label='Forecast')

plt.title('Holt-Winters Forecast for SMPEP2 (Additive model)')
plt.xlabel('Date')
plt.ylabel('SMPEP2')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()


mse_holt_add = mean_squared_error(y_test, predictions)
rmse_holt_add = np.sqrt(mse_holt_add)
residuals = y_test - predictions
forecast_variance = np.var(residuals)
print(f"Holt Winters model MSE add: {round(mse_holt_add, 3)}")
print(f"Holt Winters model RMSE add: {round(rmse_holt_add, 3)}")
print(f"Holt Winters model Variance add: {round(forecast_variance, 3)}")



# %% [markdown]
# ## The multiple linear regression model that represents the dataset
# SMPEP2 = const + ForecastWindProduction + SystemLoadEA + SMPEA + ORKTemperature + CO2Intensity + ActualWindProduction + SystemLoadEP2

# %%

# Saving mean and std for reversing scaling
original_mean = resampled_data_numOnly.mean()
original_std = resampled_data_numOnly.std()


Y = resampled_data_std['SMPEP2']
X = resampled_data_std.drop('SMPEP2', axis=1)
X = sm.add_constant(X)
X_train, X_test, y_train, y_test = train_test_split(X, Y, shuffle=False, test_size=0.2)

#%%
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
# ============================= # =============================# =============================
# Principal Component Analysis
# ============================= # =============================# =============================
num_components = 10  # Number of principal components because
pca = PCA(n_components=num_components)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Fit model on the transformed data
model = LinearRegression()
model.fit(X_train_pca, y_train)

# Predict on the test set
y_pred = model.predict(X_test_pca)
y_pred_original_scale = y_pred * original_std['SMPEP2'] + original_mean['SMPEP2']
y_test_original_scale = y_test * original_std['SMPEP2'] + original_mean['SMPEP2']
y_train_original_scale = y_train * original_std['SMPEP2'] + original_mean['SMPEP2']

# Evaluate the model
r_squared = r2_score(y_test_original_scale, y_pred_original_scale)
residuals = y_test_original_scale - y_pred_original_scale
residual_mse_pca = np.mean(residuals ** 2)
rmse_pca = np.sqrt(residual_mse_pca)
print(f"R-squared PCA: {round(r_squared, 3)}")
print(f"MSE PCA: {round(residual_mse_pca, 3)}")
print(f"RMSE PCA: {round(rmse_pca, 3)}")



explained_var = pca.explained_variance_ratio_
singular_values = pca.singular_values_
components = pca.components_


print("Explained Variance Ratio:", explained_var)
print("Singular Values:", singular_values)
# print("Principal Components:", components)

# Calculate the number of samples and number of principal components
n_samples = len(X_test)
n_components_used = num_components

# Adjusted R-squared
adj_r_squared = 1 - ((1 - r_squared) * (n_samples - 1) / (n_samples - n_components_used - 1))
print(f"Adjusted R-squared: {round(adj_r_squared, 3)}")

#%%
def auto_corr_func(data, lag, name_plot):
    y_head = data.mean()
    num_of_obs = len(data)
    ACF_lag = np.zeros(lag + 1)

    for k in range(lag + 1):  # Iterate through lags 0 to given lag
        for t in range(k, num_of_obs):
            ACF_lag[k] += (data[t] - y_head) * (data[t - k] - y_head)

        ACF_lag[k] /= np.sum((data - y_head) ** 2)

    # Create a mirrored ACF array
    mirror_ACF = np.concatenate((ACF_lag[::-1], ACF_lag[1:]))

    # Create lags for the x-axis
    mirror_lags = np.arange(-lag, lag + 1)
    # Calculate critical values for significance region
    m = 1.96 / np.sqrt(num_of_obs)

    # Plot the ACF
    (markers, stemlines, baseline) = plt.stem(mirror_lags, mirror_ACF, markerfmt= 'o', basefmt='gray')
    plt.setp(markers, color = 'red')
    plt.title(name_plot)
    plt.axhspan(-m, m, alpha = 0.2, color = 'blue')
    plt.xlabel('Lags')
    plt.ylabel('Magnitude')
    plt.tight_layout()
    plt.show()

    # Object that hold dictionary of ACF calculations for each lag rounded to 2 decimals
    # auto_corr_dict = {f'Lag {k}': round(value_acf, 2) for k, value_acf in enumerate(ACF_lag)}

    return # print(auto_corr_dict)

# Final OLS model
X_train = X_train.drop(columns=['ORKWindspeed', 'ORKTemperature', 'CO2Intensity'], axis=1)
X_test = X_test.drop(columns=['ORKWindspeed', 'ORKTemperature', 'CO2Intensity'], axis=1)
model = sm.OLS(y_train, X_train).fit()
print(model.summary())

y_pred = model.predict(X_test)

# Reverse scaling
y_pred_original_scale = y_pred * original_std['SMPEP2'] + original_mean['SMPEP2']
y_test_original_scale = y_test * original_std['SMPEP2'] + original_mean['SMPEP2']

residuals = y_test_original_scale - y_pred_original_scale


# ====================
# ACF of residuals
ACF_PACF_Plot(residuals, lags=100)
auto_corr_func(residuals, lag=100, name_plot='ACF of OLS residuals')

residual_mse_ols = np.mean(residuals ** 2)
rmse_ols = np.sqrt(residual_mse_ols)
residual_variance = np.var(residuals)
print('Residual MSE OLS:', round(residual_mse_ols, 3))
print('Residual RMSE OLS:', round(rmse_ols, 3))
print('Residual Variance:', round(residual_variance, 3))



# Plotting train, test, and predicted values
plt.figure(figsize=(15, 6))
plt.plot(y_train_original_scale.index, y_train_original_scale, label='Train')
plt.plot(y_test_original_scale.index, y_test_original_scale, label='Test')
plt.plot(y_test_original_scale.index, y_pred_original_scale, label='Predicted')
plt.xlabel('Date')
plt.ylabel('SMPEP2')
plt.title('OLS - Actual vs Predicted Values')
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()

#%%
# ========================
# Q value OLS
# ========================

Q = sm.stats.acorr_ljungbox(residuals, lags=[100], return_df=True)
lb_stat_value = Q.lb_stat.values[0]
print(f"Ljung-Box Q Statistic: {round(lb_stat_value, 3)}")





# %%
# ============================================
# Preliminary order determination SARIMAX
# ============================================

def calculate_value_for_gpac(ry, J, K):
    den = np.array([ry[np.abs(J + k - i)] for k in range(K) for i in range(K)]).reshape(K, K)
    col = np.array([ry[J+i+1] for i in range(K)])
    num = np.concatenate((den[:, :-1], col.reshape(-1, 1)), axis=1)
    return np.inf if np.linalg.det(den) == 0 else round(np.linalg.det(num)/np.linalg.det(den), 10)

def cal_GPAC(ry, J=7, K=7):
    gpac_arr = np.full((J, K), np.nan)  # Fill uninitialized values with NaN
    for k in range(1, K):
        for j in range(J):
            gpac_arr[j][k] = calculate_value_for_gpac(ry, j, k)
    gpac_arr = np.delete(gpac_arr, 0, axis=1)
    df = pd.DataFrame(gpac_arr, columns=list(range(1, K)), index=list(range(J)))

    plt.figure(figsize=(15, 15))
    sns.heatmap(df, annot=True, fmt='0.3f', linewidths=.5)
    plt.title('Generalized Partial Autocorrelation (GPAC) Table')
    plt.tight_layout()
    plt.show()
    # print(df)

def cal_GPAC_dfONLY(ry, J=7, K=7):
    gpac_arr = np.full((J, K), np.nan)  # Fill uninitialized values with NaN
    for k in range(1, K):
        for j in range(J):
            gpac_arr[j][k] = calculate_value_for_gpac(ry, j, k)
    gpac_arr = np.delete(gpac_arr, 0, axis=1)
    df = pd.DataFrame(gpac_arr, columns=list(range(1, K)), index=list(range(J)))

    return df




# %%

Y = resampled_data['SMPEP2']
X = resampled_data.drop('SMPEP2', axis=1)
X = sm.add_constant(X)
X_train, X_test, y_train, y_test = train_test_split(X, Y, shuffle=False, test_size=0.2)

# X_train = X_train.drop(columns=['ORKWindspeed', 'ORKTemperature', 'CO2Intensity'], axis=1)
# X_test = X_test.drop(columns=['ORKWindspeed', 'ORKTemperature', 'CO2Intensity'], axis=1)

y_train_df = pd.DataFrame(y_train, columns=['SMPEP2'])
y_test_df = pd.DataFrame(y_test, columns=['SMPEP2'])


# %%

ry = sm.tsa.stattools.acf(y_train_df, nlags=100)

cal_GPAC(ry, J=10, K=10)
# table_GPAC = cal_GPAC_dfONLY(ry, J=10, K=10)

# print(ACF_PACF_Plot(resampled_data['SMPEP2'], lags=50))

#%%
# model = sm.tsa.arima.ARIMA(resampled_data['SMPEP2'], order=(7, 0, 2), trend='n').fit()

#%%
# Seasonal differencing

def ses_diff(data, period):
    data = pd.DataFrame(data)
    diff_data = data.diff(periods=period)
    return diff_data[period:]

df_clean_12 = ses_diff(y_train_df,12)
print(ACF_PACF_Plot(df_clean_12, lags=50))
Cal_rolling_mean_var(df_clean_12, 'SMPEP2')
print(ADF_Cal(df_clean_12))  # stationary
print(kpss_test(df_clean_12))  # not stationary
auto_corr_func(df_clean_12['SMPEP2'], lag=25, name_plot='ACF')

ry = sm.tsa.stattools.acf(df_clean_12['SMPEP2'], nlags=100)

cal_GPAC(ry, J=10, K=10)
# table_GPAC = cal_GPAC_dfONLY(ry, J=7, K=7)

#%%
# Uncomment below function to run SARIMX model
# Note you need to manually submit order of AR and MA for seasonal and non-seasonal
# Or put a break point here as it will take
model_SARIMAX = sm.tsa.SARIMAX(y_train_df['SMPEP2'], exog=X_train, order=(1, 0, 0), seasonal_order=(6, 1, 1, 12)).fit()
print(model_SARIMAX.summary())


#%%
model_hat = model_SARIMAX.predict(start=0, end=len(y_train_df['SMPEP2'])-1, exog=X_train)
model_hat_pred = model_SARIMAX.forecast(steps=len(y_test_df),  exog=X_test)

#%%
residuals = y_train_df['SMPEP2'] - model_hat
residuals_pred = y_test_df['SMPEP2'] - model_hat_pred

print("Train dataset")
residual_variance = np.var(residuals)
print('Residual Variance:', round(residual_variance, 3))
residual_mean = np.mean(residuals)
print('Residual Mean:', round(residual_mean, 3))
residual_mse = np.mean(np.square(residuals))
print('Residual MSE:', round(residual_mse, 3))
print()
print('Test dataset')
residual_variance_pred = np.var(residuals_pred)
print('Residual Variance:', round(residual_variance_pred, 3))
residual_mean_pred = np.mean(residuals_pred)
print('Residual Mean:', round(residual_mean_pred, 3))
residual_mse_pred = np.mean(np.square(residuals_pred))
print('Residual MSE:', round(residual_mse_pred, 3))

cov_test = np.cov(residuals_pred)
print("Covariance of Test Residuals:", cov_test)



lags = 100

Q = sm.stats.acorr_ljungbox(residuals, lags=[80], return_df=True)
lb_stat_value = Q.lb_stat.values[0]
print(Q)
print(f"Ljung-Box Q Statistic: {round(lb_stat_value, 3)}")

# Change na and bn based on estimated orders manually
DOF = lags - 7 - 1
alfa = 0.01

from scipy.stats import chi2
chi_critical = chi2.ppf(1-alfa, DOF)
print('Chi critical:', round(chi_critical, 3))

if lb_stat_value < chi_critical:
    print("The residual is white ")
else:
    print("The residual is NOT white ")


print(ACF_PACF_Plot(residuals, lags=100))

auto_corr_func(residuals, lag=100, name_plot='residuals')


#%%
# Uncomment or comment below line of code to see the train, test, predicted
# or all at one in one plot

# y_test_df     y_train_df
plt.figure(figsize=(15, 6))
# plt.plot(y_train_df.index, y_train_df['SMPEP2'], label='Train')
# plt.plot(y_train_df.index, model_hat, label='Predicted', )
plt.plot(y_test_df.index, y_test_df['SMPEP2'], label='Test')
plt.plot(y_test_df.index, model_hat_pred, label='Predicted', )
plt.legend()
plt.title('SARIMAX Model Predicted vs True Values')
plt.xlabel('Time')
plt.ylabel('SMPEP2')
plt.grid()
plt.tight_layout()
plt.show()




#%%

# ============================
# Forecast function
# ============================












