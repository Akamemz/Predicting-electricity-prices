The main project file is the FTP_Tim.py file. Data file is 'electricity_prices.csv'
Note that .py file is calling dataset from my open GitHub repository. If you encounter a problem in reading a file please uncomment this line of code: 

elec_df = pd.read_csv('electricity_prices.csv')


====================================================================================
The toolbox.py or such file not used. All functions are presented in the FTP_Tim.py file.

We can just run FTP_Tim.py. But pleas commend out the lines 1099 till the end or simply put break point on this line of code:

model_SARIMAX = sm.tsa.SARIMAX(y_train_df['SMPEP2'], exog=X_train, order=(1, 0, 0), seasonal_order=(6, 1, 1, 12)).fit()
====================================================================================

As this code line will run SARIMAX function which is very demanding on CPU and it might take a while for it to finish. So it's recommended to not have multiple applications running when you execute this command. 

Run it when you are ready. Also it's best if you run it sections by sections to keep track of python file.  



