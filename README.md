# WIKI-GOOGLE-stock-prediction-using-LinearRegression
We use scikit learns linear regression to predict google stock prices

Linear Regression is a very simple algorith that cannot actually figure out  the relation between the base features or columns given in the Raw Dataset.

So we define those relations by defining variables  HL_PCT which is high to low perecentage and PCT_CNG which is the percentage change in stock price in a day.
We use these features to predict the price.

We define a column called 'label' which is shifted version of Adj. Close. So it is the price of stock somedays into future. The no of days can be figured out by forecast_out variable.

n_jobs = -1 implies that we can thread the operation and the maximum number of computations(possible by your system) are done at once to reduce compuatation time.
