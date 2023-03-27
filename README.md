# MATH_Research
Files related to "On Selecting Training Algorithms and Activation Functions in Neural Networks for Regression"

Most technical analysis data added based on code from KRATISAXENA on Kaggle.

SCHD is an ETF containing stocks that have consistently growing dividends. These stocks are relatively stable and slow-moving.
SPY is an ETF tracking the 500 largest compaines in the US. It is generally indicative of the stock market as a whole.
VUG is an ETF containing large- and medium-cap growth stocks. These stocks are expected to move more than the average amount.

Between the three above securities, we expect the datasets to be sufficiently different from one another to highlight potential strengths and weaknesses of different combinations of activation functions and optimizers. By using the same type of data (that is, stock market data) throughout, we are able to effectively isolate the hyperparameters that we want to study and do not have to worry about other variables.

One critical fact to note is that we are ultimately unconcerned with the real performance of the neural networks we create. Rather, we only care about the relative performances of the neural networks compared to each other. Many more technical analysis features, dramatically more data, and significantly more sophisticated neural network architectures could be used if the goal was to create the best possible stock market bot possible. I cannot emphasize enough- true prediction accuracy is not the point of this reserach. Based on this work, we can perhaps build such a model in the future (an autoregressive neural network containing daily/weekly/monthly time series data for thousands of different stocks accounting for all kinds of qualitative data and general economic data about a given security).

Next steps (can be done in parallel):
- Create datasets with features scaled by percentage and/or standard deviation where applicable
- Create and train neural networks using combinations of optimizers and activation functions for all three stocks
