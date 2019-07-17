## Project Description

This project is to use Machine Learning and Natural Language Processing algorithms to train a classifier 
to classify consumer financial product complaint narratives into different product types, for example, is
the complaint very likely to be mortgage-related issue? Is the complaint very likely to be credit card-related
issue etc. The dataset used to train the models is available at https://www.consumerfinance.gov/data-research/consumer-complaints . 

The jupyter notebook "Data Exploration and Visualizations" is about exploratory data analysis, geometric 
visualization of the consumer complaints on mortgages, and heat map/correlations among various features 
in consumer complaints.

A logistic regression model and an LSTM-based RNN model were trained. Both models performed relatively well  
on the test set, where the LSTM-based RNN has slightly higher precision and recall (84%) than the logistic 
regression (83%). 

A working interactive website of this application is live at: http://complaint-classification.herokuapp.com .
The "Web App" folder contains the Python and Flask implementation details of the interactive website.