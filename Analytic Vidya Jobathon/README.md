# Credit Card Lead Prediction:
  ## Preprocessing:
      -After getting the Data set. First i analysed the data and found that it contain outliers. So i removed the outliers from Vintage and Average account balance.
      -After that i filled the Null values with 'yan'. I did'nt deleted the Null value as the amount of data were large and it would have minimised my prediction.
  ## Visualisation:-
      -After that i started Visualizing each and very Columns of the data set.
      -So first i found that the Label data was not balanced.
      -I found that male were more interested in credit card than females.
      -I found people of age limit 30-60 are more interested in credit card.
      -I found people from some particular region were more interested.
      -Enterpreneur and Self_Employed are more interested in credit card while Salaried people are least. 
       As we know while doing any kind of bussiness we may need to put money in our bussiness due to which
       we may remain with less balance for personal use especially when starting new bussiness.
       Maybe due to this Entrepreneur and self_employed people are more interested in credit card.
      -People with X3 and X2 Channel_Code are more interested in Credit Card.
      -We can see people with Vintage more than 60 are more interested in Credit card.
      -People who already have credit product is more interested in Credit card. 
       People who is already in debt need more money to repay due which they are more interested in credit cards.
      -People with higher average account balance are more interested in credit card. Maybe people with lower 
       average account balance donot have confidence in repaying the debt on time.
      -Customers those are active in last 3 months are more interested in credit cards. Maybe those inactive 
       customers currently don't need extra money and this maybe the reason for there inactivity.
  ## Data Processing:-
      -After that i started Data Processing.
      -Balanced the data such that there were 2.5 times more number of zero than one(I did this because from experiment i found that is the best case).
      -I one hot encoded some datas like gender and Is_active
      -I created dummies Variable of some columns such as Channel code and occupation.
      -I binned Avg_Account balance.
      -After that i scaled the data using StandardScaler.
  ## Model Creation:-
      -After trying Decision Tree and Random Forest i found that Deep Learning is predicting better for this Data.
      -I created a Deep learning model.
      -After experimenting on number of nodes and hidden layer i found 7 hidden layer of 35 nodes performed the best.
      -After that i trained the model.
  ## Testing:-
      -Finally i tested the model and got a roc of 0.8700320345
     
