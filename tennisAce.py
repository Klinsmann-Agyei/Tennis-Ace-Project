import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# load and investigate the data here:
# load and investigate the data here:
data=pd.read_csv('tennis_stats.csv')
print(data.head())
print(data.columns)
print(data.describe())

# exploratory analysis
plt.scatter(data.FirstServeReturnPointsWon, data.Winnings)
plt.title('FirstServeReturnPointsWon vs Winnings')
plt.xlabel('FirstServeReturnPointsWon')
plt.ylabel('Winnings')

plt.show()
plt.clf()


plt.scatter(data.BreakPointsOpportunities, data.Winnings)
plt.title('BreakPointsOpportunities vs Winnings')
plt.xlabel('BreakPointsOpportunities')
plt.ylabel('Winnings')

plt.show()
plt.clf()


plt.scatter(data.BreakPointsSaved, data.Winnings)
plt.title('BreakPointsSaved vs Winnings')
plt.xlabel('BreakPointsSaved')
plt.ylabel('Winnings')
plt.show()
plt.clf()


plt.scatter(data.TotalPointsWon, data.Ranking)
plt.title('TotalPointsWon vs Ranking')
plt.xlabel('TotalPointsWon')
plt.ylabel('Ranking')
plt.show()
plt.clf()

plt.scatter(data.TotalServicePointsWon,data.Wins)
plt.title('TotalServicePointsWon vs Wins')
plt.xlabel('TotalServicePointsWon')
plt.ylabel('Wins')
plt.show()
plt.clf()

plt.scatter(data.Aces, data.Winnings)
plt.title('Aces vs Wins')
plt.xlabel('Aces')
plt.ylabel('Wins')
plt.show()
plt.clf()

plt.scatter(data.ServiceGamesWon, data.Winnings)
plt.title('ServiceGamesWon vs Wins')
plt.xlabel('ServiceGamesWon')
plt.ylabel('Wins')
plt.show()
plt.clf()


## single feature linear regression (FirstServeReturnPointsWon)
X=data[['FirstServeReturnPointsWon']]
y=data['Winnings']
X_train,X_val,y_train,y_val=train_test_split(X,y,train_size=0.8,test_size=0.2)
mdl=LinearRegression()
my_mdl=mdl.fit(X_train,y_train)
print('FirstServeReturnPointsWon vs Winnings Score > ' , my_mdl.score(X_val,y_val))
pred=my_mdl.predict(X_val)
plt.scatter(y_val,pred,alpha=0.4)
plt.title('Predicted Winnings vs. Actual Winnings - Single Feature')
plt.xlabel('Actual Winnings')
plt.ylabel('Predicted Winnings')
plt.show()
plt.clf()


## single feature linear regression (BreakPointsOpportunities)
X2=data[['BreakPointsOpportunities']]
y2=data['Winnings']
X2_train,X2_val,y2_train,y2_val=train_test_split(X2,y2,train_size=0.8,test_size=0.2)
mdl2=LinearRegression()
my_mdl2=mdl2.fit(X2_train,y2_train)
print('BreakPointsOpportunities vs Winnings Score > ',my_mdl2.score(X2_val,y2_val))
pred2=my_mdl2.predict(X2_val)
plt.scatter(y2_val,pred2,alpha=0.4)
plt.title('Predicted Winnings vs. Actual Winnings - Single Feature')
plt.xlabel('Actual Winnings')
plt.ylabel('Predicted Winnings')
plt.show()
plt.clf()



## two feature linear regression
features=['BreakPointsOpportunities', 'FirstServeReturnPointsWon']
X3=data[features]
y3=data['Winnings']
X3_train,X3_val,y3_train,y3_val=train_test_split(X3,y3,train_size=0.8,test_size=0.2)
mdl3=LinearRegression()
my_mdl3=mdl3.fit(X3_train,y3_train)
print('2 Features vs Winnings Score > ', my_mdl3.score(X3_val,y3_val))
pred3=my_mdl3.predict(X3_val)
plt.scatter(y3_val,pred3,alpha=0.4)
plt.title('Predicted Winnings vs. Actual Winnings - 2 Features')
plt.xlabel('Actual Winnings')
plt.ylabel('Predicted Winnings')
plt.show()
plt.clf()

## multiple features linear regression
features2=['BreakPointsOpportunities','FirstServeReturnPointsWon','ServiceGamesPlayed','ServiceGamesWon','BreakPointsOpportunities','BreakPointsSaved','DoubleFaults','ReturnGamesPlayed']

X4=data[features2]
y4=data['Winnings']
X4_train,X4_val,y4_train,y4_val=train_test_split(X4,y4,train_size=0.8,test_size=0.2)
mdl4=LinearRegression()
my_mdl4=mdl4.fit(X4_train,y4_train)
print('Predicting Winnings with Multiple Features Test Score > ' , my_mdl4.score(X4_val,y4_val))
pred4=my_mdl4.predict(X4_val)
plt.scatter(y4_val,pred4,alpha=0.4)
plt.title('Predicted Winnings vs. Actual Winnings - Multiple Features')
plt.xlabel('Actual Winnings')
plt.ylabel('Predicted Winnings')
plt.show()
plt.clf()








