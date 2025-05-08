import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

dane=pd.read_csv('transport_dane.csv', sep=';')
X=dane[['trasa', 'double']]
y=dane['czas']
x_train,x_test,y_train,y_test = train_test_split(X,y, test_size=0.2, random_state=42)

model=LinearRegression()
model.fit(x_train,y_train)

y_pred=model.predict(x_test)


mse=mean_squared_error(y_test,y_pred)
print(f"Błąd średniokwadratowy: {mse:.2f}")

nowa_trasa=[[592,0]]
przewidywany_czas=model.predict(nowa_trasa)
print(f"Przewidywany czas dojechania towaru na miejsce: {przewidywany_czas[0]:.2f} godz.")