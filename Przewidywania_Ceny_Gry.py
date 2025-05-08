import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
#wczytanie danych z pliku csv
dane=pd.read_csv('gry.csv', sep=',')

#przygotowanie cech (X) i zmiennej docelowej (y)
X=dane[['godziny', 'popularnosc', 'rok', 'multiplayer']]
y=dane['cena']

#podział danych 80/20 (trenowanie/test)
x_train,x_test,y_train,y_test = train_test_split(X,y , test_size=0.2, random_state=42)


#tworzenie modelu
model=LinearRegression()
model.fit(x_train,y_train) #trenowanie modelu

#przewidywanie cen na testowych danych
y_pred=model.predict(x_test)

mse=mean_squared_error(y_test,y_pred) #pbliczanie błedu średniokwadratowego
print(f"Błąd średniokwadratowy: {mse:.2f}")

#przewidywanie ceny dla nowej gry
nowa_gra=[[100,95,2025,0]]

cena_przewidziana=model.predict(nowa_gra) #przewidywanie na podstawie cech
print(f"Przewidywana cena gry: {cena_przewidziana[0]:.2f} zł")