from sklearn.datasets import make_classification  # do generowania przykładowych danych
from sklearn.ensemble import RandomForestClassifier  # do używania klasyfikatora lasu losowego
from sklearn.model_selection import train_test_split  # do podziału danych na zestawy treningowe i testowe
from sklearn.metrics import accuracy_score  # do oceny skuteczności modelu

# Generowanie przykładowych danych klasyfikacyjnych
X, y = make_classification(
    n_samples=100,           # liczba próbek
    n_features=4,            # liczba cech (zmiennych)
    n_informative=2,         # liczba cech informacyjnych
    n_redundant=0,           # liczba cech redundantnych
    random_state=0           # ustalenie ziarna generatora liczb losowych
)

# Podział danych na zestaw treningowy i testowy
x_train, x_test, y_train, y_test = train_test_split(
    X, y,                     # dane wejściowe i etykiety
    test_size=0.3,            # 30% danych do testowania
    random_state=0            # ustalenie ziarna dla losowego podziału
)

# Tworzenie modelu klasyfikacji z użyciem lasu losowego
model = RandomForestClassifier(
    n_estimators=10,          # liczba drzew w lesie
    random_state=0            # ustalenie ziarna generatora liczb losowych
)

# Trenowanie modelu na danych treningowych
model.fit(x_train, y_train)

# Przewidywanie etykiet na podstawie danych testowych
y_pred = model.predict(x_test)

# Obliczanie dokładności modelu
acc = accuracy_score(y_test, y_pred)

# Wyświetlenie dokładności
print(f"Dokładność: {acc * 100:.2f}%")
