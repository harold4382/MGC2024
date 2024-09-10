# evaluar.py

import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Cargar los datos
data = pd.read_excel('/app/Datos/MGC hoy.xlsx')

# Preprocesamiento similar a train.py
bd = data.copy()
bd['RESULTADO'] = bd['RESULTADO'].apply(lambda x: 1 if x == 'CON INFRACCIÓN' else 0)

# Mapear las variables categóricas
size_mapping = {'MICROEMPRESA': 1, 'PEQUEÑA EMPRESA': 2, ...}  # Completar este mapeo
bd['TAM'] = bd['TAMAÑO'].map(size_mapping)

# Otros mapeos
ventas_mapping = {...}  # Completar este mapeo
bd['VENT'] = bd['VENTAS'].map(ventas_mapping)
contricat_mapping = {...}  # Completar este mapeo
bd['CONTR'] = bd['CONTRICAT'].map(contricat_mapping)
contab_mapping = {...}  # Completar este mapeo
bd['CONT'] = bd['CONTAB'].map(contab_mapping)

# Variables independientes y dependiente
X = bd[['MES', 'CONTR', 'CONT', 'ANEXOS', 'ANTIGUEDAD', 'PORC_LOC', 'PORC_TSB', 'PROM_SLE']]
y = bd['RESULTADO']

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Escalar los datos
scaler = MinMaxScaler()
X_test_scaled = scaler.fit_transform(X_test)

# Cargar el modelo guardado
model = load_model('/app/Origen/model.h5')

# Evaluar el modelo
loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=1)
print(f"Accuracy: {accuracy}")

