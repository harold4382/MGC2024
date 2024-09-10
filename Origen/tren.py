# train.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization
from keras.regularizers import l2

# Cargar los datos
data = pd.read_excel('/app/Datos/MGC hoy.xlsx')

# Preprocesamiento
bd = data.copy()
bd['RESULTADO'] = bd['RESULTADO'].apply(lambda x: 1 if x == 'CON INFRACCIÓN' else 0)

# TAM -> TAMAÑO
size_mapping = {'MICROEMPRESA': 1, 'PEQUEÑA EMPRESA': 2, 'MEDIANA EMPRESA': 3, 'GRAN EMPRESA': 4}
bd['TAM'] = bd['TAMAÑO'].map(size_mapping)

# VENT -> VENTAS
ventas_mapping = {'DE 1 A 25 UIT': 1,    'DE 25 A 50 UIT': 2,    'DE 50 A 75 UIT': 3,    'DE 75 A 100 UIT': 4,    'DE 100 A 125 UIT': 5,    'DE 125 A 150 UIT': 6,    'DE 150 A 200 UIT': 7,    'DE 200 A 300 UIT': 8,
    'DE 300 A 400 UIT': 9,    'DE 400 A 500 UIT': 10,    'DE 500 A 600 UIT': 11,    'DE 600 A 700 UIT': 12,    'DE 700 A 800 UIT': 13,    'DE 800 A 900 UIT': 14,    'DE 900 A 1000 UIT': 15,    'DE 1000 A 1100 UIT': 16,
    'DE 1100 A 1200 UIT': 17,    'DE 1200 A 1300 UIT': 18,    'DE 1300 A 1400 UIT': 19,    'DE 1400 A 1500 UIT': 20,    'DE 1500 A 1600 UIT': 21,    'DE 1600 A 1700 UIT': 22,    'DE 1700 A 1800 UIT': 23,    'DE 1800 A 1900 UIT': 24,
    'DE 1900 A 2000 UIT': 25,    'DE 2000 A 2100 UIT': 26,    'DE 2100 A 2200 UIT': 27,    'DE 2200 A 2300 UIT': 28,    'DE 2300 A 2500 UIT': 29,    'DE 2500 A 3000 UIT': 30,    'DE 3000 A 3500 UIT': 31,    'DE 3500 A 4000 UIT': 32,
    'DE 4000 A 5000 UIT': 33,    'DE 5000 A 10000 UIT': 34,    'DE 10000 UIT A MAS': 35}
bd['VENT'] = bd['VENTAS'].map(ventas_mapping)

# CONTRI -> CONTRICAT
contricat_mapping = {'Empresas y Entidades Comerciales': 5, 'Otros': 1}
bd['CONTR'] = bd['CONTRICAT'].map(contricat_mapping)

# CONT <- CONTAB
contab_mapping = {    'MANUAL': 1,    'MANUAL/COMPUTARIZADO': 2,    'COMPUTARIZADO': 3}
bd['CONT'] = bd['CONTAB'].map(contab_mapping)

# Variables independientes y dependiente
X = bd[['MES', 'CONTR', 'CONT', 'ANEXOS', 'ANTIGUEDAD', 'PORC_LOC', 'PORC_TSB', 'PROM_SLE']]
y = bd['RESULTADO']

# Imputación de valores faltantes
imputer = KNNImputer(n_neighbors=5)
X_imputed = imputer.fit_transform(X)
X_imputed = pd.DataFrame(X_imputed, columns=X.columns)

# Balanceo con SMOTE
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X_imputed, y)

# División en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.3, random_state=42)

# Escalado
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Definir el modelo
def create_model(reg_param=0.0001):
    model = Sequential()
    model.add(Dense(32, activation='relu', kernel_regularizer=l2(reg_param), input_shape=(X_train.shape[1],)))
    model.add(BatchNormalization())
    model.add(Dense(16, activation='relu', kernel_regularizer=l2(reg_param)))
    model.add(BatchNormalization())
    model.add(Dense(1, activation='sigmoid'))
    return model

# Compilar y entrenar el modelo
model = create_model()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train_scaled, y_train, epochs=100, batch_size=64, validation_split=0.2, verbose=1)

# Guardar el modelo entrenado
model.save('/app/Origen/model.h5')
print("Modelo entrenado y guardado.")

