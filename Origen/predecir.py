# predecir.py

import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer

# Cargar el archivo de predicción
pred = pd.read_excel('/app/Datos/PREDECIR.xlsx')

# Preprocesamiento
pred['MES'] = 9  # Agregar mes presente

# CONTR -> CONTRICAT
contricat_mapping = { 'Empresas y Entidades Comerciales': 5,    'Entidades Gubernamentales y Públicas': 4,    'Instituciones Educativas y Culturales': 3,    'Organizaciones Sociales y Comunitarias': 2,    'Otros': 1}
bd['CONTR'] = bd['CONTRICAT'].map(contricat_mapping)

# CONT <- CONTAB
contab_mapping = {    'MANUAL': 1,    'MANUAL/COMPUTARIZADO': 2,    'COMPUTARIZADO': 3}
bd['CONT'] = bd['CONTAB'].map(contab_mapping)

# Variables independientes
pred = pred[['MES', 'CONTR', 'CONT', 'ANEXOS', 'ANTIGUEDAD', 'PORC_LOC', 'PORC_TSB', 'PROM_SLE']]

# Imputación de valores faltantes
imputer = KNNImputer(n_neighbors=5)
pred_imputed = imputer.fit_transform(pred)
pred_imputed = pd.DataFrame(pred_imputed, columns=pred.columns)

# Escalar los datos
scaler = MinMaxScaler()
pred_scaled = scaler.fit_transform(pred_imputed)

# Cargar el modelo guardado
model = load_model('/app/Origen/model.h5')

# Realizar predicciones
probabilities = model.predict(pred_scaled)
predictions = (probabilities > 0.55).astype(int)

# Agregar predicciones al DataFrame
pred_imputed['Probabilidad'] = probabilities
pred_imputed['Prediccion'] = predictions

# Guardar las predicciones
pred_imputed.to_excel('/app/prediccion_MGC.xlsx', index=False)
print("Predicciones guardadas.")

