import numpy as np
import pandas as pd
from flask import Flask, request, render_template, jsonify
import joblib
import os

port = int(os.environ.get("PORT", 5000))

alcaldia_coords = {
    'ALVARO OBREGON': {'centroid_lat': 19.38682526, 'centroid_long': -99.20218888},
    'MIGUEL HIDALGO': {'centroid_lat': 19.44845879, 'centroid_long': -99.19990957},
    'CUAJIMALPA DE MORELOS': {'centroid_lat': 19.35171809, 'centroid_long': -99.292708},
    'BENITO JUAREZ': {'centroid_lat': 19.36410582, 'centroid_long': -99.17406489},
    'CUAUHTEMOC': {'centroid_lat': 19.40558625, 'centroid_long': -99.14026834},
}

intervalos_categorias = {
    'Baja': 'Desde $6,999 hasta $26,999',
    'Media': 'Desde $27,000 hasta $46,999',
    'Alta': 'Desde $47,000 hasta $207,000'
}

# Create flask app
app = Flask(__name__)

# Cargar el modelo XGBoost
model = joblib.load('precio_modelo.joblib')

# Cargar el encoder de variables
encoder = joblib.load('precio_modelo_encoding.joblib')

## Cargamos y mostramos el archivo index.html como la pagina principal
@app.route("/")
def Home():
    return render_template("index1.html")


# Definimos la ruta llamada predict, que es la accedera el usuario para tener la prediccion
@app.route("/predict", methods = ["POST"])
def predict():

    # Obtener los valores desde el formulario
    Mantenimiento = float(request.form['Mantenimiento'])
    Superficie_m2 = float(request.form['Superficie m2'])
    Recamaras = int(request.form['Recamaras'])
    Baños = int(request.form['Baños'])
    Estacionamiento = int(request.form['Estacionamiento'])
    Alcaldia = request.form['Alcaldia']
    centroid_lat = 1
    centroid_long = 1
    valor_unitario_suelo = 391.147810
    subsidio = 8419.086261

    # Crear un DataFrame con los valores enviados
    datos_entrada = pd.DataFrame({
        'Mantenimiento': [Mantenimiento],
        'Superficie m2': [Superficie_m2],
        'Recamaras': [Recamaras],
        'Baños': [Baños],
        'Estacionamiento': [Estacionamiento],
        'Alcaldia': [Alcaldia],
        'centroid_lat': [centroid_lat],
        'centroid_long': [centroid_long],
        'valor_unitario_suelo': [valor_unitario_suelo],
        'subsidio': [subsidio]
    })
    # Asignar la latitud y longitud por alcaldia
    datos_entrada['centroid_lat'] = datos_entrada['Alcaldia'].apply(lambda x: alcaldia_coords[x]['centroid_lat'])
    datos_entrada['centroid_long'] = datos_entrada['Alcaldia'].apply(lambda x: alcaldia_coords[x]['centroid_long'])

    # Realizar la predicción utilizando el DataFrame
    prediction = model.predict(datos_entrada)
    # Quitamos el encoder de los resultados
    final = encoder.inverse_transform(prediction)
    # Crear una lista con los intervalos correspondientes a las predicciones
    rangos_final = [intervalos_categorias[label] for label in final]

    return render_template(
        "index1.html",
        prediction_texto="La categoria de renta es: {}".format(final),
        rango_precios_texto="Intervalo de precios: {}".format(rangos_final),
        finished = True
    )

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=port)
