import streamlit as st
import numpy as np
import gzip
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder

@st.cache_resource
def load_model():
    filename = "mejor_modelo_redes_Xpromax.pkl.gz"
    with gzip.open(filename, 'rb') as f:
        model = pickle.load(f)
    return model

@st.cache_resource
def load_label_encoders():
    encoder_file = "label_encoders.pkl"
    with open(encoder_file, "rb") as f:
        encoders = pickle.load(f)
    return encoders

model = load_model()
label_encoders = load_label_encoders()

st.title("Predicción de Alzheimer")

# Definir características categóricas y numéricas
categorical_features = [
    'Country', 'Gender', 'Smoking Status', 'Alcohol Consumption', 'Diabetes',
    'Hypertension', 'Cholesterol Level', 'Family History of Alzheimer’s',
    'Employment Status', 'Marital Status', 'Genetic Risk Factor (APOE-ε4 allele)',
    'Urban vs Rural Living', 'Physical Activity Level', 'Depression Level',
    'Sleep Quality', 'Dietary Habits', 'Air Pollution Exposure',
    'Social Engagement Level', 'Income Level', 'Stress Levels'
]

numeric_features = ['Age', 'Education Level', 'Cognitive Test Score']
continuous_features = ['BMI']
user_input = {}

# Obtener valores de entrada numéricos
for feature in numeric_features:
    user_input[feature] = st.number_input(feature, min_value=0, step=1, format="%d")

for feature in continuous_features:
    user_input[feature] = st.number_input(feature, value=0.0, format="%.2f")

# Obtener valores de entrada categóricos
for feature in categorical_features:
    if feature in label_encoders:
        user_input[feature] = st.selectbox(feature, label_encoders[feature].classes_)

if st.button("Predecir"):
    if model is None:
        st.error("No se puede realizar la predicción porque el modelo no se cargó correctamente.")
    else:
        try:
            df_input = pd.DataFrame([user_input])

            # Aplicar Label Encoding correctamente
            for col in categorical_features:
                if col in label_encoders:
                    if user_input[col] in label_encoders[col].classes_:
                        df_input[col] = label_encoders[col].transform([user_input[col]])[0]
                    else:
                        st.error(f"El valor '{user_input[col]}' no está en el conjunto de entrenamiento del LabelEncoder.")
                        st.stop()

            # Convertir todas las columnas numéricas a float32
            df_input = df_input.astype(np.float32)

            # Convertir a array NumPy con la forma correcta
            input_array = df_input.to_numpy().reshape(1, -1)

            # Hacer la predicción
            prediction = model.predict(input_array)
            st.markdown(prediction)
            resultado = "Positivo para Alzheimer" if prediction[0] == 1 else "Negativo para Alzheimer"
            st.subheader("Resultado de la Predicción")
            st.write(resultado)
        except Exception as e:
            st.error(f"Ocurrió un error al hacer la predicción: {str(e)}")


