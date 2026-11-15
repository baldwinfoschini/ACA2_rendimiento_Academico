# ACA 2 · Aprendizaje Automático – Rendimiento académico con Redes Neuronales (xAPI-Edu-Data)

Proyecto 2 (Educación). Predice el nivel de rendimiento de un estudiante (**L** bajo, **M** medio, **H** alto)
con un Perceptrón Multicapa (MLP) profundo en PyTorch y lo despliega en Streamlit.

## Estructura
```
proyecto/
├── ACA2_Aprendizaje_Automatico_xAPI.ipynb   # Pasos 1-4: conexión a datos, EDA, pipeline, red neuronal, métricas
├── app.py                                   # Paso 5: aplicación Streamlit
├── utils.py                                 # Conexión a Drive, limpieza y definición de la red (compartido)
├── requirements.txt
├── data/xAPI-Edu-Data.csv                   # Copia local (respaldo si no hay acceso a Drive)
└── modelo/
    ├── mlp_rendimiento.pt                   # Pesos de la red
    ├── preprocesador.joblib                 # Pipeline de escalado + one-hot
    └── metadata.json                        # Configuración, métricas, historial y matriz de confusión
```

## Conexión a la base de datos
Los datos se leen en formato CSV desde Google Drive:
`https://drive.google.com/file/d/1cIeVbKI3NpUvt3GHW7yuZibyADUCq3uw/view`

`utils.cargar_datos()` usa la URL de descarga directa `https://drive.google.com/uc?export=download&id=<ID>`.
El archivo debe estar compartido como **"Cualquier persona con el enlace"**. Si no hay conexión, se usa
`data/xAPI-Edu-Data.csv` automáticamente.

## Cómo ejecutar
```bash
pip install -r requirements.txt
jupyter notebook ACA2_Aprendizaje_Automatico_xAPI.ipynb   # re-entrena y regenera ./modelo
streamlit run app.py                                       # abre la app en http://localhost:8501
```
En **Google Colab**: sube la carpeta completa, ejecuta `%cd` hacia ella y corre el notebook.

## Despliegue en Streamlit Community Cloud
1. Sube esta carpeta a un repositorio de GitHub (incluyendo `modelo/` y `data/`).
2. En https://share.streamlit.io → *New app* → selecciona el repositorio y `app.py`.

## Resultados (conjunto de prueba, 72 estudiantes)
| Modelo | Accuracy | F1 macro | F1 macro CV (5 part.) |
|---|---|---|---|
| MLP final (64-32-16, dropout 0.3, con aumento de datos) | 0.764 | 0.760 | 0.774 |
| MLP sin aumento de datos | 0.722 | 0.722 | 0.758 |
| Regresión logística (referencia) | 0.806 | 0.805 | 0.745 |

## Referencia
Aljarah, I. (2016). *xAPI-Edu-Data (Student Academic Performance)* [Conjunto de datos]. Kaggle.
https://www.kaggle.com/datasets/aljarah/xAPI-Edu-Data
