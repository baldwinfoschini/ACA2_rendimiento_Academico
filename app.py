"""
app.py - Paso 5: Despliegue en Streamlit
Predicción del rendimiento académico (xAPI-Edu-Data) con una red neuronal MLP (PyTorch).

Ejecutar:  streamlit run app.py
"""
import json
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch

import utils

BASE = Path(__file__).resolve().parent
DIR_MODELO = BASE / "modelo"
COLORES = {"L": "#E4572E", "M": "#F3A712", "H": "#29A36A"}
ETIQUETA = {"L": "Bajo (L)", "M": "Medio (M)", "H": "Alto (H)"}

st.set_page_config(page_title="Rendimiento académico · MLP", page_icon="🎓", layout="wide")


# ---------------------------------------------------------------- carga de recursos
@st.cache_resource
def cargar_modelo():
    meta = json.loads((DIR_MODELO / "metadata.json").read_text(encoding="utf-8"))
    modelo = utils.MLPRendimiento(meta["n_entradas"], tuple(meta["capas"]), dropout=meta["dropout"])
    modelo.load_state_dict(torch.load(DIR_MODELO / "mlp_rendimiento.pt", map_location="cpu"))
    modelo.eval()
    prep = joblib.load(DIR_MODELO / "preprocesador.joblib")
    return modelo, prep, meta


@st.cache_data(ttl=3600, show_spinner="Conectando con la base de datos (Google Drive)…")
def cargar_dataset():
    return utils.cargar_datos(verbose=False)


modelo, prep, meta = cargar_modelo()
PAISES = [p for p in meta["categorias"]["Nationality"] if p != "Other"]


def predecir(df_crudo: pd.DataFrame) -> pd.DataFrame:
    """Recibe filas con el formato original del CSV y devuelve probabilidades y clase predicha."""
    d = utils.limpiar(df_crudo, paises_validos=meta["categorias"]["Nationality"])
    X = prep.transform(utils.preparar_X(d)).astype(np.float32)
    with torch.no_grad():
        proba = torch.softmax(modelo(torch.tensor(X)), 1).numpy()
    out = pd.DataFrame(proba, columns=[f"P({c})" for c in utils.CLASES], index=df_crudo.index)
    out["Prediccion"] = [utils.CLASES[i] for i in proba.argmax(1)]
    return out


# ---------------------------------------------------------------- barra lateral
with st.sidebar:
    st.title("🎓 Rendimiento académico")
    st.caption("ACA 2 · Aprendizaje Automático · CUN")
    st.markdown(
        "Red neuronal **MLP** entrenada con el dataset **xAPI-Edu-Data** (480 estudiantes) para predecir "
        "el nivel de rendimiento:\n\n"
        "- **L**: bajo (0–69)\n- **M**: medio (70–89)\n- **H**: alto (90–100)"
    )
    fila = next(r for r in meta["metricas_test"] if r["Modelo"] == "MLP final")
    c1, c2 = st.columns(2)
    c1.metric("Accuracy (test)", f"{fila['Accuracy']:.1%}")
    c2.metric("F1 macro (test)", f"{fila['F1 macro']:.3f}")
    st.metric("F1 macro (CV 5 particiones)", f"{fila['F1 macro CV (5 part.)']:.3f}")
    st.caption(f"Arquitectura: {meta['n_entradas']} → {' → '.join(map(str, meta['capas']))} → 3 · "
               f"dropout {meta['dropout']}")

tab_pred, tab_lote, tab_eda, tab_modelo = st.tabs(
    ["🔮 Predicción individual", "📂 Predicción por lotes", "📊 Exploración de datos", "📈 Desempeño del modelo"]
)

# ---------------------------------------------------------------- 1. predicción individual
with tab_pred:
    st.subheader("Datos del estudiante")
    with st.form("form_estudiante"):
        a, b, c = st.columns(3)
        with a:
            st.markdown("**Interacción con la plataforma**")
            raised = st.slider("Veces que levantó la mano", 0, 100, 50)
            visited = st.slider("Recursos visitados", 0, 100, 55)
            announ = st.slider("Anuncios vistos", 0, 100, 38)
            discus = st.slider("Participación en discusiones", 0, 100, 43)
        with b:
            st.markdown("**Datos académicos**")
            grade = st.selectbox("Grado", [f"G-{g:02d}" for g in range(2, 13)], index=5)
            stage = st.selectbox("Nivel educativo", ["lowerlevel", "MiddleSchool", "HighSchool"], index=1,
                                 format_func={"lowerlevel": "Primaria", "MiddleSchool": "Secundaria",
                                              "HighSchool": "Bachillerato"}.get)
            section = st.selectbox("Sección", ["A", "B", "C"])
            topic = st.selectbox("Asignatura", meta["categorias"]["Topic"])
            semester = st.radio("Semestre", ["F", "S"], horizontal=True,
                                format_func={"F": "Primero", "S": "Segundo"}.get)
            absence = st.radio("Días de ausencia", ["Under-7", "Above-7"], horizontal=True,
                               format_func={"Under-7": "Menos de 7", "Above-7": "Más de 7"}.get)
        with c:
            st.markdown("**Datos personales y familiares**")
            gender = st.radio("Género", ["M", "F"], horizontal=True,
                              format_func={"M": "Masculino", "F": "Femenino"}.get)
            nat = st.selectbox("Nacionalidad", PAISES + ["Otra"])
            relation = st.radio("Responsable", ["Father", "Mum"], horizontal=True,
                                format_func={"Father": "Padre", "Mum": "Madre"}.get)
            survey = st.radio("¿El responsable respondió la encuesta?", ["Yes", "No"], horizontal=True,
                              format_func={"Yes": "Sí", "No": "No"}.get)
            satisf = st.radio("Satisfacción del responsable con la escuela", ["Good", "Bad"], horizontal=True,
                              format_func={"Good": "Buena", "Bad": "Mala"}.get)
        enviar = st.form_submit_button("Predecir rendimiento", type="primary", width="stretch")

    if enviar:
        registro = pd.DataFrame([{
            "gender": gender, "NationalITy": "Other" if nat == "Otra" else nat, "PlaceofBirth": "",
            "StageID": stage, "GradeID": grade, "SectionID": section, "Topic": topic, "Semester": semester,
            "Relation": relation, "raisedhands": raised, "VisITedResources": visited,
            "AnnouncementsView": announ, "Discussion": discus, "ParentAnsweringSurvey": survey,
            "ParentschoolSatisfaction": satisf, "StudentAbsenceDays": absence,
        }])
        res = predecir(registro).iloc[0]
        clase = res["Prediccion"]
        izq, der = st.columns([1, 2])
        with izq:
            st.markdown(
                f"<div style='padding:1.2rem;border-radius:12px;background:{COLORES[clase]}22;"
                f"border:2px solid {COLORES[clase]}'>"
                f"<div style='font-size:0.9rem'>Nivel de rendimiento predicho</div>"
                f"<div style='font-size:2rem;font-weight:700;color:{COLORES[clase]}'>{ETIQUETA[clase]}</div>"
                f"<div>Confianza: {res[f'P({clase})']:.1%}</div></div>",
                unsafe_allow_html=True,
            )
            if clase == "L":
                st.warning("Riesgo de bajo rendimiento: conviene un seguimiento temprano.")
            elif clase == "H":
                st.success("Perfil asociado a alto rendimiento.")
        with der:
            dfp = pd.DataFrame({"Clase": [ETIQUETA[c] for c in utils.CLASES],
                                "Probabilidad": [res[f"P({c})"] for c in utils.CLASES],
                                "cod": utils.CLASES})
            graf = alt.Chart(dfp).mark_bar(cornerRadiusEnd=4).encode(
                x=alt.X("Probabilidad:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
                y=alt.Y("Clase:N", sort=None, title=None, axis=alt.Axis(labelOverlap=False, labelLimit=200)),
                color=alt.Color("cod:N", scale=alt.Scale(domain=list(COLORES), range=list(COLORES.values())),
                                legend=None),
                tooltip=["Clase", alt.Tooltip("Probabilidad:Q", format=".1%")],
            ).properties(height=210, title="Probabilidad por clase")
            st.altair_chart(graf, width="stretch")

# ---------------------------------------------------------------- 2. predicción por lotes
with tab_lote:
    st.subheader("Predicción para varios estudiantes")
    st.markdown("Sube un CSV con las mismas columnas del dataset xAPI-Edu-Data (la columna `Class` es opcional). "
                "Si incluye `Class`, se calcula la exactitud.")
    archivo = st.file_uploader("Archivo CSV", type="csv")
    usar_ejemplo = st.checkbox("Usar una muestra de 25 estudiantes del dataset como ejemplo")
    df_in = None
    if archivo is not None:
        df_in = pd.read_csv(archivo)
    elif usar_ejemplo:
        df_in = cargar_dataset()[0].sample(25, random_state=7).reset_index(drop=True)
    if df_in is not None:
        requeridas = ["gender", "NationalITy", "StageID", "GradeID", "SectionID", "Topic", "Semester", "Relation",
                      "raisedhands", "VisITedResources", "AnnouncementsView", "Discussion",
                      "ParentAnsweringSurvey", "ParentschoolSatisfaction", "StudentAbsenceDays"]
        faltan = [c for c in requeridas if c not in df_in.columns]
        if faltan:
            st.error(f"Faltan columnas: {faltan}")
        else:
            res = pd.concat([df_in, predecir(df_in)], axis=1)
            if "Class" in df_in:
                acc = (res["Class"] == res["Prediccion"]).mean()
                st.metric("Exactitud sobre el archivo", f"{acc:.1%}")
            cnt = res["Prediccion"].value_counts().reindex(utils.CLASES, fill_value=0)
            k1, k2, k3 = st.columns(3)
            for col, cl in zip((k1, k2, k3), utils.CLASES):
                col.metric(ETIQUETA[cl], int(cnt[cl]))
            st.dataframe(res, width="stretch", height=380)
            st.download_button("Descargar predicciones (CSV)", res.to_csv(index=False).encode("utf-8"),
                               "predicciones.csv", "text/csv")

# ---------------------------------------------------------------- 3. EDA
with tab_eda:
    df, origen = cargar_dataset()
    st.subheader("Exploración del dataset")
    st.caption(f"Fuente de datos: **{origen}** · {df.shape[0]} registros · {df.shape[1]} columnas")
    if origen != "Google Drive":
        st.info("No se pudo acceder a Google Drive; se muestran los datos de la copia local.")
    with st.expander("Ver datos"):
        st.dataframe(df, width="stretch")

    escala = alt.Scale(domain=list(COLORES), range=list(COLORES.values()))
    c1, c2 = st.columns(2)
    with c1:
        st.altair_chart(alt.Chart(df).mark_bar().encode(
            x=alt.X("Class:N", sort=utils.CLASES, title="Clase", axis=alt.Axis(labelAngle=0)), y=alt.Y("count():Q", title="Estudiantes"),
            color=alt.Color("Class:N", scale=escala, legend=None), tooltip=["Class", "count()"],
        ).properties(title="Distribución de la clase", height=300), width="stretch")
    with c2:
        var = st.selectbox("Variable numérica", ["raisedhands", "VisITedResources", "AnnouncementsView", "Discussion"])
        st.altair_chart(alt.Chart(df).mark_boxplot(size=40).encode(
            x=alt.X("Class:N", sort=utils.CLASES, title="Clase", axis=alt.Axis(labelAngle=0)), y=alt.Y(f"{var}:Q"),
            color=alt.Color("Class:N", scale=escala, legend=None),
        ).properties(title=f"{var} por clase", height=300), width="stretch")

    c3, c4 = st.columns(2)
    with c3:
        cat = st.selectbox("Variable categórica", ["StudentAbsenceDays", "ParentAnsweringSurvey",
                                                    "ParentschoolSatisfaction", "Relation", "gender", "StageID",
                                                    "Topic", "NationalITy", "Semester", "SectionID"])
        st.altair_chart(alt.Chart(df).mark_bar().encode(
            y=alt.Y(f"{cat}:N", title=None), x=alt.X("count():Q", stack="normalize", title="% de estudiantes",
                                                      axis=alt.Axis(format="%")),
            color=alt.Color("Class:N", scale=escala, sort=utils.CLASES),
            order=alt.Order("orden:Q"), tooltip=[cat, "Class", "count()"],
        ).transform_calculate(orden="indexof(['L','M','H'], datum.Class)")
         .properties(title=f"Composición de la clase según {cat}", height=300), width="stretch")
    with c4:
        ejes = ["raisedhands", "VisITedResources", "AnnouncementsView", "Discussion"]
        ex = st.selectbox("Eje X", ejes, index=1)
        ey = st.selectbox("Eje Y", ejes, index=0)
        st.altair_chart(alt.Chart(df).mark_circle(size=45, opacity=0.65).encode(
            x=f"{ex}:Q", y=f"{ey}:Q", color=alt.Color("Class:N", scale=escala, sort=utils.CLASES),
            tooltip=[ex, ey, "Class", "Topic"],
        ).properties(title="Dispersión por clase", height=300).interactive(), width="stretch")

# ---------------------------------------------------------------- 4. desempeño del modelo
with tab_modelo:
    st.subheader("Desempeño del modelo (conjunto de prueba)")
    tabla = pd.DataFrame(meta["metricas_test"]).set_index("Modelo")
    st.dataframe(tabla.style.format("{:.3f}").highlight_max(axis=0, subset=["Accuracy", "F1 macro",
                 "F1 ponderado", "F1 macro CV (5 part.)"], color="#29A36A33"), width="stretch")
    st.caption(f"Partición: {meta['particion']['train']} entrenamiento "
               f"({meta['particion']['train_aumentado']} tras aumento de datos) · "
               f"{meta['particion']['val']} validación · {meta['particion']['test']} prueba. "
               "El conjunto de prueba es pequeño, por eso se reporta también el F1 macro con validación cruzada.")

    h = meta["historial"]
    hist = pd.DataFrame({"Época": range(1, len(h["loss"]) + 1), "Entrenamiento": h["loss"],
                         "Validación": h["val_loss"]}).melt("Época", var_name="Conjunto", value_name="Pérdida")
    hacc = pd.DataFrame({"Época": range(1, len(h["acc"]) + 1), "Entrenamiento": h["acc"],
                         "Validación": h["val_acc"]}).melt("Época", var_name="Conjunto", value_name="Accuracy")
    regla = alt.Chart(pd.DataFrame({"x": [meta["mejor_epoca"]]})).mark_rule(strokeDash=[4, 4]).encode(x=alt.X("x:Q", title="Época"))
    g1, g2 = st.columns(2)
    g1.altair_chart((alt.Chart(hist).mark_line().encode(x="Época:Q", y="Pérdida:Q", color="Conjunto:N") + regla)
                    .properties(title="Curva de pérdida", height=280), width="stretch")
    g2.altair_chart((alt.Chart(hacc).mark_line().encode(x="Época:Q", y="Accuracy:Q", color="Conjunto:N") + regla)
                    .properties(title="Curva de exactitud", height=280), width="stretch")
    st.caption(f"Línea punteada: mejor época ({meta['mejor_epoca']}), pesos restaurados por early stopping.")

    m1, m2 = st.columns(2)
    with m1:
        cm = np.array(meta["matriz_confusion"])
        dcm = pd.DataFrame([(utils.CLASES[i], utils.CLASES[j], int(cm[i, j]))
                            for i in range(3) for j in range(3)], columns=["Real", "Predicha", "n"])
        base = alt.Chart(dcm).encode(x=alt.X("Predicha:N", sort=utils.CLASES, axis=alt.Axis(labelAngle=0)), y=alt.Y("Real:N", sort=utils.CLASES))
        st.altair_chart((base.mark_rect().encode(color=alt.Color("n:Q", scale=alt.Scale(scheme="blues"), legend=None))
                         + base.mark_text(fontSize=18).encode(
                             text="n:Q", color=alt.condition("datum.n > 12", alt.value("white"), alt.value("black"))))
                        .properties(title="Matriz de confusión", height=300), width="stretch")
    with m2:
        rep = meta["reporte"]
        st.dataframe(pd.DataFrame({c: rep[c] for c in utils.CLASES + ["macro avg", "weighted avg"]}).T
                     .rename(columns={"precision": "Precisión", "recall": "Recall", "f1-score": "F1",
                                      "support": "Soporte"}).style.format("{:.3f}"), width="stretch")
        imp = pd.DataFrame(list(meta["importancia"].items()), columns=["Variable", "Importancia"])
        st.altair_chart(alt.Chart(imp).mark_bar(color="#5B5BD6").encode(
            x=alt.X("Importancia:Q", title="Caída del F1 macro al permutar"), y=alt.Y("Variable:N", sort="-x"),
        ).properties(title="Importancia de variables (permutación)", height=330), width="stretch")
