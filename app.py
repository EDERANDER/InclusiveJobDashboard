import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

# Configuración de la página de Streamlit
st.set_page_config(page_title="Inclusión Laboral Dashboard", layout="wide", page_icon="🌐")

# Cargar los datos automáticamente con st.cache_data
@st.cache_data
def load_data():
    return pd.read_csv('1_vacantes_limpio.csv')

df = load_data()

# Menú de navegación
st.sidebar.title("🔍 Explora el Dashboard")
menu = st.sidebar.radio("📊 Secciones:", ["🏠 Bienvenido", "📊 Análisis de Datos", "🤖 Predicciones ML"])

# --- SECCIÓN 1: Bienvenido ---
if menu == "🏠 Bienvenido":
    st.title("🌐 Dashboard de Inclusión Laboral para Personas con Discapacidad 🌐")
    st.write(
        """
        💼 En el contexto laboral actual, las oportunidades para personas con discapacidad son limitadas.
        Este proyecto busca analizar la inclusión laboral mediante un análisis de sectores y otros factores clave.
        
        🎯 **Objetivo**: Proveer una visión clara del estado de la inclusión laboral y utilizar Machine Learning para predecir patrones en vacantes inclusivas.
        
        """
    )
    
    # Imagen representativa centrada
    image = Image.open("img/inclusion.jpeg")  # Asegúrate de tener esta imagen en tu directorio
    st.image(image, caption="Inclusión laboral para todos:(https://www.cepal.org/sites/default/files/styles/1920x1080/public/images/featured/inclusion-social.jpg?itok=E6Dj-ehF)", use_column_width=True)

    st.subheader("Preguntas Clave")
    st.write(
        """
        - ¿Cuáles son los sectores con más oportunidades inclusivas?
        - ¿Qué regiones ofrecen más vacantes para personas con discapacidad?
        - ¿Qué requisitos de experiencia y competencias se solicitan más?
        """
    )

    st.subheader("Equipo de Desarrollo")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image("https://randomuser.me/api/portraits/men/1.jpg", width=100)  
        st.write("**Ángel**\nRol: Data Scientist")
    with col2:
        st.image("https://randomuser.me/api/portraits/men/2.jpg", width=100)
        st.write("**Beckham**\nRol: Desarrollador Full Stack")
    with col3:
        st.image("https://randomuser.me/api/portraits/men/3.jpg", width=100)
        st.write("**Ander**\nRol: Analista de Datos")





# --- SECCIÓN 2: Análisis de Datos ---

# --- SECCIÓN 2: Estadísticas Básicas ---
elif menu == "📊 Análisis de Datos":
    st.title("📊 Análisis de Datos: Estadísticas Básicas 📈")

    st.subheader("1. Distribución de Vacantes por Sector")
    sector_counts = df['SECTOR'].value_counts().reset_index()
    sector_counts.columns = ['Sector', 'Number of Vacancies']
    fig1 = px.bar(
        sector_counts,
        x='Sector',
        y='Number of Vacancies',
        color='Number of Vacancies',
        color_continuous_scale='Blues',
        labels={'Sector': 'Sector', 'Number of Vacancies': 'Número de Vacantes'},
        title="Vacantes por Sector"
    )
    st.plotly_chart(fig1)
    st.write("**Interpretación**: Esta visualización muestra los sectores con más vacantes inclusivas, permitiendo identificar cuáles áreas son más accesibles.")

    st.subheader("2. Distribución de Vacantes por Provincia")
    provincia_counts = df['PROVINCIA'].value_counts().reset_index()
    provincia_counts.columns = ['Provincia', 'Number of Vacancies']
    fig2 = px.bar(
        provincia_counts,
        x='Provincia',
        y='Number of Vacancies',
        color='Number of Vacancies',
        color_continuous_scale='OrRd',
        labels={'Provincia': 'Provincia', 'Number of Vacancies': 'Número de Vacantes'},
        title="Vacantes por Provincia"
    )
    st.plotly_chart(fig2)
    st.write("**Interpretación**: Nos ayuda a comprender si existen oportunidades de empleo inclusivo en todas las provincias o si están concentradas en ciertas regiones.")

    st.subheader("3. Requisitos de Experiencia en Vacantes")
    experience_counts = df['SINEXPERIENCIA'].value_counts().reset_index()
    experience_counts.columns = ['Experiencia', 'Número de Vacantes']
    fig3 = px.pie(
        experience_counts,
        names='Experiencia',
        values='Número de Vacantes',
        title="Vacantes con o sin Requisitos de Experiencia",
        color_discrete_sequence=px.colors.sequential.RdBu
    )
    st.plotly_chart(fig3)
    st.write("**Interpretación**: Indica qué proporción de las vacantes requieren experiencia previa, un factor que puede limitar el acceso para algunos candidatos.")

    st.subheader("4. Experiencia Promedio por Sector")
    avg_experience = df.groupby('SECTOR')['EXPERIENCIA_MESES'].mean().reset_index()
    avg_experience.columns = ['Sector', 'Experiencia Promedio (meses)']
    fig4 = px.bar(
        avg_experience,
        x='Sector',
        y='Experiencia Promedio (meses)',
        color='Experiencia Promedio (meses)',
        color_continuous_scale='Viridis',
        title="Experiencia Promedio Requerida por Sector"
    )
    st.plotly_chart(fig4)
    st.write("**Interpretación**: Identifica sectores que imponen mayores barreras de experiencia, proporcionando información valiosa para ajustar las políticas inclusivas.")


    # Distribución de Vacantes para Personas con Discapacidad
    st.subheader("1. Distribución de Vacantes para Personas con Discapacidad")
    fig_pie = px.pie(df, names='ESPCD', title="Distribución de Vacantes para Personas con Discapacidad",
                     color_discrete_sequence=px.colors.sequential.Blues)
    st.plotly_chart(fig_pie)

    # Comparación de Vacantes Exclusivas para Personas con Discapacidad
    st.subheader("2. Comparación de Vacantes Exclusivas para Personas con Discapacidad")
    vacantes_counts = df['ESPCD'].value_counts().reset_index()
    vacantes_counts.columns = ['Vacantes para PCD', 'Número de Vacantes']
    fig_bar = px.bar(vacantes_counts, x='Vacantes para PCD', y='Número de Vacantes',
                     title="Comparación de Vacantes Exclusivas para Personas con Discapacidad",
                     color='Número de Vacantes', color_continuous_scale=px.colors.sequential.Pinkyl)
    st.plotly_chart(fig_bar)




# --- SECCIÓN 3: Predicciones ML ---
elif menu == "🤖 Predicciones ML":
    st.title("🤖 Predicciones y Clasificación de Vacantes Inclusivas")

    # Explicación adicional para el usuario
    st.write(
        """
        ### ¿Cómo Funciona el Modelo?
        Este modelo de Machine Learning usa variables como la experiencia y el sector para predecir si una vacante es inclusiva.
        Puedes ajustar los valores en los controles para ver cómo afectan la predicción.
        
        ### Explicación de la Predicción
        - **Experiencia en meses**: Mayor experiencia puede influir en la inclusión, dependiendo del sector.
        - **Sector**: Algunos sectores tienen más vacantes inclusivas en comparación con otros.
        """
    )
    
    # Modelo de Machine Learning
    df_ml = df.copy()
    df_ml['ESPCD'] = df_ml['ESPCD'].apply(lambda x: 1 if x == "SI" else 0)
    encoder = LabelEncoder()
    df_ml['SECTOR'] = encoder.fit_transform(df_ml['SECTOR'])
    X = df_ml[['EXPERIENCIA_MESES', 'SECTOR']].fillna(0)
    y = df_ml['ESPCD']

    # Entrenamiento del modelo
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    
    # Resultados del modelo
    st.write("### Resultados del Modelo")
    st.text(classification_report(y_test, model.predict(X_test)))

    # Matriz de Confusión
    st.write("### Matriz de Confusión del Modelo")
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    fig_cm = go.Figure(data=go.Heatmap(
        z=cm,
        x=["No Inclusiva", "Inclusiva"],
        y=["No Inclusiva", "Inclusiva"],
        colorscale="Blues",
        text=cm,
        texttemplate="%{text}"
    ))
    fig_cm.update_layout(title="Matriz de Confusión", xaxis_title="Predicción", yaxis_title="Actual")
    st.plotly_chart(fig_cm)

    # Interacción para predicción
    st.write("### Predicción Interactiva")
    st.write("Ingresa la cantidad de experiencia en meses y el sector para predecir si una vacante es inclusiva.")
    exp_input = st.slider("Experiencia (en meses)", 0, 60, 10)
    sector_input = st.selectbox("Selecciona el Sector", df['SECTOR'].unique())
    sector_encoded = encoder.transform([sector_input])[0]

        # Explicación adicional para el usuario
    st.write(
        """
        ### Prueba el Modelo
        Modifica los controles para experimentar y entender cómo cambia la predicción de inclusión.
        """
    )

    # Realizar predicción
    prediction = model.predict([[exp_input, sector_encoded]])
    st.write("La vacante es inclusiva 👍" if prediction[0] == 1 else "La vacante no es inclusiva 👎")


