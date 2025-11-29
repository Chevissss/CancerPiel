"""
APLICACIÓN WEB - DETECTOR DE CÁNCER DE PIEL
Framework: Streamlit
Modelo: MobileNetV2
Versión Mejorada - UI Optimizada
"""

import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import json
import os
import io

# ============================================================================
# CONFIGURACIÓN DE LA PÁGINA
# ============================================================================
st.set_page_config(
    page_title="🔬 Detector de Cáncer de Piel",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado mejorado
st.markdown("""
<style>
    /* Fuentes y colores base */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        text-align: center;
        padding: 40px 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 16px;
        margin-bottom: 30px;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
    }
    
    .main-header h1 {
        font-size: 2.5em;
        margin-bottom: 10px;
        font-weight: 700;
    }
    
    .main-header p {
        font-size: 1.1em;
        opacity: 0.95;
    }
    
    /* Alertas mejoradas */
    .alert-box {
        padding: 20px 24px;
        border-radius: 12px;
        margin: 24px 0;
        border-left: 5px solid;
        background: white;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    
    .alert-warning {
        border-color: #ff9800;
        background: linear-gradient(to right, #fff8e1, #ffffff);
    }
    
    .alert-danger {
        border-color: #f44336;
        background: linear-gradient(to right, #ffebee, #ffffff);
    }
    
    .alert-success {
        border-color: #4caf50;
        background: linear-gradient(to right, #e8f5e9, #ffffff);
    }
    
    .alert-info {
        border-color: #2196f3;
        background: linear-gradient(to right, #e3f2fd, #ffffff);
    }
    
    /* Tarjetas de métricas */
    .metric-card {
        background: white;
        padding: 24px;
        border-radius: 12px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.08);
        margin: 12px 0;
        border: 1px solid #e0e0e0;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(0,0,0,0.12);
    }
    
    .metric-card h4 {
        margin: 0 0 12px 0;
        font-size: 1em;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .metric-card h2 {
        margin: 0;
        font-size: 2.5em;
        font-weight: 700;
    }
    
    /* Resultado principal */
    .result-main {
        padding: 40px 30px;
        border-radius: 16px;
        text-align: center;
        margin: 30px 0;
        border: 3px solid;
        box-shadow: 0 8px 32px rgba(0,0,0,0.12);
    }
    
    .result-main h1 {
        margin: 0 0 12px 0;
        font-size: 2.8em;
        font-weight: 700;
    }
    
    .result-main h3 {
        margin: 0;
        font-size: 1.4em;
        opacity: 0.9;
    }
    
    /* Badges */
    .badge {
        display: inline-block;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9em;
        margin: 4px;
    }
    
    .badge-success {
        background: #4caf50;
        color: white;
    }
    
    .badge-danger {
        background: #f44336;
        color: white;
    }
    
    .badge-warning {
        background: #ff9800;
        color: white;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8f9fa 0%, #ffffff 100%);
    }
    
    /* Botones */
    .stButton > button {
        font-weight: 600;
        border-radius: 8px;
        padding: 12px 24px;
        font-size: 1.1em;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        font-weight: 600;
        font-size: 1em;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# CARGAR MODELO Y CONFIGURACIÓN
# ============================================================================
@st.cache_resource
def load_model_and_config():
    """Carga el modelo y la configuración (se ejecuta una sola vez)"""
    
    MODEL_PATH = "skin_cancer_model_fast.h5"
    CONFIG_PATH = "model_config_fast.json"
    
    try:
        with open(CONFIG_PATH, 'r') as f:
            config = json.load(f)
        
        model = keras.models.load_model(MODEL_PATH)
        
        return model, config
    except Exception as e:
        st.error(f"❌ Error al cargar el modelo: {str(e)}")
        st.info("""
        **Archivos requeridos:**
        - `skin_cancer_model_fast.h5`
        - `model_config_fast.json`
        """)
        return None, None

model, CONFIG = load_model_and_config()

# ============================================================================
# FUNCIÓN DE PREDICCIÓN
# ============================================================================
def predict_image(image, model, config):
    """Realiza predicción sobre una imagen"""
    img = image.resize(config["INPUT_SHAPE"][:2])
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array * config["RESCALE"]
    
    predictions = model.predict(img_array, verbose=0)
    
    predicted_class_idx = np.argmax(predictions[0])
    predicted_class = config["CLASSES"][predicted_class_idx]
    confidence = predictions[0][predicted_class_idx] * 100
    
    return {
        "prediccion": predicted_class,
        "confianza": confidence,
        "prob_benign": predictions[0][0] * 100,
        "prob_malignant": predictions[0][1] * 100,
        "raw_predictions": predictions[0]
    }

# ============================================================================
# HEADER
# ============================================================================
st.markdown("""
<div class="main-header">
    <h1>🔬 Detector de Cáncer de Piel</h1>
    <p>Análisis de lesiones cutáneas con Inteligencia Artificial</p>
</div>
""", unsafe_allow_html=True)

# DISCLAIMER PRINCIPAL
st.markdown("""
<div class="alert-box alert-danger">
    <h3 style="margin-top: 0; color: #d32f2f;">⚠️ Advertencia Médica</h3>
    <p style="font-size: 1.05em; line-height: 1.6; margin-bottom: 8px;">
        <strong>Este sistema es solo para fines educativos.</strong> 
        NO reemplaza el diagnóstico médico profesional ni debe usarse como única herramienta de diagnóstico.
    </p>
    <p style="font-size: 1.05em; margin: 0; color: #d32f2f;">
        <strong>→ Siempre consulte con un dermatólogo para cualquier lesión cutánea.</strong>
    </p>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.header("📋 Información")
    
    if CONFIG:
        st.markdown(f"""
        **Modelo:** {CONFIG['MODEL_ARCHITECTURE']}  
        **Entrada:** {CONFIG['INPUT_SHAPE'][0]}×{CONFIG['INPUT_SHAPE'][1]} px  
        **Clases:** {', '.join(CONFIG['CLASSES'])}  
        """)
    
    st.markdown("---")
    
    st.markdown("""
    ### 📖 Cómo usar
    
    1. Carga una imagen o toma una foto
    2. Presiona **"Analizar"**
    3. Revisa los resultados
    4. **Consulta a un dermatólogo**
    
    ### 🎯 Consejos para la foto
    
    - Buena iluminación natural
    - Imagen clara y enfocada
    - Sin sombras ni reflejos
    - Acercamiento adecuado
    """)
    
    with st.expander("ℹ️ Regla ABCDE"):
        st.markdown("""
        **Señales de alerta:**
        - **A**simetría
        - **B**ordes irregulares
        - **C**olor variado
        - **D**iámetro >6mm
        - **E**volución/cambios
        """)

# ============================================================================
# VERIFICAR MODELO
# ============================================================================
if model is None or CONFIG is None:
    st.error("⚠️ No se pudo cargar el modelo.")
    st.stop()

# ============================================================================
# ÁREA DE CARGA
# ============================================================================

st.markdown("## 📸 Cargar Imagen")

tab1, tab2 = st.tabs(["📁 Subir Archivo", "📷 Usar Cámara"])

uploaded_image = None

with tab1:
    uploaded_file = st.file_uploader(
        "Selecciona una imagen (JPG, PNG)",
        type=['jpg', 'jpeg', 'png'],
        label_visibility="collapsed"
    )
    
    if uploaded_file is not None:
        uploaded_image = Image.open(uploaded_file)
        st.image(uploaded_image, caption="Imagen cargada", use_container_width=True)

with tab2:
    camera_image = st.camera_input("Toma una foto")
    
    if camera_image is not None:
        uploaded_image = Image.open(camera_image)

# ============================================================================
# ANÁLISIS
# ============================================================================

if uploaded_image is not None:
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        analyze_button = st.button(
            "🔍 ANALIZAR IMAGEN", 
            type="primary", 
            use_container_width=True
        )
    
    if analyze_button:
        with st.spinner("🧠 Analizando..."):
            result = predict_image(uploaded_image, model, CONFIG)
        
        # ============================================================================
        # RESULTADOS
        # ============================================================================
        
        st.markdown("---")
        
        # Configuración según resultado
        if result['prediccion'] == 'benign':
            color = "#4caf50"
            bg_color = "#e8f5e9"
            emoji = "✅"
            mensaje = "BENIGNO"
            badge_class = "badge-success"
        else:
            color = "#f44336"
            bg_color = "#ffebee"
            emoji = "⚠️"
            mensaje = "REQUIERE ATENCIÓN"
            badge_class = "badge-danger"
        
        # Resultado principal
        st.markdown(f"""
        <div class="result-main" style="background: {bg_color}; border-color: {color};">
            <h1 style="color: {color};">{emoji} {mensaje}</h1>
            <h3 style="color: {color};">Confianza: {result['confianza']:.1f}%</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Layout de resultados
        col_img, col_metrics = st.columns([1, 1])
        
        with col_img:
            st.markdown("### 📸 Imagen Analizada")
            st.image(uploaded_image, use_container_width=True)
        
        with col_metrics:
            st.markdown("### 📊 Probabilidades")
            
            # Benigno
            st.markdown(f"""
            <div class="metric-card">
                <h4 style="color: #4caf50;">🟢 Benigno</h4>
                <h2 style="color: #4caf50;">{result['prob_benign']:.1f}%</h2>
            </div>
            """, unsafe_allow_html=True)
            
            # Maligno
            st.markdown(f"""
            <div class="metric-card">
                <h4 style="color: #f44336;">🔴 Maligno</h4>
                <h2 style="color: #f44336;">{result['prob_malignant']:.1f}%</h2>
            </div>
            """, unsafe_allow_html=True)
            
            # Nivel de confianza
            if result['confianza'] >= 90:
                nivel = "MUY ALTA"
                nivel_color = "#4caf50"
            elif result['confianza'] >= 75:
                nivel = "ALTA"
                nivel_color = "#ff9800"
            elif result['confianza'] >= 60:
                nivel = "MODERADA"
                nivel_color = "#ff9800"
            else:
                nivel = "BAJA"
                nivel_color = "#f44336"
            
            st.markdown(f"""
            <div class="metric-card">
                <h4>📈 Confianza del Modelo</h4>
                <h2 style="color: {nivel_color};">{nivel}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        # Gráfico
        st.markdown("### 📊 Distribución")
        
        import plotly.graph_objects as go
        
        fig = go.Figure(data=[
            go.Bar(
                x=['Benigno', 'Maligno'],
                y=[result['prob_benign'], result['prob_malignant']],
                marker_color=['#4caf50', '#f44336'],
                text=[f"{result['prob_benign']:.1f}%", f"{result['prob_malignant']:.1f}%"],
                textposition='auto',
                textfont=dict(size=16, color='white', family='Inter')
            )
        ])
        
        fig.update_layout(
            yaxis_title="Probabilidad (%)",
            yaxis_range=[0, 100],
            showlegend=False,
            height=350,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Inter', size=14)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Recomendación
        st.markdown("---")
        st.markdown("### 💡 Recomendación")
        
        if result['prediccion'] == 'benign':
            st.markdown("""
            <div class="alert-box alert-success">
                <p style="font-size: 1.05em; margin: 0; line-height: 1.6;">
                    <strong>El análisis sugiere características benignas.</strong> 
                    Sin embargo, se recomienda consultar con un dermatólogo para confirmar 
                    el diagnóstico y establecer un seguimiento adecuado.
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="alert-box alert-danger">
                <p style="font-size: 1.05em; margin: 0 0 12px 0; line-height: 1.6;">
                    <strong>El análisis detectó características que requieren atención médica.</strong>
                </p>
                <p style="font-size: 1.05em; margin: 0; color: #d32f2f; line-height: 1.6;">
                    <strong>→ Consulte INMEDIATAMENTE con un dermatólogo</strong> para 
                    una evaluación completa. El diagnóstico temprano es crucial.
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        # Información adicional
        with st.expander("📚 Información Adicional"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                **Tipos comunes:**
                - Melanoma
                - Carcinoma Basocelular
                - Carcinoma Escamocelular
                
                **Cuándo consultar:**
                - Lunares que cambian
                - Sangrado o picazón
                - Heridas que no sanan
                """)
            
            with col2:
                st.markdown("""
                **Limitaciones del sistema:**
                - No considera historial médico
                - Solo analiza la imagen
                - Puede tener falsos positivos/negativos
                - No sustituye examen profesional
                """)
        
        # Descargar resultados
        st.markdown("---")
        
        resultado_texto = f"""
ANÁLISIS - DETECTOR DE CÁNCER DE PIEL
⚠️ SOLO PARA FINES EDUCATIVOS - NO ES DIAGNÓSTICO MÉDICO

RESULTADO: {mensaje}
Confianza: {result['confianza']:.2f}%

PROBABILIDADES:
- Benigno: {result['prob_benign']:.2f}%
- Maligno: {result['prob_malignant']:.2f}%

RECOMENDACIÓN:
Consultar con un dermatólogo profesional.

Sistema: {CONFIG['MODEL_ARCHITECTURE']}
        """
        
        st.download_button(
            label="💾 Descargar Resultados",
            data=resultado_texto,
            file_name=f"analisis_{result['prediccion']}.txt",
            mime="text/plain",
            use_container_width=True
        )

else:
    st.info("👆 Carga una imagen para comenzar el análisis")

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 30px 20px; background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); border-radius: 12px;">
    <p style="font-size: 1.1em; color: #d32f2f; font-weight: 600; margin: 0 0 12px 0;">
        ⚕️ Este sistema NO diagnostica cáncer de piel
    </p>
    <p style="color: #666; margin: 0 0 20px 0; line-height: 1.6;">
        Solo un dermatólogo profesional puede proporcionar un diagnóstico definitivo.
        Ante cualquier duda, consulta inmediatamente con un especialista.
    </p>
    <p style="font-size: 0.9em; color: #999; margin: 0;">
        Desarrollado con TensorFlow & Streamlit | Fines Educativos
    </p>
</div>
""", unsafe_allow_html=True)
