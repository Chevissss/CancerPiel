"""
APLICACIÓN WEB - DETECTOR DE CÁNCER DE PIEL
Framework: Streamlit
Modelo: MobileNetV2
Incluye: Upload, Cámara, Disclaimers médicos, Análisis detallado
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

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 30px;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 15px;
        border-radius: 5px;
        margin: 20px 0;
    }
    .danger-box {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
        padding: 15px;
        border-radius: 5px;
        margin: 20px 0;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 15px;
        border-radius: 5px;
        margin: 20px 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# CARGAR MODELO Y CONFIGURACIÓN
# ============================================================================
@st.cache_resource
def load_model_and_config():
    """Carga el modelo y la configuración (se ejecuta una sola vez)"""
    
    # IMPORTANTE: Ajustar estas rutas según donde tengas los archivos
    MODEL_PATH = "skin_cancer_model_fast.h5"
    CONFIG_PATH = "model_config_fast.json"
    
    try:
        # Cargar configuración
        with open(CONFIG_PATH, 'r') as f:
            config = json.load(f)
        
        # Cargar modelo
        model = keras.models.load_model(MODEL_PATH)
        
        return model, config
    except Exception as e:
        st.error(f"❌ Error al cargar el modelo: {str(e)}")
        st.info("""
        **Archivos requeridos:**
        - `skin_cancer_model_fast.h5`
        - `model_config_fast.json`
        
        Asegúrate de tenerlos en el mismo directorio que esta aplicación.
        """)
        return None, None

model, CONFIG = load_model_and_config()

# ============================================================================
# FUNCIÓN DE PREDICCIÓN
# ============================================================================
def predict_image(image, model, config):
    """
    Realiza predicción sobre una imagen
    """
    # Redimensionar imagen
    img = image.resize(config["INPUT_SHAPE"][:2])
    
    # Convertir a array y normalizar
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array * config["RESCALE"]
    
    # Predicción
    predictions = model.predict(img_array, verbose=0)
    
    # Procesar resultados
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
# HEADER Y DISCLAIMERS
# ============================================================================
st.markdown("""
<div class="main-header">
    <h1>🔬 Detector de Cáncer de Piel</h1>
    <p style="font-size: 1.2em; margin-top: 10px;">Sistema de análisis de lesiones cutáneas con Inteligencia Artificial</p>
</div>
""", unsafe_allow_html=True)

# DISCLAIMER PRINCIPAL - MUY IMPORTANTE
st.markdown("""
<div class="danger-box">
    <h3>⚠️ ADVERTENCIA MÉDICA IMPORTANTE</h3>
    <p><strong>Este sistema es SOLO para fines educativos e informativos.</strong></p>
    <ul>
        <li>❌ <strong>NO</strong> reemplaza el diagnóstico de un dermatólogo profesional</li>
        <li>❌ <strong>NO</strong> debe usarse como única herramienta de diagnóstico</li>
        <li>❌ <strong>NO</strong> es un dispositivo médico certificado</li>
        <li>✅ Siempre consulte con un médico especialista para cualquier lesión cutánea</li>
        <li>✅ El diagnóstico definitivo requiere examen clínico y posiblemente biopsia</li>
    </ul>
    <p><strong>En caso de duda, consulte inmediatamente a un dermatólogo.</strong></p>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# SIDEBAR - INFORMACIÓN Y OPCIONES
# ============================================================================
with st.sidebar:
    st.header("📋 Información del Sistema")
    
    if CONFIG:
        st.markdown(f"""
        **🤖 Modelo:** {CONFIG['MODEL_ARCHITECTURE']}  
        **📏 Tamaño de entrada:** {CONFIG['INPUT_SHAPE'][0]}x{CONFIG['INPUT_SHAPE'][1]}  
        **🎯 Clases:** {', '.join(CONFIG['CLASSES'])}  
        """)
        
        with st.expander("⚙️ Parámetros Técnicos"):
            st.json({
                "Arquitectura": CONFIG["MODEL_ARCHITECTURE"],
                "Input Shape": CONFIG["INPUT_SHAPE"],
                "Batch Size": CONFIG["BATCH_SIZE"],
                "Épocas": CONFIG["EPOCHS"],
                "Learning Rate": CONFIG["INITIAL_LEARNING_RATE"],
                "Dropout": CONFIG["DROPOUT_RATE"]
            })
    
    st.markdown("---")
    
    st.markdown("""
    ### 📖 Cómo usar
    
    1. **Sube una foto** o **toma una con la cámara**
    2. Asegúrate de que la imagen sea clara
    3. Presiona **"🔍 Analizar"**
    4. Revisa los resultados
    5. **Consulta a un médico**
    
    ### ⚕️ Recuerda
    
    - Esta herramienta NO es un diagnóstico médico
    - Siempre busca atención profesional
    - El cáncer de piel es tratable si se detecta temprano
    """)
    
    st.markdown("---")
    
    st.markdown("""
    <div style='text-align: center; color: gray; font-size: 0.8em;'>
        <p><strong>Desarrollado con fines educativos</strong></p>
        <p>Powered by TensorFlow & Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# VERIFICAR MODELO
# ============================================================================
if model is None or CONFIG is None:
    st.error("⚠️ No se pudo cargar el modelo. Verifica que los archivos estén disponibles.")
    st.stop()

# ============================================================================
# ÁREA PRINCIPAL - OPCIONES DE CARGA DE IMAGEN
# ============================================================================

st.markdown("## 📸 Cargar Imagen para Análisis")

# Tabs para diferentes métodos de carga
tab1, tab2 = st.tabs(["📁 Subir Archivo", "📷 Usar Cámara"])

uploaded_image = None

with tab1:
    st.markdown("""
    <div class="warning-box">
        <strong>💡 Consejos para mejores resultados:</strong>
        <ul>
            <li>Usa buena iluminación natural</li>
            <li>Enfoca bien la lesión</li>
            <li>Evita sombras o reflejos</li>
            <li>La imagen debe estar clara y nítida</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Selecciona una imagen de la lesión cutánea",
        type=['jpg', 'jpeg', 'png'],
        help="Formatos soportados: JPG, JPEG, PNG"
    )
    
    if uploaded_file is not None:
        uploaded_image = Image.open(uploaded_file)
        st.image(uploaded_image, caption="Imagen cargada", use_container_width=True)

with tab2:
    st.markdown("""
    <div class="warning-box">
        <strong>📱 Para dispositivos móviles:</strong>
        <ul>
            <li>Permite el acceso a la cámara cuando se solicite</li>
            <li>Usa la cámara trasera para mejor calidad</li>
            <li>Mantén el teléfono estable al tomar la foto</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    camera_image = st.camera_input("Toma una foto de la lesión")
    
    if camera_image is not None:
        uploaded_image = Image.open(camera_image)
        st.success("✅ Foto capturada exitosamente")

# ============================================================================
# ANÁLISIS DE LA IMAGEN
# ============================================================================

if uploaded_image is not None:
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        analyze_button = st.button(
            "🔍 ANALIZAR IMAGEN", 
            type="primary", 
            use_container_width=True,
            help="Haz clic para iniciar el análisis con IA"
        )
    
    if analyze_button:
        # Advertencia antes del análisis
        st.markdown("""
        <div class="warning-box">
            <p><strong>⏳ Analizando imagen...</strong></p>
            <p>Recuerda: Este análisis NO reemplaza la opinión de un médico profesional.</p>
        </div>
        """, unsafe_allow_html=True)
        
        with st.spinner("🧠 Procesando con Inteligencia Artificial..."):
            result = predict_image(uploaded_image, model, CONFIG)
        
        st.success("✅ Análisis completado")
        
        # ============================================================================
        # RESULTADOS DEL ANÁLISIS
        # ============================================================================
        
        st.markdown("---")
        st.markdown("## 📊 Resultados del Análisis")
        
        # Determinar color y mensaje según resultado
        if result['prediccion'] == 'benign':
            color = "#28a745"
            bg_color = "#d4edda"
            emoji = "✅"
            mensaje = "BENIGNO"
            interpretacion = """
            <div class="success-box">
                <h4>✅ Resultado: BENIGNO</h4>
                <p><strong>El sistema ha identificado características consistentes con una lesión benigna.</strong></p>
                <p><strong>⚠️ IMPORTANTE:</strong> Aunque el resultado indica características benignas, 
                se recomienda fuertemente consultar con un dermatólogo para:</p>
                <ul>
                    <li>Confirmación del diagnóstico mediante examen clínico</li>
                    <li>Seguimiento periódico de la lesión</li>
                    <li>Evaluación de factores de riesgo personales</li>
                </ul>
            </div>
            """
        else:
            color = "#dc3545"
            bg_color = "#f8d7da"
            emoji = "⚠️"
            mensaje = "SOSPECHOSO / MALIGNO"
            interpretacion = """
            <div class="danger-box">
                <h4>⚠️ Resultado: SOSPECHOSO / MALIGNO</h4>
                <p><strong>El sistema ha detectado características que podrían ser preocupantes.</strong></p>
                <p><strong>🚨 ACCIÓN REQUERIDA:</strong></p>
                <ul>
                    <li><strong>Consulte INMEDIATAMENTE con un dermatólogo</strong></li>
                    <li>Solicite una evaluación clínica completa</li>
                    <li>Puede ser necesaria una biopsia para diagnóstico definitivo</li>
                    <li>No ignore este resultado ni demore la consulta médica</li>
                </ul>
                <p><strong>⏰ El diagnóstico temprano es crucial para el tratamiento exitoso.</strong></p>
            </div>
            """
        
        # Mostrar resultado principal
        st.markdown(f"""
        <div style='background-color: {bg_color}; padding: 30px; border-radius: 15px; 
                    border: 3px solid {color}; text-align: center; margin: 20px 0;'>
            <h1 style='color: {color}; margin: 0;'>{emoji} {mensaje}</h1>
            <h2 style='color: {color}; margin-top: 10px;'>Confianza del modelo: {result['confianza']:.2f}%</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Crear dos columnas para la visualización
        col_img, col_metrics = st.columns([1, 1])
        
        with col_img:
            st.markdown("### 📸 Imagen Analizada")
            st.image(uploaded_image, use_container_width=True)
        
        with col_metrics:
            st.markdown("### 📈 Probabilidades Detalladas")
            
            # Métricas de probabilidad
            st.markdown(f"""
            <div class="metric-card">
                <h4 style='color: #28a745; margin-bottom: 10px;'>🟢 Probabilidad BENIGNO</h4>
                <h2 style='color: #28a745; margin: 0;'>{result['prob_benign']:.2f}%</h2>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="metric-card">
                <h4 style='color: #dc3545; margin-bottom: 10px;'>🔴 Probabilidad MALIGNO</h4>
                <h2 style='color: #dc3545; margin: 0;'>{result['prob_malignant']:.2f}%</h2>
            </div>
            """, unsafe_allow_html=True)
            
            # Nivel de confianza
            if result['confianza'] >= 90:
                nivel = "MUY ALTA"
                nivel_color = "#28a745"
            elif result['confianza'] >= 75:
                nivel = "ALTA"
                nivel_color = "#ffc107"
            elif result['confianza'] >= 60:
                nivel = "MODERADA"
                nivel_color = "#fd7e14"
            else:
                nivel = "BAJA"
                nivel_color = "#dc3545"
            
            st.markdown(f"""
            <div class="metric-card">
                <h4 style='margin-bottom: 10px;'>📊 Nivel de Confianza del Modelo</h4>
                <h2 style='color: {nivel_color}; margin: 0;'>{nivel}</h2>
                <p style='margin-top: 10px; color: gray;'>
                    El modelo tiene una confianza del {result['confianza']:.2f}% en este resultado.
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        # Gráfico de barras de probabilidades
        st.markdown("### 📊 Distribución de Probabilidades")
        
        import plotly.graph_objects as go
        
        fig = go.Figure(data=[
            go.Bar(
                x=['Benigno', 'Maligno'],
                y=[result['prob_benign'], result['prob_malignant']],
                marker_color=['#28a745', '#dc3545'],
                text=[f"{result['prob_benign']:.2f}%", f"{result['prob_malignant']:.2f}%"],
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title="Probabilidades de Clasificación",
            yaxis_title="Probabilidad (%)",
            yaxis_range=[0, 100],
            showlegend=False,
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Interpretación del resultado
        st.markdown("---")
        st.markdown("### 💡 Interpretación del Resultado")
        st.markdown(interpretacion, unsafe_allow_html=True)
        
        # Información adicional
        st.markdown("---")
        st.markdown("### 📚 Información Adicional")
        
        with st.expander("ℹ️ Sobre el Cáncer de Piel"):
            st.markdown("""
            **Tipos comunes de cáncer de piel:**
            
            1. **Melanoma:** El más peligroso, pero curable si se detecta temprano
            2. **Carcinoma Basocelular:** El más común, generalmente de crecimiento lento
            3. **Carcinoma Escamocelular:** Común, puede ser agresivo
            
            **Signos de advertencia (Regla ABCDE):**
            - **A**simetría: Una mitad diferente a la otra
            - **B**ordes: Irregulares, borrosos o dentados
            - **C**olor: Varios colores o distribución desigual
            - **D**iámetro: Mayor a 6mm (tamaño de un borrador)
            - **E**volución: Cambios en tamaño, forma o color
            
            **🚨 Consulta inmediatamente si observas:**
            - Lunares que cambian
            - Sangrado o picazón
            - Heridas que no sanan
            - Nuevas lesiones pigmentadas
            """)
        
        with st.expander("🔬 Sobre este Sistema"):
            st.markdown(f"""
            **Detalles Técnicos:**
            - **Modelo:** {CONFIG['MODEL_ARCHITECTURE']}
            - **Entrenado con:** Transfer Learning
            - **Dataset:** Imágenes de lesiones cutáneas clasificadas
            - **Precisión en test:** Variable según los datos
            
            **Limitaciones:**
            - Solo analiza imágenes, no realiza examen físico
            - No considera historial médico del paciente
            - Puede dar falsos positivos o negativos
            - No detecta todos los tipos de cáncer de piel
            - Calidad de imagen afecta el resultado
            
            **Este sistema NO sustituye:**
            - Examen dermatológico profesional
            - Dermatoscopia
            - Biopsia
            - Análisis histopatológico
            """)
        
        # Botón para descargar resultados
        st.markdown("---")
        
        resultado_texto = f"""
RESULTADO DEL ANÁLISIS - DETECTOR DE CÁNCER DE PIEL

⚠️ ESTE RESULTADO ES SOLO INFORMATIVO Y NO REEMPLAZA EL DIAGNÓSTICO MÉDICO

Fecha del análisis: {result.get('fecha', 'N/A')}

RESULTADO: {mensaje}
Confianza del modelo: {result['confianza']:.2f}%

PROBABILIDADES:
- Benigno: {result['prob_benign']:.2f}%
- Maligno: {result['prob_malignant']:.2f}%

RECOMENDACIÓN:
Independientemente del resultado, se recomienda consultar con un dermatólogo 
profesional para una evaluación clínica completa.

---
Sistema: {CONFIG['MODEL_ARCHITECTURE']}
Este es un sistema de ayuda diagnóstica NO certificado como dispositivo médico.
        """
        
        st.download_button(
            label="💾 Descargar Resultados (TXT)",
            data=resultado_texto,
            file_name=f"analisis_lesion_{result['prediccion']}.txt",
            mime="text/plain",
            use_container_width=True
        )

else:
    # Mensaje cuando no hay imagen cargada
    st.info("👆 Por favor, carga una imagen o toma una foto con la cámara para comenzar el análisis.")

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 20px; background-color: #f8f9fa; border-radius: 10px;'>
    <h3>⚕️ Recordatorio Final</h3>
    <p style='font-size: 1.1em; color: #dc3545; font-weight: bold;'>
        Esta herramienta NO diagnostica cáncer de piel. 
        Solo un dermatólogo puede proporcionar un diagnóstico definitivo.
    </p>
    <p style='color: #666; margin-top: 15px;'>
        Si tienes alguna preocupación sobre una lesión cutánea, 
        consulta inmediatamente con un profesional de la salud.
    </p>
    <hr style='margin: 20px 0;'>
    <p style='font-size: 0.9em; color: gray;'>
        <strong>Desarrollado con fines educativos</strong><br>
        Tecnología: TensorFlow, Keras, Streamlit<br>
        Modelo: {CONFIG['MODEL_ARCHITECTURE'] if CONFIG else 'N/A'}
    </p>
</div>
""", unsafe_allow_html=True)