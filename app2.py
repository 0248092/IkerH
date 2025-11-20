from __future__ import annotations
import warnings
from datetime import datetime
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import google.generativeai as genai
import os
import time
import json

warnings.filterwarnings("ignore")

# =========================
# CONFIGURACIÓN
# =========================
st.set_page_config(
    page_title="Asesor de Portafolio con IA",
    layout="wide",
    page_icon="🤖"
)

# Configurar Gemini
try:
    from dotenv import load_dotenv
    load_dotenv()
except:
    pass

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY", "")

if not GEMINI_API_KEY:
    st.error("❌ No se encontró GEMINI_API_KEY")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash")  # ✅ Actualizado a 2.5

# =========================
# ESTILOS CSS MEJORADOS
# =========================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;900&display=swap');
    
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 3rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 20px 60px rgba(102, 126, 234, 0.4);
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 3rem;
        font-weight: 900;
        letter-spacing: -1px;
    }
    
    .main-header p {
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
        opacity: 0.95;
    }
    
    .questionnaire-card {
        background: white;
        padding: 2.5rem;
        border-radius: 16px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.08);
        margin-bottom: 2rem;
        border: 1px solid #e0e0e0;
    }
    
    .ai-section {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 2.5rem;
        border-radius: 16px;
        color: white;
        margin: 2rem 0;
        box-shadow: 0 12px 30px rgba(245, 87, 108, 0.3);
    }
    
    .ai-section h2 {
        margin: 0 0 0.5rem 0;
        font-size: 2rem;
        font-weight: 800;
    }
    
    .metric-box {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.06);
        border: 2px solid #e9ecef;
        transition: all 0.3s ease;
    }
    
    .metric-box:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 24px rgba(102, 126, 234, 0.15);
        border-color: #667eea;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 900;
        color: #667eea;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        color: #6c757d;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 600;
    }
    
    .metric-sublabel {
        color: #adb5bd;
        font-size: 0.75rem;
        margin-top: 0.25rem;
    }
    
    .enhanced-table {
        background: white;
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        margin: 1.5rem 0;
    }
    
    .stDataFrame {
        font-size: 0.95rem;
    }
    
    .stDataFrame th {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        font-weight: 700 !important;
        padding: 1rem !important;
        text-transform: uppercase;
        font-size: 0.85rem;
        letter-spacing: 0.5px;
    }
    
    .stDataFrame td {
        padding: 1rem !important;
        border-bottom: 1px solid #f0f0f0 !important;
    }
    
    .stDataFrame tr:hover {
        background: #f8f9fa !important;
    }
    
    .success-badge {
        display: inline-block;
        background: #d4edda;
        color: #155724;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
        margin: 0.5rem 0;
    }
    
    .info-card {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #2196F3;
        margin: 1rem 0;
    }
    
    .warning-card {
        background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #ff9800;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# =========================
# FUNCIONES DE DATOS
# =========================

@st.cache_data(ttl=3600)
def get_company_details(ticker: str) -> Dict:
    """Obtiene detalles de la empresa"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        return {
            'nombre': info.get('longName', info.get('shortName', ticker)),
            'sector': info.get('sector', 'N/D'),
            'industria': info.get('industry', 'N/D'),
            'empleados': info.get('fullTimeEmployees', 0),
            'descripcion': info.get('longBusinessSummary', '')
        }
    except:
        return {
            'nombre': ticker,
            'sector': 'N/D',
            'industria': 'N/D',
            'empleados': 0,
            'descripcion': ''
        }

@st.cache_data(ttl=3600)
def get_batch_stock_data(tickers: List[str], period: str = "1y") -> Dict:
    """Descarga datos en batch"""
    try:
        tickers_str = " ".join(tickers)
        data = yf.download(
            tickers_str,
            period=period,
            group_by='ticker',
            auto_adjust=True,
            progress=False,
            threads=False
        )
        
        result = {}
        for ticker in tickers:
            try:
                if len(tickers) == 1:
                    ticker_data = data
                else:
                    ticker_data = data[ticker] if ticker in data.columns.get_level_values(0) else pd.DataFrame()
                
                if not ticker_data.empty and 'Close' in ticker_data.columns:
                    result[ticker] = {
                        'history': ticker_data,
                        'current_price': float(ticker_data['Close'].iloc[-1]),
                        'success': True
                    }
                else:
                    result[ticker] = {'success': False, 'history': pd.DataFrame()}
            except:
                result[ticker] = {'success': False, 'history': pd.DataFrame()}
        
        return result
    except Exception as e:
        return {}

def calculate_metrics(prices: pd.DataFrame) -> Dict:
    """Calcula métricas financieras"""
    if prices.empty or 'Close' not in prices.columns:
        return {}
    
    returns = prices['Close'].pct_change().dropna()
    
    metrics = {
        'rendimiento_total': ((prices['Close'].iloc[-1] / prices['Close'].iloc[0]) - 1) * 100,
        'volatilidad_anual': returns.std() * np.sqrt(252) * 100,
        'rendimiento_anual': returns.mean() * 252 * 100,
        'max_drawdown': ((prices['Close'] / prices['Close'].cummax()) - 1).min() * 100,
        'sharpe_ratio': (returns.mean() * 252) / (returns.std() * np.sqrt(252)) if returns.std() != 0 else 0,
        'returns': returns
    }
    
    return metrics

def calculate_portfolio_metrics(batch_data: Dict, weights: Dict, periodo: str) -> Dict:
    """Calcula métricas del portafolio ponderado"""
    portfolio_returns = None
    dates = None
    
    for ticker, weight in weights.items():
        if ticker in batch_data and batch_data[ticker]['success']:
            hist = batch_data[ticker]['history']
            if not hist.empty and 'Close' in hist.columns:
                returns = hist['Close'].pct_change().dropna()
                
                if portfolio_returns is None:
                    portfolio_returns = returns * weight
                    dates = returns.index
                else:
                    portfolio_returns = portfolio_returns.add(returns * weight, fill_value=0)
    
    if portfolio_returns is None or portfolio_returns.empty:
        return {}
    
    cumulative_returns = (1 + portfolio_returns).cumprod()
    
    metrics = {
        'rendimiento_total': (cumulative_returns.iloc[-1] - 1) * 100,
        'volatilidad_anual': portfolio_returns.std() * np.sqrt(252) * 100,
        'rendimiento_anual': portfolio_returns.mean() * 252 * 100,
        'max_drawdown': ((cumulative_returns / cumulative_returns.cummax()) - 1).min() * 100,
        'sharpe_ratio': (portfolio_returns.mean() * 252) / (portfolio_returns.std() * np.sqrt(252)) if portfolio_returns.std() != 0 else 0,
        'cumulative_returns': cumulative_returns,
        'dates': dates
    }
    
    return metrics

# =========================
# FUNCIONES DE IA
# =========================

def generate_portfolio_with_gemini(perfil: str, horizonte: str, capital: str, objetivos: List[str], sector_pref: str) -> Tuple[List[Dict], str]:
    """Genera un portafolio sugerido con Gemini"""
    
    objetivos_str = ", ".join(objetivos)
    
    prompt = f"""
    Eres un asesor financiero experto. Basándote en el siguiente perfil de inversionista, sugiere un portafolio óptimo de inversión.
    
    **Perfil del Inversionista:**
    - Perfil de riesgo: {perfil}
    - Horizonte temporal: {horizonte}
    - Capital aproximado: {capital}
    - Objetivos: {objetivos_str}
    - Preferencia sectorial: {sector_pref}
    
    **Instrucciones:**
    1. Sugiere entre 5 y 10 acciones/ETFs para maximizar rendimiento ajustado por riesgo
    2. Para perfil Conservador: prioriza ETFs diversificados, empresas blue-chip, dividendos
    3. Para perfil Moderado: balance entre crecimiento y estabilidad
    4. Para perfil Agresivo: empresas de alto crecimiento, tecnología, emergentes
    5. Considera el horizonte temporal para la estrategia
    6. Diversifica por sectores
    
    **Responde ÚNICAMENTE con un JSON válido en este formato exacto:**
```json
    {{
        "portafolio": [
            {{"ticker": "AAPL", "peso": 15, "razon": "Líder en tecnología con sólidos fundamentales"}},
            {{"ticker": "MSFT", "peso": 20, "razon": "Crecimiento estable en cloud computing"}},
            ...
        ],
        "justificacion": "Este portafolio está diseñado para..."
    }}
```
    
    **IMPORTANTE:** 
    - Los pesos deben sumar exactamente 100
    - Solo tickers válidos de NYSE/NASDAQ
    - Incluye 5-10 posiciones
    - NO incluyas texto adicional, solo el JSON
    """
    
    try:
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        
        # Limpiar markdown
        response_text = response_text.replace('```json', '').replace('```', '').strip()
        
        # Parsear JSON
        portfolio_data = json.loads(response_text)
        
        portafolio = portfolio_data.get('portafolio', [])
        justificacion = portfolio_data.get('justificacion', '')
        
        # Validar y normalizar pesos
        total_peso = sum([p['peso'] for p in portafolio])
        if abs(total_peso - 100) > 1:
            for p in portafolio:
                p['peso'] = (p['peso'] / total_peso) * 100
        
        return portafolio, justificacion
        
    except Exception as e:
        st.error(f"Error generando portafolio: {str(e)}")
        return [], ""

def analyze_portfolio_with_gemini(portfolio_data: pd.DataFrame, metrics: Dict, perfil: str, justificacion_inicial: str) -> str:
    """Análisis final del portafolio con Gemini - MÁXIMO 300 PALABRAS"""
    
    prompt = f"""
    Como asesor financiero experto, realiza un análisis CONCISO del portafolio que sugeriste.
    
    **Portafolio Sugerido Inicialmente:**
    {justificacion_inicial}
    
    **Datos Reales del Portafolio:**
    {portfolio_data.to_string()}
    
    **Métricas del Portafolio:**
    - Rendimiento Total: {metrics.get('rendimiento_total', 0):.2f}%
    - Rendimiento Anualizado: {metrics.get('rendimiento_anual', 0):.2f}%
    - Volatilidad Anual: {metrics.get('volatilidad_anual', 0):.2f}%
    - Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}
    - Max Drawdown: {metrics.get('max_drawdown', 0):.2f}%
    
    **Perfil del Inversionista:** {perfil}
    
    Proporciona un análisis de MÁXIMO 300 PALABRAS que incluya:
    
    1. **Evaluación del Desempeño** (50 palabras)
       - ¿El portafolio cumple con el perfil de riesgo?
       - Análisis del Sharpe Ratio
    
    2. **Análisis por Posición** (80 palabras)
       - Top 2 mejores/peores posiciones
       - Ajustes de ponderación
    
    3. **Gestión de Riesgo** (70 palabras)
       - Evaluación del Max Drawdown
       - Recomendaciones de diversificación
    
    4. **Recomendaciones y Conclusión** (100 palabras)
       - Acciones específicas: mantener/aumentar/reducir
       - Próximos pasos inmediatos
    
    **MUY IMPORTANTE:** Sé directo, específico y conciso. NO excedas las 300 palabras. Usa bullets solo si es necesario.
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generando análisis: {str(e)}"

# =========================
# INTERFAZ PRINCIPAL
# =========================

st.markdown("""
<div class="main-header">
    <h1>🤖 Asesor de Portafolio con IA</h1>
    <p>Portafolios personalizados generados por Gemini 2.5 Flash</p>
</div>
""", unsafe_allow_html=True)

# =========================
# CUESTIONARIO
# =========================

st.markdown('<div class="questionnaire-card">', unsafe_allow_html=True)
st.header("📋 Cuestionario de Perfil de Inversión")
st.write("Responde las siguientes preguntas para que la IA diseñe tu portafolio óptimo:")

col1, col2 = st.columns(2)

with col1:
    perfil = st.selectbox(
        "🎯 ¿Cuál es tu perfil de riesgo?",
        ["Conservador", "Moderado", "Agresivo"],
        help="Conservador: Bajo riesgo | Moderado: Riesgo medio | Agresivo: Alto riesgo"
    )
    
    horizonte = st.selectbox(
        "⏰ ¿Cuál es tu horizonte de inversión?",
        ["Corto plazo (< 1 año)", "Mediano plazo (1-5 años)", "Largo plazo (> 5 años)"],
        index=1
    )
    
    capital = st.selectbox(
        "💰 ¿Cuál es tu capital aproximado de inversión?",
        ["< $10,000", "$10,000 - $50,000", "$50,000 - $100,000", "> $100,000"],
        index=1
    )

with col2:
    objetivos = st.multiselect(
        "🎯 ¿Cuáles son tus objetivos de inversión?",
        [
            "Crecimiento de capital",
            "Generación de ingresos (dividendos)",
            "Preservación de capital",
            "Diversificación internacional",
            "Inversión ESG/Sostenible"
        ],
        default=["Crecimiento de capital"]
    )
    
    sector_pref = st.selectbox(
        "🏢 ¿Preferencia sectorial?",
        ["Sin preferencia (diversificado)", "Tecnología", "Salud", "Finanzas", "Energía", "Consumo"],
        index=0
    )
    
    periodo_analisis = st.selectbox(
        "📊 Período de análisis histórico",
        ["6mo", "1y", "2y", "5y"],
        index=1,
        help="Período para calcular métricas históricas"
    )

st.markdown('</div>', unsafe_allow_html=True)

# Botón de generación
generate_btn = st.button("🚀 GENERAR PORTAFOLIO CON IA", type="primary", use_container_width=True)

# =========================
# GENERACIÓN Y ANÁLISIS
# =========================

if generate_btn:
    
    if not objetivos:
        st.error("❌ Selecciona al menos un objetivo de inversión")
        st.stop()
    
    # PASO 1: Generar portafolio con IA
    with st.spinner("🧠 IA generando portafolio personalizado..."):
        portafolio, justificacion = generate_portfolio_with_gemini(
            perfil, horizonte, capital, objetivos, sector_pref
        )
    
    if not portafolio:
        st.error("❌ No se pudo generar el portafolio. Intenta de nuevo.")
        st.stop()
    
    # Mostrar éxito
    st.markdown(f'<div class="success-badge">✅ Portafolio de {len(portafolio)} posiciones generado exitosamente</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # PASO 2: Obtener detalles de empresas y crear gráfico interactivo
    st.header("📊 Portafolio Sugerido por IA")
    
    st.markdown(f'<div class="info-card"><strong>💡 Estrategia:</strong> {justificacion}</div>', unsafe_allow_html=True)
    
    with st.spinner("📥 Obteniendo detalles de las empresas..."):
        # Obtener detalles de cada empresa
        portfolio_details = []
        for item in portafolio:
            details = get_company_details(item['ticker'])
            portfolio_details.append({
                **item,
                **details
            })
        time.sleep(1)
    
    # Crear gráfico de pastel interactivo con hover personalizado
    fig_pie = go.Figure()
    
    hover_text = []
    for item in portfolio_details:
        empleados_str = f"{item['empleados']:,}" if item['empleados'] > 0 else "N/D"
        hover_info = (
            f"<b>{item['nombre']}</b><br>"
            f"<b>Ticker:</b> {item['ticker']}<br>"
            f"<b>Sector:</b> {item['sector']}<br>"
            f"<b>Industria:</b> {item['industria']}<br>"
            f"<b>Empleados:</b> {empleados_str}<br>"
            f"<b>Ponderación:</b> {item['peso']:.1f}%<br><br>"
            f"<b>💡 Razón:</b><br>{item['razon']}"
        )
        hover_text.append(hover_info)
    
    colors = ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe', '#00f2fe', '#43e97b', '#38f9d7', '#fa709a', '#fee140']
    
    fig_pie.add_trace(go.Pie(
        labels=[f"{item['ticker']}<br>{item['peso']:.1f}%" for item in portfolio_details],
        values=[item['peso'] for item in portfolio_details],
        text=[item['ticker'] for item in portfolio_details],
        textposition='inside',
        textfont=dict(size=14, color='white', family='Inter', weight='bold'),
        hovertext=hover_text,
        hoverinfo='text',
        hole=0.4,
        marker=dict(
            colors=colors[:len(portfolio_details)],
            line=dict(color='white', width=3)
        ),
        pull=[0.05 if i == 0 else 0 for i in range(len(portfolio_details))]  # Destacar primera posición
    ))
    
    fig_pie.update_layout(
        title={
            'text': f'<b>Composición del Portafolio</b><br><sub>Hover para ver detalles de cada posición</sub>',
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=24, family='Inter', color='#333')
        },
        height=600,
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.05,
            font=dict(size=12, family='Inter')
        ),
        hoverlabel=dict(
            bgcolor="white",
            font_size=13,
            font_family="Inter",
            bordercolor="#667eea"
        )
    )
    
    st.plotly_chart(fig_pie, use_container_width=True)
    
    # PASO 3: Obtener datos reales
    tickers = [p['ticker'] for p in portafolio]
    weights = {p['ticker']: p['peso'] / 100 for p in portafolio}
    
    with st.spinner(f"📈 Descargando datos históricos de {len(tickers)} activos..."):
        batch_data = get_batch_stock_data(tickers, periodo_analisis)
        time.sleep(1)
    
    # Verificar datos
    successful_tickers = [t for t in tickers if t in batch_data and batch_data[t]['success']]
    failed_tickers = [t for t in tickers if t not in successful_tickers]
    
    if failed_tickers:
        st.markdown(f'<div class="warning-card">⚠️ <strong>Advertencia:</strong> No se pudieron obtener datos de: {", ".join(failed_tickers)}</div>', unsafe_allow_html=True)
    
    if not successful_tickers:
        st.error("❌ No se pudieron obtener datos de ningún activo")
        st.stop()
    
    # Ajustar pesos
    total_successful_weight = sum([weights[t] for t in successful_tickers])
    adjusted_weights = {t: weights[t] / total_successful_weight for t in successful_tickers}
    
    st.markdown(f'<div class="success-badge">✅ Datos obtenidos para {len(successful_tickers)} activos</div>', unsafe_allow_html=True)
    
    # PASO 4: Calcular métricas individuales
    st.markdown("---")
    st.header("📊 Análisis Individual de Activos")
    
    individual_data = []
    for ticker in successful_tickers:
        hist = batch_data[ticker]['history']
        metrics = calculate_metrics(hist)
        
        # Agregar indicadores visuales
        rend_icon = "🟢" if metrics.get('rendimiento_total', 0) > 0 else "🔴"
        sharpe_icon = "⭐" if metrics.get('sharpe_ratio', 0) > 1 else "⚠️" if metrics.get('sharpe_ratio', 0) > 0 else "❌"
        
        individual_data.append({
            'Ticker': ticker,
            'Peso': f"{adjusted_weights[ticker]*100:.1f}%",
            'Precio': f"${batch_data[ticker]['current_price']:.2f}",
            'Rendimiento': f"{rend_icon} {metrics.get('rendimiento_total', 0):.2f}%",
            'Volatilidad': f"{metrics.get('volatilidad_anual', 0):.1f}%",
            'Sharpe': f"{sharpe_icon} {metrics.get('sharpe_ratio', 0):.2f}",
            'Max DD': f"{metrics.get('max_drawdown', 0):.1f}%"
        })
    
    individual_df = pd.DataFrame(individual_data)
    
    st.markdown('<div class="enhanced-table">', unsafe_allow_html=True)
    st.dataframe(individual_df, use_container_width=True, hide_index=True, height=400)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Botón de descarga
    csv = individual_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "📥 Descargar Tabla (CSV)",
        csv,
        f"portafolio_ia_{datetime.now().strftime('%Y%m%d')}.csv",
        "text/csv",
        use_container_width=False
    )
    
    # PASO 5: Métricas del portafolio
    st.markdown("---")
    st.header("📈 Métricas del Portafolio Ponderado")
    
    portfolio_metrics = calculate_portfolio_metrics(batch_data, adjusted_weights, periodo_analisis)
    
    if portfolio_metrics:
        # Métricas visuales mejoradas
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            rend_total = portfolio_metrics.get('rendimiento_total', 0)
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">Rendimiento Total</div>
                <div class="metric-value" style="color: {'#4CAF50' if rend_total > 0 else '#f44336'}">{rend_total:+.2f}%</div>
                <div class="metric-sublabel">{periodo_analisis}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            rend_anual = portfolio_metrics.get('rendimiento_anual', 0)
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">Rendimiento Anual</div>
                <div class="metric-value" style="color: {'#4CAF50' if rend_anual > 0 else '#f44336'}">{rend_anual:+.2f}%</div>
                <div class="metric-sublabel">Anualizado</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            vol = portfolio_metrics.get('volatilidad_anual', 0)
            vol_color = '#4CAF50' if vol < 15 else '#FF9800' if vol < 25 else '#f44336'
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">Volatilidad</div>
                <div class="metric-value" style="color: {vol_color}">{vol:.1f}%</div>
                <div class="metric-sublabel">Anual</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            sharpe = portfolio_metrics.get('sharpe_ratio', 0)
            sharpe_color = '#4CAF50' if sharpe > 1.5 else '#FF9800' if sharpe > 0.5 else '#f44336'
            sharpe_icon = '⭐⭐⭐' if sharpe > 1.5 else '⭐⭐' if sharpe > 0.5 else '⭐'
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">Sharpe Ratio</div>
                <div class="metric-value" style="color: {sharpe_color}">{sharpe:.2f}</div>
                <div class="metric-sublabel">{sharpe_icon}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col5:
            dd = portfolio_metrics.get('max_drawdown', 0)
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">Max Drawdown</div>
                <div class="metric-value" style="color: #f44336">{dd:.1f}%</div>
                <div class="metric-sublabel">Pérdida Máxima</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Interpretación rápida
        if sharpe > 1.5:
            st.success("✅ **Excelente rendimiento ajustado por riesgo** - El portafolio está optimizado para el perfil seleccionado")
        elif sharpe > 0.5:
            st.info("ℹ️ **Rendimiento aceptable** - El portafolio muestra balance entre riesgo y retorno")
        else:
            st.warning("⚠️ **Considerar ajustes** - El rendimiento ajustado por riesgo podría mejorar")
        
        # PASO 6: Gráficos
        st.markdown("---")
        st.header("📊 Visualización del Portafolio")
        
        tab1, tab2, tab3 = st.tabs(["📈 Desempeño Histórico", "⚖️ Riesgo vs Retorno", "📊 Comparación Individual"])
        
        with tab1:
            # Gráfico de desempeño
            fig_perf = go.Figure()
            
            cumulative = portfolio_metrics.get('cumulative_returns')
            dates = portfolio_metrics.get('dates')
            
            if cumulative is not None and dates is not None:
                fig_perf.add_trace(go.Scatter(
                    x=dates,
                    y=(cumulative - 1) * 100,
                    name='Portafolio',
                    line=dict(color='#667eea', width=4),
                    fill='tonexty',
                    fillcolor='rgba(102, 126, 234, 0.2)',
                    hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Rendimiento: %{y:.2f}%<extra></extra>'
                ))
                
                fig_perf.add_hline(y=0, line_dash="dot", line_color="gray", line_width=2)
                
                fig_perf.update_layout(
                    title=dict(
                        text='<b>Evolución del Rendimiento Acumulado</b>',
                        font=dict(size=22, family='Inter', color='#333')
                    ),
                    xaxis_title="Fecha",
                    yaxis_title="Rendimiento Acumulado (%)",
                    hovermode='x unified',
                    height=500,
                    template='plotly_white',
                    font=dict(family='Inter')
                )
                st.plotly_chart(fig_perf, use_container_width=True)
        
        with tab2:
            # Scatter plot riesgo vs retorno
            scatter_data = []
            for ticker in successful_tickers:
                hist = batch_data[ticker]['history']
                metrics = calculate_metrics(hist)
                scatter_data.append({
                    'ticker': ticker,
                    'rend': metrics.get('rendimiento_total', 0),
                    'vol': metrics.get('volatilidad_anual', 0),
                    'peso': adjusted_weights[ticker] * 100,
                    'sharpe': metrics.get('sharpe_ratio', 0)
                })
            
            fig_scatter = go.Figure()
            
            for item in scatter_data:
                fig_scatter.add_trace(go.Scatter(
                    x=[item['vol']],
                    y=[item['rend']],
                    mode='markers+text',
                    name=item['ticker'],
                    text=item['ticker'],
                    textposition='top center',
                    marker=dict(
                        size=item['peso'] * 3,  # Tamaño proporcional al peso
                        color=item['sharpe'],
                        colorscale='RdYlGn',
                        showscale=True,
                        colorbar=dict(title="Sharpe<br>Ratio"),
                        line=dict(color='white', width=2)
                    ),
                    hovertemplate=(
                        f"<b>{item['ticker']}</b><br>"
                        f"Rendimiento: {item['rend']:.2f}%<br>"
                        f"Volatilidad: {item['vol']:.2f}%<br>"
                        f"Peso: {item['peso']:.1f}%<br>"
                        f"Sharpe: {item['sharpe']:.2f}<extra></extra>"
                    )
                ))
            
            # Agregar portafolio total
            fig_scatter.add_trace(go.Scatter(
                x=[portfolio_metrics.get('volatilidad_anual', 0)],
                y=[portfolio_metrics.get('rendimiento_total', 0)],
                mode='markers+text',
                name='PORTAFOLIO',
                text='PORTAFOLIO',
                textposition='top center',
                marker=dict(
                    size=40,
                    color='#667eea',
                    symbol='star',
                    line=dict(color='white', width=3)
                ),
                hovertemplate=(
                    f"<b>PORTAFOLIO COMPLETO</b><br>"
                    f"Rendimiento: {portfolio_metrics.get('rendimiento_total', 0):.2f}%<br>"
                    f"Volatilidad: {portfolio_metrics.get('volatilidad_anual', 0):.2f}%<br>"
                    f"Sharpe: {portfolio_metrics.get('sharpe_ratio', 0):.2f}<extra></extra>"
                )
            ))
            
            fig_scatter.update_layout(
                title=dict(
                    text='<b>Análisis Riesgo vs Retorno</b><br><sub>Tamaño = Ponderación | Color = Sharpe Ratio</sub>',
                    font=dict(size=22, family='Inter', color='#333')
                ),
                xaxis_title="Volatilidad Anual (%)",
                yaxis_title="Rendimiento Total (%)",
                showlegend=False,
                height=550,
                template='plotly_white',
                font=dict(family='Inter')
            )
            
            st.plotly_chart(fig_scatter, use_container_width=True)
            
            st.info("💡 **Interpretación:** Busca activos en la parte superior izquierda (alto retorno, baja volatilidad). El tamaño de cada círculo representa su peso en el portafolio.")
        
        with tab3:
            # Comparación individual vs portafolio
            fig_comp = go.Figure()
            
            for ticker in successful_tickers:
                hist = batch_data[ticker]['history']
                if not hist.empty and 'Close' in hist.columns:
                    normalized = (hist['Close'] / hist['Close'].iloc[0] - 1) * 100
                    fig_comp.add_trace(go.Scatter(
                        x=hist.index,
                        y=normalized,
                        name=f"{ticker} ({adjusted_weights[ticker]*100:.1f}%)",
                        mode='lines',
                        line=dict(width=2),
                        opacity=0.6
                    ))
            
            # Portafolio destacado
            if cumulative is not None and dates is not None:
                fig_comp.add_trace(go.Scatter(
                    x=dates,
                    y=(cumulative - 1) * 100,
                    name='PORTAFOLIO (ponderado)',
                    line=dict(color='#667eea', width=5),
                    mode='lines'
                ))
            
            fig_comp.update_layout(
                title=dict(
                    text='<b>Comparación: Activos Individuales vs Portafolio</b>',
                    font=dict(size=22, family='Inter', color='#333')
                ),
                xaxis_title="Fecha",
                yaxis_title="Rendimiento (%)",
                hovermode='x unified',
                height=550,
                template='plotly_white',
                font=dict(family='Inter'),
                legend=dict(
                    orientation="v",
                    yanchor="top",
                    y=1,
                    xanchor="left",
                    x=1.02
                )
            )
            
            st.plotly_chart(fig_comp, use_container_width=True)
        
        # PASO 7: Análisis final con IA (MÁXIMO 300 PALABRAS)
        st.markdown("---")
        st.markdown("""
        <div class="ai-section">
            <h2>🤖 Análisis Final con IA</h2>
            <p>Evaluación concisa del portafolio generado</p>
        </div>
        """, unsafe_allow_html=True)
        
        with st.spinner("🧠 Gemini analizando el portafolio..."):
            final_analysis = analyze_portfolio_with_gemini(
                individual_df,
                portfolio_metrics,
                perfil,
                justificacion
            )
            st.markdown(final_analysis)
        
        st.success("✅ Análisis completo generado exitosamente")
        
    else:
        st.error("❌ No se pudieron calcular métricas del portafolio")

else:
    # Pantalla inicial
    st.info("👆 **Completa el cuestionario** y presiona el botón para generar tu portafolio personalizado")
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🤖 IA Personalizada
        - **Gemini 2.5 Flash** analiza tu perfil
        - Sugiere portafolio óptimo
        - 5-10 posiciones balanceadas
        - Máxima diversificación
        """)
    
    with col2:
        st.markdown("""
        ### 📊 Análisis Completo
        - Métricas de riesgo/retorno
        - Ponderaciones optimizadas
        - Visualizaciones interactivas
        - Gráficos de desempeño
        """)
    
    with col3:
        st.markdown("""
        ### 💡 Recomendaciones
        - Evaluación de desempeño
        - Sugerencias de ajuste
        - Estrategia personalizada
        - Análisis conciso (300 palabras)
        """)

st.markdown("---")
st.caption("🤖 Powered by Gemini 2.5 Flash | Datos de Yahoo Finance | © 2025")