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
import re
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
model = genai.GenerativeModel("gemini-2.5-flash")

# =========================
# ESTILOS CSS
# =========================
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    }
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }
    .questionnaire-card {
        background: white;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
        border-left: 5px solid #667eea;
    }
    .portfolio-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 2rem;
        border-radius: 12px;
        margin: 1.5rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    .ai-section {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 2rem;
        border-radius: 12px;
        color: white;
        margin-top: 2rem;
        box-shadow: 0 8px 20px rgba(245, 87, 108, 0.3);
    }
    .metric-box {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border-top: 4px solid #667eea;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #667eea;
    }
    .metric-label {
        color: #666;
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# =========================
# FUNCIONES DE DATOS
# =========================

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
        st.error(f"Error descargando datos: {str(e)}")
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
    portfolio_value = []
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
    
    # Calcular valor acumulado del portafolio
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
    - Solo tickers válidos de NYSE/NASDAQ/BMV
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
        
        # Validar que los pesos sumen 100
        total_peso = sum([p['peso'] for p in portafolio])
        if abs(total_peso - 100) > 1:  # Tolerancia de 1%
            # Normalizar
            for p in portafolio:
                p['peso'] = (p['peso'] / total_peso) * 100
        
        return portafolio, justificacion
        
    except Exception as e:
        st.error(f"Error generando portafolio: {str(e)}")
        return [], ""

def analyze_portfolio_with_gemini(portfolio_data: pd.DataFrame, metrics: Dict, perfil: str, justificacion_inicial: str) -> str:
    """Análisis final del portafolio con Gemini"""
    
    prompt = f"""
    Como asesor financiero experto, realiza un análisis detallado del portafolio que sugeriste.
    
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
    
    Proporciona un análisis completo que incluya:
    
    1. **Evaluación del Desempeño**
       - ¿El portafolio está cumpliendo con el perfil de riesgo?
       - Análisis de rendimiento vs volatilidad
       - Evaluación del Sharpe Ratio
    
    2. **Análisis por Posición**
       - ¿Qué acciones están generando mejor/peor desempeño?
       - ¿Las ponderaciones son adecuadas?
    
    3. **Gestión de Riesgo**
       - Evaluación del Max Drawdown
       - Diversificación sectorial
       - Recomendaciones de ajuste
    
    4. **Recomendaciones Específicas**
       - ¿Mantener, aumentar o reducir posiciones?
       - ¿Agregar nuevas posiciones?
       - ¿Rebalancear el portafolio?
    
    5. **Conclusión y Próximos Pasos**
       - Acción inmediata recomendada
       - Monitoreo sugerido
    
    Sé específico, técnico pero claro. Por ningpun motivo te excedas de 300 palabras.
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
    <p>Portafolios personalizados generados por Inteligencia Artificial</p>
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
    
    # Mostrar portafolio sugerido
    st.success(f"✅ Portafolio de {len(portafolio)} posiciones generado")
    
    st.markdown('<div class="portfolio-card">', unsafe_allow_html=True)
    st.header("📊 Portafolio Sugerido por IA")
    
    st.markdown(f"**Justificación:** {justificacion}")
    
    # Tabla del portafolio
    portfolio_df = pd.DataFrame(portafolio)
    portfolio_df['peso'] = portfolio_df['peso'].apply(lambda x: f"{x:.1f}%")
    
    st.dataframe(
        portfolio_df[['ticker', 'peso', 'razon']].rename(columns={
            'ticker': 'Ticker',
            'peso': 'Ponderación',
            'razon': 'Razón'
        }),
        use_container_width=True,
        hide_index=True
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    # PASO 2: Obtener datos reales
    tickers = [p['ticker'] for p in portafolio]
    weights = {p['ticker']: p['peso'] / 100 for p in portafolio}
    
    with st.spinner(f"📈 Descargando datos de {len(tickers)} activos..."):
        batch_data = get_batch_stock_data(tickers, periodo_analisis)
        time.sleep(1)
    
    # Verificar qué datos se obtuvieron
    successful_tickers = [t for t in tickers if t in batch_data and batch_data[t]['success']]
    failed_tickers = [t for t in tickers if t not in successful_tickers]
    
    if failed_tickers:
        st.warning(f"⚠️ No se pudieron obtener datos de: {', '.join(failed_tickers)}")
    
    if not successful_tickers:
        st.error("❌ No se pudieron obtener datos de ningún activo")
        st.stop()
    
    # Ajustar pesos solo para tickers exitosos
    total_successful_weight = sum([weights[t] for t in successful_tickers])
    adjusted_weights = {t: weights[t] / total_successful_weight for t in successful_tickers}
    
    st.success(f"✅ Datos obtenidos para {len(successful_tickers)} activos")
    
    # PASO 3: Calcular métricas individuales
    st.markdown("---")
    st.header("📊 Análisis Individual de Activos")
    
    individual_data = []
    for ticker in successful_tickers:
        hist = batch_data[ticker]['history']
        metrics = calculate_metrics(hist)
        
        individual_data.append({
            'Ticker': ticker,
            'Peso': f"{adjusted_weights[ticker]*100:.1f}%",
            'Precio Actual': f"${batch_data[ticker]['current_price']:.2f}",
            'Rendimiento': f"{metrics.get('rendimiento_total', 0):.2f}%",
            'Volatilidad': f"{metrics.get('volatilidad_anual', 0):.2f}%",
            'Sharpe': f"{metrics.get('sharpe_ratio', 0):.2f}",
            'Max DD': f"{metrics.get('max_drawdown', 0):.2f}%"
        })
    
    individual_df = pd.DataFrame(individual_data)
    st.dataframe(individual_df, use_container_width=True, hide_index=True)
    
    # Descargar tabla
    csv = individual_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "📥 Descargar Datos",
        csv,
        f"portafolio_ia_{datetime.now().strftime('%Y%m%d')}.csv",
        "text/csv"
    )
    
    # PASO 4: Calcular métricas del portafolio ponderado
    st.markdown("---")
    st.header("📊 Métricas del Portafolio Ponderado")
    
    portfolio_metrics = calculate_portfolio_metrics(batch_data, adjusted_weights, periodo_analisis)
    
    if portfolio_metrics:
        # Métricas en cards
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-value">{portfolio_metrics.get('rendimiento_total', 0):.2f}%</div>
                <div class="metric-label">Rendimiento Total</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-value">{portfolio_metrics.get('rendimiento_anual', 0):.2f}%</div>
                <div class="metric-label">Rendimiento Anual</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-value">{portfolio_metrics.get('volatilidad_anual', 0):.2f}%</div>
                <div class="metric-label">Volatilidad Anual</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            sharpe_color = "#4CAF50" if portfolio_metrics.get('sharpe_ratio', 0) > 1 else "#FF9800"
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-value" style="color: {sharpe_color}">{portfolio_metrics.get('sharpe_ratio', 0):.2f}</div>
                <div class="metric-label">Sharpe Ratio</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col5:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-value" style="color: #f44336">{portfolio_metrics.get('max_drawdown', 0):.2f}%</div>
                <div class="metric-label">Max Drawdown</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # PASO 5: Gráficos
        st.header("📈 Visualización del Portafolio")
        
        tab1, tab2, tab3 = st.tabs(["📈 Desempeño", "🥧 Composición", "📊 Comparación"])
        
        with tab1:
            # Gráfico de valor acumulado del portafolio
            fig_portfolio = go.Figure()
            
            cumulative = portfolio_metrics.get('cumulative_returns')
            dates = portfolio_metrics.get('dates')
            
            if cumulative is not None and dates is not None:
                fig_portfolio.add_trace(go.Scatter(
                    x=dates,
                    y=(cumulative - 1) * 100,
                    name='Portafolio',
                    line=dict(color='#667eea', width=3),
                    fill='tonexty',
                    fillcolor='rgba(102, 126, 234, 0.1)'
                ))
                
                fig_portfolio.add_hline(y=0, line_dash="dash", line_color="gray")
                
                fig_portfolio.update_layout(
                    title="Rendimiento Acumulado del Portafolio",
                    xaxis_title="Fecha",
                    yaxis_title="Rendimiento (%)",
                    hovermode='x unified',
                    height=500
                )
                st.plotly_chart(fig_portfolio, use_container_width=True)
        
        with tab2:
            # Gráfico de pie
            fig_pie = go.Figure(data=[go.Pie(
                labels=list(adjusted_weights.keys()),
                values=[w*100 for w in adjusted_weights.values()],
                hole=0.4,
                marker=dict(colors=['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe', '#00f2fe', '#43e97b', '#38f9d7'])
            )])
            
            fig_pie.update_layout(
                title="Composición del Portafolio",
                height=500
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with tab3:
            # Comparación de activos individuales
            fig_comparison = go.Figure()
            
            for ticker in successful_tickers:
                hist = batch_data[ticker]['history']
                if not hist.empty and 'Close' in hist.columns:
                    normalized = (hist['Close'] / hist['Close'].iloc[0] - 1) * 100
                    fig_comparison.add_trace(go.Scatter(
                        x=hist.index,
                        y=normalized,
                        name=f"{ticker} ({adjusted_weights[ticker]*100:.1f}%)",
                        mode='lines'
                    ))
            
            # Agregar portafolio
            if cumulative is not None and dates is not None:
                fig_comparison.add_trace(go.Scatter(
                    x=dates,
                    y=(cumulative - 1) * 100,
                    name='Portafolio (ponderado)',
                    line=dict(color='black', width=3, dash='dash')
                ))
            
            fig_comparison.update_layout(
                title="Comparación: Activos Individuales vs Portafolio",
                xaxis_title="Fecha",
                yaxis_title="Rendimiento (%)",
                hovermode='x unified',
                height=500
            )
            st.plotly_chart(fig_comparison, use_container_width=True)
        
        # PASO 6: Análisis final con IA
        st.markdown("---")
        st.markdown("""
        <div class="ai-section">
            <h2>🤖 Análisis Final del Portafolio</h2>
            <p>La IA analiza el desempeño real del portafolio sugerido</p>
        </div>
        """, unsafe_allow_html=True)
        
        with st.spinner("🧠 Generando análisis detallado..."):
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
    st.info("👆 Completa el cuestionario y presiona **GENERAR PORTAFOLIO CON IA**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🤖 IA Personalizada
        - Gemini analiza tu perfil
        - Sugiere portafolio óptimo
        - 5-10 posiciones balanceadas
        """)
    
    with col2:
        st.markdown("""
        ### 📊 Análisis Completo
        - Métricas de riesgo/retorno
        - Ponderaciones optimizadas
        - Visualizaciones interactivas
        """)
    
    with col3:
        st.markdown("""
        ### 💡 Recomendaciones
        - Evaluación de desempeño
        - Sugerencias de ajuste
        - Estrategia personalizada
        """)

st.markdown("---")
st.caption("🤖 Powered by Gemini AI | Datos de Yahoo Finance")