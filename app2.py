from __future__ import annotations
import warnings
from datetime import datetime, timedelta
from typing import Dict, List
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import google.generativeai as genai

warnings.filterwarnings("ignore")

# =========================
# CONFIGURACIÓN
# =========================
st.set_page_config(
    page_title="Análisis de Portafolio de Acciones",
    layout="wide",
    page_icon="📊"
)

# Configurar Gemini
API_KEY = st.env["API_KEY"]
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash-exp")

# =========================
# ESTILOS CSS
# =========================
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    .stock-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        border: 2px solid #e9ecef;
    }
    .ai-section {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# =========================
# FUNCIONES DE SCRAPING
# =========================

@st.cache_data(ttl=3600)
def scrape_yfinance_data(ticker: str) -> Dict:
    """Obtiene datos de Yahoo Finance usando web scraping"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Datos básicos
        data = {
            'nombre': info.get('longName', ticker),
            'sector': info.get('sector', 'N/D'),
            'industria': info.get('industry', 'N/D'),
            'precio_actual': info.get('currentPrice', 0),
            'market_cap': info.get('marketCap', 0),
            'pe_ratio': info.get('trailingPE', 0),
            'dividend_yield': info.get('dividendYield', 0),
            'beta': info.get('beta', 0),
            'eps': info.get('trailingEps', 0),
            'precio_objetivo': info.get('targetMeanPrice', 0)
        }
        
        return data
    except Exception as e:
        st.warning(f"Error obteniendo datos de {ticker}: {str(e)}")
        return {}

@st.cache_data(ttl=3600)
def scrape_finviz_data(ticker: str) -> Dict:
    """Obtiene datos de Finviz usando web scraping"""
    try:
        url = f"https://finviz.com/quote.ashx?t={ticker}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code != 200:
            return {}
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extraer tabla de datos
        table = soup.find('table', {'class': 'snapshot-table2'})
        data = {}
        
        if table:
            rows = table.find_all('tr')
            for row in rows:
                cols = row.find_all('td')
                for i in range(0, len(cols), 2):
                    if i+1 < len(cols):
                        key = cols[i].text.strip()
                        value = cols[i+1].text.strip()
                        data[key] = value
        
        return data
    except Exception as e:
        st.warning(f"Error scraping Finviz para {ticker}: {str(e)}")
        return {}

@st.cache_data(ttl=1800)
def get_historical_prices(ticker: str, period: str = "1y") -> pd.DataFrame:
    """Obtiene precios históricos"""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        return hist
    except Exception as e:
        st.error(f"Error obteniendo histórico de {ticker}: {str(e)}")
        return pd.DataFrame()

# =========================
# FUNCIONES DE ANÁLISIS
# =========================

def calculate_metrics(prices: pd.DataFrame) -> Dict:
    """Calcula métricas financieras del histórico"""
    if prices.empty:
        return {}
    
    returns = prices['Close'].pct_change().dropna()
    
    metrics = {
        'rendimiento_total': ((prices['Close'].iloc[-1] / prices['Close'].iloc[0]) - 1) * 100,
        'volatilidad_anual': returns.std() * np.sqrt(252) * 100,
        'rendimiento_anual': returns.mean() * 252 * 100,
        'max_drawdown': ((prices['Close'] / prices['Close'].cummax()) - 1).min() * 100,
        'sharpe_ratio': (returns.mean() * 252) / (returns.std() * np.sqrt(252)) if returns.std() != 0 else 0
    }
    
    return metrics

def compare_stocks(tickers: List[str]) -> pd.DataFrame:
    """Compara múltiples acciones"""
    comparison_data = []
    
    for ticker in tickers:
        yf_data = scrape_yfinance_data(ticker)
        finviz_data = scrape_finviz_data(ticker)
        hist = get_historical_prices(ticker, "1y")
        metrics = calculate_metrics(hist)
        
        if yf_data:
            comparison_data.append({
                'Ticker': ticker,
                'Nombre': yf_data.get('nombre', 'N/D'),
                'Sector': yf_data.get('sector', 'N/D'),
                'Precio': f"${yf_data.get('precio_actual', 0):.2f}",
                'Market Cap': f"${yf_data.get('market_cap', 0)/1e9:.2f}B",
                'P/E': f"{yf_data.get('pe_ratio', 0):.2f}",
                'Beta': f"{yf_data.get('beta', 0):.2f}",
                'Div. Yield': f"{yf_data.get('dividend_yield', 0)*100:.2f}%",
                'Rendimiento 1Y': f"{metrics.get('rendimiento_total', 0):.2f}%",
                'Volatilidad': f"{metrics.get('volatilidad_anual', 0):.2f}%",
                'Sharpe': f"{metrics.get('sharpe_ratio', 0):.2f}"
            })
    
    return pd.DataFrame(comparison_data)

# =========================
# FUNCIÓN DE IA
# =========================

def analyze_with_gemini(portfolio_data: pd.DataFrame, profile: str) -> str:
    """Análisis con Gemini según perfil de inversión"""
    
    perfiles = {
        "Conservador": "bajo riesgo, enfocado en preservación de capital y dividendos estables",
        "Moderado": "riesgo medio, balance entre crecimiento y estabilidad",
        "Agresivo": "alto riesgo, enfocado en máximo crecimiento a largo plazo"
    }
    
    perfil_descripcion = perfiles.get(profile, perfiles["Moderado"])
    
    prompt = f"""
    Eres un asesor financiero experto. Analiza el siguiente portafolio de acciones para un inversionista con perfil {profile} ({perfil_descripcion}).
    
    Datos del portafolio:
    {portfolio_data.to_string()}
    
    Proporciona un análisis detallado que incluya:
    
    1. **Evaluación General del Portafolio**
       - Diversificación sectorial
       - Balance riesgo-retorno
       - Valoración general (P/E ratios)
    
    2. **Recomendaciones Específicas por Acción**
       - Qué acciones mantener, comprar más o vender
       - Justificación basada en métricas
    
    3. **Estrategia para el Perfil {profile}**
       - Ajustes recomendados
       - Ponderación sugerida del portafolio
       - Riesgos a considerar
    
    4. **Conclusión y Acción Inmediata**
       - Siguiente paso concreto
    
    Sé específico, técnico pero claro. Usa las métricas del portafolio.
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generando análisis: {str(e)}"

# =========================
# INTERFAZ PRINCIPAL
# =========================

# Header
st.markdown("""
<div class="main-header">
    <h1>📊 Análisis Comparativo de Portafolio</h1>
    <p>Análisis avanzado de acciones con datos de Yahoo Finance, Finviz e IA</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuración")
    
    # Input de tickers
    st.subheader("Selección de Acciones")
    ticker_input = st.text_area(
        "Ingresa los tickers (uno por línea)",
        value="AAPL\nMSFT\nGOOGL\nAMZN",
        height=150,
        help="Ingresa un ticker por línea. Ejemplo: AAPL, MSFT, GOOGL"
    )
    
    # Convertir input a lista
    tickers = [t.strip().upper() for t in ticker_input.split('\n') if t.strip()]
    
    st.metric("Acciones Seleccionadas", len(tickers))
    
    st.markdown("---")
    
    # Perfil de inversión
    st.subheader("🎯 Perfil de Inversión")
    perfil = st.selectbox(
        "Selecciona tu perfil",
        ["Conservador", "Moderado", "Agresivo"],
        index=1
    )
    
    st.info(f"**Perfil {perfil}**: " + 
            ("Bajo riesgo, dividendos" if perfil == "Conservador" else
             "Riesgo medio, balanceado" if perfil == "Moderado" else
             "Alto riesgo, crecimiento"))
    
    st.markdown("---")
    
    # Período de análisis
    st.subheader("📅 Período")
    periodo = st.selectbox(
        "Período histórico",
        ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
        index=3
    )
    
    st.markdown("---")
    
    # Botón de análisis
    analyze_btn = st.button("🚀 ANALIZAR PORTAFOLIO", type="primary", use_container_width=True)

# =========================
# ANÁLISIS PRINCIPAL
# =========================

if analyze_btn:
    if not tickers:
        st.error("❌ Por favor ingresa al menos un ticker")
        st.stop()
    
    with st.spinner("🔄 Analizando portafolio..."):
        
        # Comparación de acciones
        st.header("📊 Comparación de Acciones")
        comparison_df = compare_stocks(tickers)
        
        if not comparison_df.empty:
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)
            
            # Descargar datos
            csv = comparison_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Descargar Comparación (CSV)",
                csv,
                f"comparacion_portafolio_{datetime.now().strftime('%Y%m%d')}.csv",
                "text/csv"
            )
        
        st.markdown("---")
        
        # Gráficos comparativos
        st.header("📈 Desempeño Histórico")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Gráfico de precios normalizados
            fig_prices = go.Figure()
            
            for ticker in tickers:
                hist = get_historical_prices(ticker, periodo)
                if not hist.empty:
                    # Normalizar a base 100
                    normalized = (hist['Close'] / hist['Close'].iloc[0]) * 100
                    fig_prices.add_trace(go.Scatter(
                        x=hist.index,
                        y=normalized,
                        name=ticker,
                        mode='lines'
                    ))
            
            fig_prices.update_layout(
                title="Precios Normalizados (Base 100)",
                xaxis_title="Fecha",
                yaxis_title="Valor (Base 100)",
                hovermode='x unified',
                height=400
            )
            st.plotly_chart(fig_prices, use_container_width=True)
        
        with col2:
            # Gráfico de volatilidad
            volatilidades = []
            for ticker in tickers:
                hist = get_historical_prices(ticker, periodo)
                metrics = calculate_metrics(hist)
                volatilidades.append({
                    'Ticker': ticker,
                    'Volatilidad': metrics.get('volatilidad_anual', 0)
                })
            
            vol_df = pd.DataFrame(volatilidades)
            
            fig_vol = go.Figure(data=[
                go.Bar(
                    x=vol_df['Ticker'],
                    y=vol_df['Volatilidad'],
                    marker_color='lightblue',
                    text=vol_df['Volatilidad'].apply(lambda x: f'{x:.1f}%'),
                    textposition='outside'
                )
            ])
            
            fig_vol.update_layout(
                title="Volatilidad Anual por Acción",
                xaxis_title="Ticker",
                yaxis_title="Volatilidad (%)",
                height=400
            )
            st.plotly_chart(fig_vol, use_container_width=True)
        
        st.markdown("---")
        
        # Análisis detallado por acción
        st.header("🔍 Análisis Detallado por Acción")
        
        tabs = st.tabs(tickers)
        
        for i, ticker in enumerate(tickers):
            with tabs[i]:
                col_a, col_b = st.columns([1, 2])
                
                with col_a:
                    st.subheader(f"{ticker}")
                    yf_data = scrape_yfinance_data(ticker)
                    finviz_data = scrape_finviz_data(ticker)
                    
                    if yf_data:
                        st.markdown(f"""
                        <div class="stock-card">
                            <h4>{yf_data.get('nombre', 'N/D')}</h4>
                            <p><strong>Sector:</strong> {yf_data.get('sector', 'N/D')}</p>
                            <p><strong>Industria:</strong> {yf_data.get('industria', 'N/D')}</p>
                            <p><strong>Precio:</strong> ${yf_data.get('precio_actual', 0):.2f}</p>
                            <p><strong>P/E Ratio:</strong> {yf_data.get('pe_ratio', 0):.2f}</p>
                            <p><strong>Beta:</strong> {yf_data.get('beta', 0):.2f}</p>
                            <p><strong>Div. Yield:</strong> {yf_data.get('dividend_yield', 0)*100:.2f}%</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    if finviz_data:
                        st.subheader("Datos Finviz")
                        st.json(finviz_data)
                
                with col_b:
                    # Gráfico de velas
                    hist = get_historical_prices(ticker, periodo)
                    if not hist.empty:
                        fig_candle = go.Figure(data=[go.Candlestick(
                            x=hist.index,
                            open=hist['Open'],
                            high=hist['High'],
                            low=hist['Low'],
                            close=hist['Close']
                        )])
                        
                        fig_candle.update_layout(
                            title=f"{ticker} - Gráfico de Velas",
                            xaxis_title="Fecha",
                            yaxis_title="Precio ($)",
                            height=400,
                            xaxis_rangeslider_visible=False
                        )
                        st.plotly_chart(fig_candle, use_container_width=True)
                        
                        # Métricas
                        metrics = calculate_metrics(hist)
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Rendimiento Total", f"{metrics.get('rendimiento_total', 0):.2f}%")
                        m2.metric("Volatilidad", f"{metrics.get('volatilidad_anual', 0):.2f}%")
                        m3.metric("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.2f}")
                        m4.metric("Max Drawdown", f"{metrics.get('max_drawdown', 0):.2f}%")
        
        st.markdown("---")
        
        # Análisis con IA
        st.markdown("""
        <div class="ai-section">
            <h2>🤖 Análisis con Inteligencia Artificial</h2>
            <p>Análisis personalizado según tu perfil de inversión</p>
        </div>
        """, unsafe_allow_html=True)
        
        with st.spinner("🧠 Generando análisis con Gemini..."):
            ai_analysis = analyze_with_gemini(comparison_df, perfil)
            st.markdown(ai_analysis)
        
        st.success("✅ Análisis completado")

else:
    # Pantalla de inicio
    st.info("👈 Configura tu portafolio en la barra lateral y presiona **ANALIZAR PORTAFOLIO**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 📊 Análisis Completo
        - Datos de Yahoo Finance
        - Scraping de Finviz
        - Métricas financieras avanzadas
        """)
    
    with col2:
        st.markdown("""
        ### 📈 Comparación Visual
        - Gráficos interactivos
        - Análisis de volatilidad
        - Desempeño histórico
        """)
    
    with col3:
        st.markdown("""
        ### 🤖 IA Personalizada
        - Análisis con Gemini
        - Recomendaciones por perfil
        - Estrategias de inversión
        """)

# Footer
st.markdown("---")
st.caption("📊 Análisis de Portafolio | Datos de Yahoo Finance & Finviz | Powered by Gemini AI")