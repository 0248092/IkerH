from __future__ import annotations
import warnings
from datetime import datetime
from typing import Dict, List
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
import google.generativeai as genai
import os
import time

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
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
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
# FUNCIONES MEJORADAS - BATCH DOWNLOAD
# =========================

@st.cache_data(ttl=3600)
def get_batch_stock_data(tickers: List[str], period: str = "1y") -> Dict:
    """
    Descarga datos de múltiples acciones en batch (más eficiente y menos propenso a rate limiting)
    """
    try:
        # Descargar todos los datos a la vez
        tickers_str = " ".join(tickers)
        
        # Método 1: Batch download (más robusto)
        data = yf.download(
            tickers_str,
            period=period,
            group_by='ticker',
            auto_adjust=True,
            progress=False,
            threads=False  # Desactivar threading para evitar rate limiting
        )
        
        result = {}
        
        for ticker in tickers:
            try:
                # Obtener histórico para este ticker
                if len(tickers) == 1:
                    ticker_data = data
                else:
                    ticker_data = data[ticker] if ticker in data.columns.get_level_values(0) else pd.DataFrame()
                
                if not ticker_data.empty and 'Close' in ticker_data.columns:
                    result[ticker] = {
                        'history': ticker_data,
                        'current_price': float(ticker_data['Close'].iloc[-1]) if len(ticker_data) > 0 else 0,
                        'success': True
                    }
                else:
                    result[ticker] = {'success': False, 'history': pd.DataFrame()}
                    
            except Exception as e:
                result[ticker] = {'success': False, 'history': pd.DataFrame()}
        
        return result
        
    except Exception as e:
        st.error(f"Error en batch download: {str(e)}")
        return {}

@st.cache_data(ttl=3600)
def get_stock_info_alternative(ticker: str, hist_data: pd.DataFrame) -> Dict:
    """
    Método alternativo usando solo datos históricos sin llamar a .info
    """
    if hist_data.empty:
        return {}
    
    try:
        # Calcular métricas básicas desde el histórico
        current_price = float(hist_data['Close'].iloc[-1])
        returns = hist_data['Close'].pct_change().dropna()
        
        # Estimaciones básicas
        data = {
            'nombre': ticker,
            'sector': 'N/D',
            'industria': 'N/D',
            'precio_actual': current_price,
            'market_cap': 0,
            'pe_ratio': 0,
            'dividend_yield': 0,
            'beta': returns.std() * np.sqrt(252) / 0.16,  # Estimación vs mercado
            'eps': 0,
            'precio_objetivo': 0
        }
        
        return data
        
    except Exception:
        return {}

@st.cache_data(ttl=3600)
def scrape_finviz_safe(ticker: str) -> Dict:
    """Scraping de Finviz con mejor manejo"""
    try:
        time.sleep(1)
        url = f"https://finviz.com/quote.ashx?t={ticker}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
        }
        response = requests.get(url, headers=headers, timeout=15)
        
        if response.status_code != 200:
            return {}
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extraer datos clave
        data = {}
        table = soup.find('table', {'class': 'snapshot-table2'})
        
        if table:
            rows = table.find_all('tr')
            for row in rows:
                cols = row.find_all('td')
                for i in range(0, len(cols), 2):
                    if i+1 < len(cols):
                        key = cols[i].text.strip()
                        value = cols[i+1].text.strip()
                        
                        # Extraer métricas importantes
                        if key == 'P/E':
                            data['pe_ratio'] = value
                        elif key == 'Market Cap':
                            data['market_cap'] = value
                        elif key == 'Dividend %':
                            data['dividend_yield'] = value
                        elif key == 'Sector':
                            data['sector'] = value
                        elif key == 'Industry':
                            data['industry'] = value
        
        return data
    except Exception:
        return {}

# =========================
# FUNCIONES DE ANÁLISIS
# =========================

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
        'sharpe_ratio': (returns.mean() * 252) / (returns.std() * np.sqrt(252)) if returns.std() != 0 else 0
    }
    
    return metrics

def compare_stocks_batch(tickers: List[str], period: str = "1y") -> pd.DataFrame:
    """Compara múltiples acciones usando batch download"""
    
    # Descargar todo en batch
    batch_data = get_batch_stock_data(tickers, period)
    
    comparison_data = []
    
    for ticker in tickers:
        if ticker not in batch_data or not batch_data[ticker]['success']:
            st.warning(f"⚠️ No se pudieron obtener datos de {ticker}")
            continue
        
        hist = batch_data[ticker]['history']
        
        # Usar método alternativo (sin .info)
        stock_info = get_stock_info_alternative(ticker, hist)
        
        # Intentar complementar con Finviz
        finviz_data = scrape_finviz_safe(ticker)
        if finviz_data:
            stock_info['sector'] = finviz_data.get('sector', stock_info['sector'])
            stock_info['industria'] = finviz_data.get('industry', stock_info['industria'])
        
        # Calcular métricas
        metrics = calculate_metrics(hist)
        
        if stock_info:
            comparison_data.append({
                'Ticker': ticker,
                'Nombre': stock_info.get('nombre', ticker),
                'Sector': stock_info.get('sector', 'N/D'),
                'Precio': f"${stock_info.get('precio_actual', 0):.2f}",
                'Market Cap': finviz_data.get('market_cap', 'N/D'),
                'P/E': finviz_data.get('pe_ratio', 'N/D'),
                'Beta': f"{stock_info.get('beta', 0):.2f}",
                'Div. Yield': finviz_data.get('dividend_yield', '0.00%'),
                'Rendimiento 1Y': f"{metrics.get('rendimiento_total', 0):.2f}%",
                'Volatilidad': f"{metrics.get('volatilidad_anual', 0):.2f}%",
                'Sharpe': f"{metrics.get('sharpe_ratio', 0):.2f}"
            })
    
    return pd.DataFrame(comparison_data)

# =========================
# FUNCIÓN DE IA
# =========================

def analyze_with_gemini(portfolio_data: pd.DataFrame, profile: str) -> str:
    """Análisis con Gemini"""
    perfiles = {
        "Conservador": "bajo riesgo, preservación de capital",
        "Moderado": "riesgo medio, balance",
        "Agresivo": "alto riesgo, crecimiento"
    }
    
    prompt = f"""
    Eres un asesor financiero. Analiza este portafolio para un perfil {profile}.
    
    Datos:
    {portfolio_data.to_string()}
    
    Proporciona:
    1. Evaluación general
    2. Recomendaciones por acción
    3. Estrategia para perfil {profile}
    4. Conclusión

    Haz el analisis financiero con máximo 300 palabras inlcuyendo bullet points de un analisis con ponderaciones del portafolio por acción y alternativas de inversión contra sus peers, todo con finalidad de maximizar rendimientos con el menor riesgo posible.
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"

# =========================
# INTERFAZ
# =========================

st.markdown("""
<div class="main-header">
    <h1>📊 Análisis Comparativo de Portafolio</h1>
    <p>Análisis avanzado de acciones con IA</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Configuración")
    
    st.subheader("Selección de Acciones")
    ticker_input = st.text_area(
        "Ingresa los tickers (uno por línea)",
        value="AAPL\nMSFT\nGOOGL\nAMZN\nTSLA",
        height=150,
        help="Máximo 10 acciones"
    )
    
    tickers = [t.strip().upper() for t in ticker_input.split('\n') if t.strip()]
    
    if len(tickers) > 10:
        st.warning("⚠️ Limitado a 10 acciones")
        tickers = tickers[:10]
    
    st.metric("Acciones", len(tickers))
    
    st.markdown("---")
    
    perfil = st.selectbox(
        "🎯 Perfil de Inversión",
        ["Conservador", "Moderado", "Agresivo"],
        index=1
    )
    
    st.markdown("---")
    
    periodo = st.selectbox(
        "📅 Período",
        ["1mo", "3mo", "6mo", "1y", "2y"],
        index=3
    )
    
    st.markdown("---")
    
    analyze_btn = st.button("🚀 ANALIZAR", type="primary", use_container_width=True)

if analyze_btn:
    if not tickers:
        st.error("❌ Ingresa al menos un ticker")
        st.stop()
    
    with st.spinner(f"📊 Analizando {len(tickers)} acciones..."):
        
        try:
            # Obtener datos en batch
            st.info("⏳ Descargando datos en batch...")
            comparison_df = compare_stocks_batch(tickers, periodo)
            
            if not comparison_df.empty:
                st.success(f"✅ Datos obtenidos para {len(comparison_df)} acciones")
                
                st.header("📊 Comparación de Acciones")
                st.dataframe(comparison_df, use_container_width=True, hide_index=True)
                
                csv = comparison_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Descargar CSV",
                    csv,
                    f"portafolio_{datetime.now().strftime('%Y%m%d')}.csv",
                    "text/csv"
                )
                
                st.markdown("---")
                
                # Gráficos
                st.header("📈 Análisis Visual")
                
                batch_data = get_batch_stock_data(tickers, periodo)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_prices = go.Figure()
                    
                    for ticker in tickers:
                        if ticker in batch_data and batch_data[ticker]['success']:
                            hist = batch_data[ticker]['history']
                            if not hist.empty and 'Close' in hist.columns:
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
                        yaxis_title="Valor",
                        height=400
                    )
                    st.plotly_chart(fig_prices, use_container_width=True)
                
                with col2:
                    volatilidades = []
                    for ticker in tickers:
                        if ticker in batch_data and batch_data[ticker]['success']:
                            hist = batch_data[ticker]['history']
                            metrics = calculate_metrics(hist)
                            volatilidades.append({
                                'Ticker': ticker,
                                'Volatilidad': metrics.get('volatilidad_anual', 0)
                            })
                    
                    if volatilidades:
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
                            title="Volatilidad Anual",
                            xaxis_title="Ticker",
                            yaxis_title="Volatilidad (%)",
                            height=400
                        )
                        st.plotly_chart(fig_vol, use_container_width=True)
                
                st.markdown("---")
                
                # IA
                st.markdown("""
                <div class="ai-section">
                    <h2>🤖 Análisis con IA</h2>
                </div>
                """, unsafe_allow_html=True)
                
                with st.spinner("🧠 Generando análisis..."):
                    ai_analysis = analyze_with_gemini(comparison_df, perfil)
                    st.markdown(ai_analysis)
                
                st.success("✅ Análisis completado")
            
            else:
                st.error("❌ No se pudieron obtener datos. Intenta con otros tickers.")
        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.info("💡 Intenta reducir el número de acciones o espera unos minutos antes de reintentar.")

else:
    st.info("👈 Configura tu portafolio y presiona ANALIZAR")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 📊 Análisis\n- Yahoo Finance\n- Finviz\n- Métricas avanzadas")
    
    with col2:
        st.markdown("### 📈 Visualización\n- Gráficos interactivos\n- Volatilidad\n- Desempeño")
    
    with col3:
        st.markdown("### 🤖 IA\n- Gemini AI\n- Por perfil\n- Recomendaciones")

st.markdown("---")
st.caption("📊 Powered by Yahoo Finance, Finviz & Gemini AI")