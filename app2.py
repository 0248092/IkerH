from __future__ import annotations
import warnings
from datetime import datetime
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
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
model = genai.GenerativeModel("gemini-2.5-flash")

# Inicializar session state
if 'portfolio_generated' not in st.session_state:
    st.session_state.portfolio_generated = False
if 'selected_stock_ticker' not in st.session_state:
    st.session_state.selected_stock_ticker = None
if 'portfolio_details' not in st.session_state:
    st.session_state.portfolio_details = None
if 'batch_data' not in st.session_state:
    st.session_state.batch_data = None
if 'portafolio' not in st.session_state:
    st.session_state.portafolio = None
if 'justificacion' not in st.session_state:
    st.session_state.justificacion = None
if 'successful_tickers' not in st.session_state:
    st.session_state.successful_tickers = None
if 'adjusted_weights' not in st.session_state:
    st.session_state.adjusted_weights = None
if 'portfolio_metrics' not in st.session_state:
    st.session_state.portfolio_metrics = None
if 'individual_df' not in st.session_state:
    st.session_state.individual_df = None

# =========================
# ESTILOS CSS
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
    
    .kpi-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        margin: 0.75rem 0;
        border-left: 4px solid #667eea;
        transition: all 0.3s ease;
    }
    
    .kpi-card:hover {
        transform: translateX(5px);
        box-shadow: 0 6px 16px rgba(0,0,0,0.12);
    }
    
    .kpi-card h4 {
        margin: 0 0 0.5rem 0;
        color: #667eea;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .kpi-card .value {
        font-size: 2rem;
        font-weight: 900;
        color: #333;
        margin: 0.5rem 0;
    }
    
    .kpi-card .description {
        font-size: 0.85rem;
        color: #6c757d;
        line-height: 1.4;
    }
    
    .stock-detail-popup {
        background: linear-gradient(135deg, #fff8e1 0%, #ffecb3 100%);
        padding: 2.5rem;
        border-radius: 20px;
        margin-top: 2rem;
        border: 4px solid #ffc107;
        box-shadow: 0 12px 40px rgba(255, 193, 7, 0.4);
        animation: slideDown 0.4s ease-out;
    }
    
    @keyframes slideDown {
        from {
            opacity: 0;
            transform: translateY(-20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .stock-detail-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 24px rgba(102, 126, 234, 0.3);
    }
    
    .stock-detail-header h2 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 900;
    }
    
    .stock-detail-header .subtitle {
        font-size: 1.2rem;
        opacity: 0.9;
        margin-top: 0.5rem;
    }
    
    .enhanced-table {
        background: white;
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        margin: 1.5rem 0;
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
    
    .click-instruction {
        background: linear-gradient(135deg, #e8eaf6 0%, #c5cae9 100%);
        padding: 1.25rem;
        border-radius: 12px;
        text-align: center;
        font-weight: 600;
        color: #3f51b5;
        margin: 1rem 0;
        border: 2px dashed #3f51b5;
    }
</style>
""", unsafe_allow_html=True)

# =========================
# MAPEO DE BOLSAS
# =========================
BOLSAS_SUFIJOS = {
    "NYSE/NASDAQ (USA)": "",
    "Bolsa Mexicana de Valores (BMV)": ".MX",
    "Bolsa de Toronto (TSX)": ".TO",
    "Bolsa de Londres (LSE)": ".L",
    "Bolsa de Frankfurt (FRA)": ".DE",
    "Bolsa de París (EPA)": ".PA",
    "Bolsa de Madrid (BME)": ".MC",
    "Bolsa de São Paulo (B3)": ".SA",
    "Bolsa de Buenos Aires (BCBA)": ".BA",
    "Bolsa de Tokio (TSE)": ".T",
    "Bolsa de Hong Kong (HKEX)": ".HK",
    "Bolsa de Shanghái (SSE)": ".SS",
    "Bolsa de Australia (ASX)": ".AX"
}

# =========================
# FUNCIONES DE DATOS
# =========================

@st.cache_data(ttl=3600)
def get_company_details(ticker: str) -> Dict:
    """Obtiene detalles completos de la empresa"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        return {
            'nombre': info.get('longName', info.get('shortName', ticker)),
            'sector': info.get('sector', 'N/D'),
            'industria': info.get('industry', 'N/D'),
            'empleados': info.get('fullTimeEmployees', 0),
            'descripcion': info.get('longBusinessSummary', 'N/D'),
            'market_cap': info.get('marketCap', 0),
            'pe_ratio': info.get('trailingPE', 0),
            'forward_pe': info.get('forwardPE', 0),
            'peg_ratio': info.get('pegRatio', 0),
            'price_to_book': info.get('priceToBook', 0),
            'dividend_yield': info.get('dividendYield', 0),
            'beta': info.get('beta', 0),
            'week_52_high': info.get('fiftyTwoWeekHigh', 0),
            'week_52_low': info.get('fiftyTwoWeekLow', 0),
            'profit_margin': info.get('profitMargins', 0),
            'operating_margin': info.get('operatingMargins', 0),
            'roe': info.get('returnOnEquity', 0),
            'roa': info.get('returnOnAssets', 0),
            'revenue_growth': info.get('revenueGrowth', 0),
            'earnings_growth': info.get('earningsGrowth', 0),
            'debt_to_equity': info.get('debtToEquity', 0),
            'website': info.get('website', ''),
            'country': info.get('country', 'N/D'),
            'city': info.get('city', 'N/D')
        }
    except:
        return {
            'nombre': ticker,
            'sector': 'N/D',
            'industria': 'N/D',
            'empleados': 0,
            'descripcion': 'No disponible',
            'market_cap': 0,
            'pe_ratio': 0,
            'beta': 0,
            'country': 'N/D',
            'city': 'N/D'
        }

@st.cache_data(ttl=3600)
def get_batch_stock_data(tickers: List[str], period: str = "1y") -> Dict:
    """Descarga datos en batch"""
    try:
        tickers_str = " ".join(tickers)
        data = yf.download(tickers_str, period=period, group_by='ticker', auto_adjust=True, progress=False, threads=False)
        
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
    except:
        return {}

def calculate_metrics(prices: pd.DataFrame) -> Dict:
    """Calcula métricas financieras"""
    if prices.empty or 'Close' not in prices.columns:
        return {}
    
    returns = prices['Close'].pct_change().dropna()
    
    return {
        'rendimiento_total': ((prices['Close'].iloc[-1] / prices['Close'].iloc[0]) - 1) * 100,
        'volatilidad_anual': returns.std() * np.sqrt(252) * 100,
        'rendimiento_anual': returns.mean() * 252 * 100,
        'max_drawdown': ((prices['Close'] / prices['Close'].cummax()) - 1).min() * 100,
        'sharpe_ratio': (returns.mean() * 252) / (returns.std() * np.sqrt(252)) if returns.std() != 0 else 0,
        'returns': returns
    }

def calculate_portfolio_metrics(batch_data: Dict, weights: Dict) -> Dict:
    """Calcula métricas del portafolio"""
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
    
    return {
        'rendimiento_total': (cumulative_returns.iloc[-1] - 1) * 100,
        'volatilidad_anual': portfolio_returns.std() * np.sqrt(252) * 100,
        'rendimiento_anual': portfolio_returns.mean() * 252 * 100,
        'max_drawdown': ((cumulative_returns / cumulative_returns.cummax()) - 1).min() * 100,
        'sharpe_ratio': (portfolio_returns.mean() * 252) / (portfolio_returns.std() * np.sqrt(252)) if portfolio_returns.std() != 0 else 0,
        'cumulative_returns': cumulative_returns,
        'dates': dates
    }

# =========================
# FUNCIONES DE IA
# =========================

def generate_portfolio_with_gemini(perfil: str, horizonte: str, capital: str, objetivos: List[str], sector_pref: str, bolsas: List[str]) -> Tuple[List[Dict], str]:
    """Genera portafolio con Gemini - JUSTIFICACIÓN MÁXIMO 150 PALABRAS"""
    objetivos_str = ", ".join(objetivos)
    bolsas_str = ", ".join(bolsas)
    
    prompt = f"""
    Eres un asesor financiero experto. Genera un portafolio óptimo.
    
    **Perfil:**
    - Riesgo: {perfil}
    - Horizonte: {horizonte}
    - Capital: {capital}
    - Objetivos: {objetivos_str}
    - Sector: {sector_pref}
    - Bolsas: {bolsas_str}
    
    **Instrucciones:**
    1. 5-10 acciones/ETFs
    2. NYSE/NASDAQ: sin sufijo (AAPL, MSFT)
    3. BMV: sufijo .MX (WALMEX.MX)
    4. Diversifica
    5. Justificación: MÁXIMO 150 palabras, concisa y directa, sin mencionar el límite de palabras
    
    **Responde SOLO JSON:**
```json
    {{
        "portafolio": [
            {{"ticker": "AAPL", "peso": 15, "razon": "Líder tecnológico"}},
            ...
        ],
        "justificacion": "Este portafolio combina..."
    }}
```
    
    IMPORTANTE: Justificación de máximo 150 palabras sin mencionar cuántas palabras tiene.
    """
    
    try:
        response = model.generate_content(prompt)
        response_text = response.text.strip().replace('```json', '').replace('```', '').strip()
        portfolio_data = json.loads(response_text)
        portafolio = portfolio_data.get('portafolio', [])
        justificacion = portfolio_data.get('justificacion', '')
        
        total_peso = sum([p['peso'] for p in portafolio])
        if abs(total_peso - 100) > 1:
            for p in portafolio:
                p['peso'] = (p['peso'] / total_peso) * 100
        
        return portafolio, justificacion
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return [], ""

def analyze_portfolio_with_gemini(portfolio_data: pd.DataFrame, metrics: Dict, perfil: str, justificacion_inicial: str) -> str:
    """Análisis final - MÁXIMO 300 PALABRAS SIN MENCIONAR EL LÍMITE"""
    prompt = f"""
    Como asesor financiero, proporciona un análisis profesional y conciso del portafolio.
    
    **Contexto Inicial:** {justificacion_inicial}
    **Datos:** {portfolio_data.to_string()}
    **Métricas:**
    - Rendimiento Total: {metrics.get('rendimiento_total', 0):.2f}%
    - Volatilidad: {metrics.get('volatilidad_anual', 0):.2f}%
    - Sharpe: {metrics.get('sharpe_ratio', 0):.2f}
    - Max DD: {metrics.get('max_drawdown', 0):.2f}%
    
    **Perfil:** {perfil}
    
    Estructura tu análisis en 4 secciones breves:
    
    1. **Evaluación General del Desempeño**
       - Análisis del Sharpe Ratio y rendimiento ajustado por riesgo
       - Alineación con el perfil de riesgo
    
    2. **Análisis de Posiciones Clave**
       - Identifica las 2 mejores y 2 peores posiciones
       - Evalúa si las ponderaciones son apropiadas
    
    3. **Gestión de Riesgo**
       - Evalúa el Max Drawdown y volatilidad
       - Comenta sobre la diversificación sectorial
    
    4. **Recomendaciones Accionables**
       - Acciones específicas: mantener/aumentar/reducir posiciones
       - Próximos pasos para el inversionista
    
    Sé directo, específico y profesional. NO menciones límites de palabras ni cuántas palabras tiene tu respuesta.
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"

# =========================
# FUNCIÓN PARA RENDERIZAR DETALLE DE ACCIÓN
# =========================

def render_stock_detail_popup(ticker: str, details: Dict, batch_data: Dict):
    """Renderiza el detalle completo de la acción seleccionada"""
    
    st.markdown(f"""
    <div class="stock-detail-popup">
        <div class="stock-detail-header">
            <h2>📊 {details['nombre']}</h2>
            <div class="subtitle">{ticker} | {details['sector']} | {details['industria']}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Botón para cerrar
    if st.button("✖️ Cerrar Detalle", key="close_detail_btn"):
        st.session_state.selected_stock_ticker = None
        st.rerun()
    
    st.markdown("---")
    
    # Información básica
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="kpi-card">
            <h4>📍 Ubicación</h4>
            <div class="value" style="font-size: 1.5rem">{details['city']}</div>
            <div class="description">{details['country']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        empleados = f"{details['empleados']:,}" if details['empleados'] > 0 else "N/D"
        st.markdown(f"""
        <div class="kpi-card">
            <h4>👥 Empleados</h4>
            <div class="value" style="font-size: 1.5rem">{empleados}</div>
            <div class="description">Fuerza laboral</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        market_cap = details['market_cap']
        if market_cap > 1e12:
            cap_str = f"${market_cap/1e12:.2f}T"
        elif market_cap > 1e9:
            cap_str = f"${market_cap/1e9:.2f}B"
        elif market_cap > 1e6:
            cap_str = f"${market_cap/1e6:.2f}M"
        else:
            cap_str = "N/D"
        
        st.markdown(f"""
        <div class="kpi-card">
            <h4>💰 Market Cap</h4>
            <div class="value" style="font-size: 1.5rem">{cap_str}</div>
            <div class="description">Capitalización</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        peso_info = next((item['peso'] for item in st.session_state.portfolio_details if item['ticker'] == ticker), 0)
        st.markdown(f"""
        <div class="kpi-card">
            <h4>📊 Peso en Portafolio</h4>
            <div class="value" style="font-size: 1.5rem; color: #667eea">{peso_info:.1f}%</div>
            <div class="description">Ponderación</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Razón
    razon_info = next((item['razon'] for item in st.session_state.portfolio_details if item['ticker'] == ticker), "")
    st.markdown(f"""
    <div class="info-card">
        <strong>💡 Razón de inclusión:</strong><br>{razon_info}
    </div>
    """, unsafe_allow_html=True)
    
    # Descripción
    if details['descripcion'] and details['descripcion'] != 'N/D':
        with st.expander("📄 Sobre la empresa", expanded=False):
            st.write(details['descripcion'])
            if details.get('website'):
                st.markdown(f"🌐 **Website:** [{details['website']}]({details['website']})")
    
    st.markdown("---")
    
    # KPIs en tabs
    tab1, tab2, tab3 = st.tabs(["📊 Valuación", "💼 Operativas", "📈 Técnico"])
    
    with tab1:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            pe = details.get('pe_ratio', 0)
            pe_color = "#4CAF50" if 0 < pe < 20 else "#FF9800" if 0 < pe < 30 else "#f44336" if pe > 0 else "#999"
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">P/E Ratio</div>
                <div class="metric-value" style="color: {pe_color}">{pe:.2f if pe else 'N/D'}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            pb = details.get('price_to_book', 0)
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">P/B Ratio</div>
                <div class="metric-value">{pb:.2f if pb else 'N/D'}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            div_yield = details.get('dividend_yield', 0) * 100 if details.get('dividend_yield') else 0
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">Dividend Yield</div>
                <div class="metric-value" style="color: #4CAF50">{div_yield:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            beta = details.get('beta', 0)
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">Beta</div>
                <div class="metric-value">{beta:.2f if beta else 'N/D'}</div>
            </div>
            """, unsafe_allow_html=True)
    
    with tab2:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            profit_margin = details.get('profit_margin', 0) * 100 if details.get('profit_margin') else 0
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">Margen Utilidad</div>
                <div class="metric-value" style="color: {'#4CAF50' if profit_margin > 20 else '#FF9800'}">{profit_margin:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            roe = details.get('roe', 0) * 100 if details.get('roe') else 0
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">ROE</div>
                <div class="metric-value" style="color: {'#4CAF50' if roe > 15 else '#FF9800'}">{roe:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            rev_growth = details.get('revenue_growth', 0) * 100 if details.get('revenue_growth') else 0
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">Crecimiento</div>
                <div class="metric-value" style="color: {'#4CAF50' if rev_growth > 0 else '#f44336'}">{rev_growth:+.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
    
    with tab3:
        if ticker in batch_data and batch_data[ticker]['success']:
            hist = batch_data[ticker]['history']
            metrics = calculate_metrics(hist)
            
            # Gráfico de velas
            fig_candle = go.Figure()
            
            fig_candle.add_trace(go.Candlestick(
                x=hist.index,
                open=hist['Open'],
                high=hist['High'],
                low=hist['Low'],
                close=hist['Close'],
                name='Precio'
            ))
            
            sma_20 = hist['Close'].rolling(window=20).mean()
            sma_50 = hist['Close'].rolling(window=50).mean()
            
            fig_candle.add_trace(go.Scatter(x=hist.index, y=sma_20, name='SMA 20', line=dict(color='#FF9800', width=2)))
            fig_candle.add_trace(go.Scatter(x=hist.index, y=sma_50, name='SMA 50', line=dict(color='#2196F3', width=2)))
            
            fig_candle.update_layout(
                title=f'Gráfico de Velas - {ticker}',
                yaxis_title='Precio ($)',
                xaxis_rangeslider_visible=False,
                height=400,
                template='plotly_white'
            )
            
            st.plotly_chart(fig_candle, use_container_width=True)
            
            # Métricas
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Rendimiento 1Y", f"{metrics.get('rendimiento_total', 0):+.2f}%")
            with col2:
                st.metric("Volatilidad", f"{metrics.get('volatilidad_anual', 0):.1f}%")
            with col3:
                st.metric("Sharpe", f"{metrics.get('sharpe_ratio', 0):.2f}")
            with col4:
                st.metric("Max DD", f"{metrics.get('max_drawdown', 0):.1f}%")

# =========================
# INTERFAZ PRINCIPAL
# =========================

st.markdown("""
<div class="main-header">
    <h1>🤖 Asesor de Portafolio con IA</h1>
    <p>Portafolios personalizados generados por Gemini 2.5 Flash</p>
</div>
""", unsafe_allow_html=True)

# CUESTIONARIO (siempre visible si no hay portafolio generado)
if not st.session_state.portfolio_generated:
    st.markdown('<div class="questionnaire-card">', unsafe_allow_html=True)
    st.header("📋 Cuestionario de Perfil de Inversión")

    col1, col2 = st.columns(2)

    with col1:
        perfil = st.selectbox("🎯 Perfil de riesgo", ["Conservador", "Moderado", "Agresivo"])
        horizonte = st.selectbox("⏰ Horizonte", ["Corto plazo (< 1 año)", "Mediano plazo (1-5 años)", "Largo plazo (> 5 años)"], index=1)
        capital = st.selectbox("💰 Capital", ["< $10,000", "$10,000 - $50,000", "$50,000 - $100,000", "> $100,000"], index=1)

    with col2:
        objetivos = st.multiselect(
            "🎯 Objetivos",
            ["Crecimiento de capital", "Generación de ingresos (dividendos)", "Preservación de capital", "Diversificación internacional", "Inversión ESG/Sostenible"],
            default=["Crecimiento de capital"]
        )
        sector_pref = st.selectbox("🏢 Sector", ["Sin preferencia (diversificado)", "Tecnología", "Salud", "Finanzas", "Energía", "Consumo"], index=0)
        bolsas_preferidas = st.multiselect("🌍 Bolsas", list(BOLSAS_SUFIJOS.keys()), default=["NYSE/NASDAQ (USA)"])

    st.markdown('</div>', unsafe_allow_html=True)

    generate_btn = st.button("🚀 GENERAR PORTAFOLIO", type="primary", use_container_width=True)

    if generate_btn:
        if not objetivos or not bolsas_preferidas:
            st.error("❌ Completa todos los campos")
            st.stop()
        
        with st.spinner("🧠 Generando portafolio..."):
            portafolio, justificacion = generate_portfolio_with_gemini(perfil, horizonte, capital, objetivos, sector_pref, bolsas_preferidas)
        
        if not portafolio:
            st.error("❌ Error generando portafolio")
            st.stop()
        
        # Guardar en session state
        st.session_state.portafolio = portafolio
        st.session_state.justificacion = justificacion
        
        # Obtener detalles
        with st.spinner("📥 Obteniendo detalles..."):
            portfolio_details = []
            for item in portafolio:
                details = get_company_details(item['ticker'])
                portfolio_details.append({**item, **details})
            st.session_state.portfolio_details = portfolio_details
            time.sleep(1)
        
        # Obtener datos históricos
        tickers = [p['ticker'] for p in portafolio]
        with st.spinner("📈 Descargando datos..."):
            batch_data = get_batch_stock_data(tickers, "1y")
            st.session_state.batch_data = batch_data
            time.sleep(1)
        
        # Calcular métricas
        successful_tickers = [t for t in tickers if t in batch_data and batch_data[t]['success']]
        weights = {p['ticker']: p['peso'] / 100 for p in portafolio}
        total_successful_weight = sum([weights[t] for t in successful_tickers])
        adjusted_weights = {t: weights[t] / total_successful_weight for t in successful_tickers}
        
        st.session_state.successful_tickers = successful_tickers
        st.session_state.adjusted_weights = adjusted_weights
        
        # Calcular métricas individuales
        individual_data = []
        for ticker in successful_tickers:
            hist = batch_data[ticker]['history']
            metrics = calculate_metrics(hist)
            
            individual_data.append({
                'Ticker': ticker,
                'Peso': f"{adjusted_weights[ticker]*100:.1f}%",
                'Precio': f"${batch_data[ticker]['current_price']:.2f}",
                'Rendimiento': f"{'🟢' if metrics.get('rendimiento_total', 0) > 0 else '🔴'} {metrics.get('rendimiento_total', 0):.2f}%",
                'Volatilidad': f"{metrics.get('volatilidad_anual', 0):.1f}%",
                'Sharpe': f"{metrics.get('sharpe_ratio', 0):.2f}",
                'Max DD': f"{metrics.get('max_drawdown', 0):.1f}%"
            })
        
        st.session_state.individual_df = pd.DataFrame(individual_data)
        
        # Calcular métricas del portafolio
        portfolio_metrics = calculate_portfolio_metrics(batch_data, adjusted_weights)
        st.session_state.portfolio_metrics = portfolio_metrics
        
        # Marcar como generado
        st.session_state.portfolio_generated = True
        st.rerun()

    else:
        st.info("👆 Completa el cuestionario y presiona el botón")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            ### 🤖 IA Personalizada
            - Gemini 2.5 Flash
            - Portafolio óptimo
            - 5-10 posiciones
            - Multi-bolsa
            """)
        
        with col2:
            st.markdown("""
            ### 📊 Análisis Completo
            - Métricas avanzadas
            - Visualizaciones
            - Dashboard por acción
            - KPIs detallados
            """)
        
        with col3:
            st.markdown("""
            ### 💡 Interactivo
            - Click para ver detalle
            - Pop-up con información
            - KPIs bursátiles
            - Análisis técnico
            """)

# MOSTRAR PORTAFOLIO SI YA ESTÁ GENERADO
else:
    # Botón para generar nuevo portafolio
    if st.button("🔄 Generar Nuevo Portafolio", type="secondary"):
        # Resetear todo
        st.session_state.portfolio_generated = False
        st.session_state.selected_stock_ticker = None
        st.session_state.portfolio_details = None
        st.session_state.batch_data = None
        st.session_state.portafolio = None
        st.session_state.justificacion = None
        st.rerun()
    
    st.markdown(f'<div class="success-badge">✅ {len(st.session_state.portafolio)} posiciones generadas</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    # GRÁFICO DE PASTEL
    st.header("📊 Portafolio Sugerido por IA")
    st.markdown(f'<div class="info-card"><strong>💡 Estrategia:</strong> {st.session_state.justificacion}</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="click-instruction">👇 Selecciona una acción para ver su análisis detallado</div>', unsafe_allow_html=True)
    
    # Pie chart
    labels_list = []
    values_list = []
    hover_text_list = []
    
    for item in st.session_state.portfolio_details:
        labels_list.append(f"{item['ticker']}<br>{item['peso']:.1f}%")
        values_list.append(item['peso'])
        
        empleados_str = f"{item['empleados']:,}" if item['empleados'] > 0 else "N/D"
        hover_info = (
            f"<b>{item['nombre']}</b><br>"
            f"Sector: {item['sector']}<br>"
            f"Industria: {item['industria']}<br>"
            f"Empleados: {empleados_str}<br>"
            f"Peso: {item['peso']:.1f}%"
        )
        hover_text_list.append(hover_info)
    
    colors = ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe', '#00f2fe', '#43e97b', '#38f9d7', '#fa709a', '#fee140']
    
    fig_pie = go.Figure()
    
    fig_pie.add_trace(go.Pie(
        labels=labels_list,
        values=values_list,
        hovertext=hover_text_list,
        hoverinfo='text',
        hole=0.45,
        marker=dict(colors=colors[:len(st.session_state.portfolio_details)], line=dict(color='white', width=4)),
        textposition='inside',
        textfont=dict(size=15, color='white', family='Inter', weight='bold')
    ))
    
    fig_pie.update_layout(
        title={
            'text': '<b>Composición del Portafolio</b>',
            'y':0.98,
            'x':0.45,
            'xanchor': 'center',
            'font': dict(size=24, family='Inter', color='#333')
        },
        height=550,
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.15,
            font=dict(size=12, family='Inter'),
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor="#667eea",
            borderwidth=2
        ),
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Inter",
            bordercolor="#667eea",
            align="left"
        ),
        margin=dict(l=20, r=350, t=80, b=20)
    )
    
    st.plotly_chart(fig_pie, use_container_width=True)
    
    # BOTONES PARA SELECCIONAR ACCIÓN
    st.subheader("🔍 Selecciona una acción:")
    
    num_cols = 5
    for i in range(0, len(st.session_state.portfolio_details), num_cols):
        cols = st.columns(num_cols)
        for j, col in enumerate(cols):
            if i + j < len(st.session_state.portfolio_details):
                item = st.session_state.portfolio_details[i + j]
                with col:
                    if st.button(
                        f"📊 {item['ticker']}\n{item['peso']:.1f}%",
                        key=f"select_{item['ticker']}",
                        use_container_width=True,
                        type="primary" if st.session_state.selected_stock_ticker == item['ticker'] else "secondary"
                    ):
                        st.session_state.selected_stock_ticker = item['ticker']
                        st.rerun()
    
    # MOSTRAR DETALLE SI HAY SELECCIÓN
    if st.session_state.selected_stock_ticker:
        st.markdown("---")
        st.markdown("---")
        
        selected_ticker = st.session_state.selected_stock_ticker
        selected_details = next((item for item in st.session_state.portfolio_details if item['ticker'] == selected_ticker), None)
        
        if selected_details:
            render_stock_detail_popup(selected_ticker, selected_details, st.session_state.batch_data)
    
    # ANÁLISIS DEL PORTAFOLIO
    st.markdown("---")
    st.markdown("---")
    
    st.header("📊 Análisis Individual de Activos")
    
    st.markdown('<div class="enhanced-table">', unsafe_allow_html=True)
    st.dataframe(st.session_state.individual_df, use_container_width=True, hide_index=True, height=400)
    st.markdown('</div>', unsafe_allow_html=True)
    
    csv = st.session_state.individual_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Descargar CSV", csv, f"portafolio_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv")
    
    # MÉTRICAS
    st.markdown("---")
    st.header("📈 Métricas del Portafolio Ponderado")
    
    portfolio_metrics = st.session_state.portfolio_metrics
    
    if portfolio_metrics:
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            rend_total = portfolio_metrics.get('rendimiento_total', 0)
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">Rendimiento Total</div>
                <div class="metric-value" style="color: {'#4CAF50' if rend_total > 0 else '#f44336'}">{rend_total:+.2f}%</div>
                <div class="metric-sublabel">1 Año</div>
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
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">Sharpe Ratio</div>
                <div class="metric-value" style="color: {sharpe_color}">{sharpe:.2f}</div>
                <div class="metric-sublabel">{'⭐⭐⭐' if sharpe > 1.5 else '⭐⭐' if sharpe > 0.5 else '⭐'}</div>
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
        
        # GRÁFICO
        st.markdown("---")
        st.header("📊 Visualización del Portafolio")
        
        cumulative = portfolio_metrics.get('cumulative_returns')
        dates = portfolio_metrics.get('dates')
        
        if cumulative is not None:
            fig_perf = go.Figure()
            
            fig_perf.add_trace(go.Scatter(
                x=dates,
                y=(cumulative - 1) * 100,
                line=dict(color='#667eea', width=4),
                fill='tonexty',
                fillcolor='rgba(102, 126, 234, 0.2)'
            ))
            
            fig_perf.add_hline(y=0, line_dash="dot", line_color="gray", line_width=2)
            
            fig_perf.update_layout(
                title='Rendimiento Acumulado del Portafolio',
                yaxis_title='Rendimiento (%)',
                height=500,
                template='plotly_white'
            )
            
            st.plotly_chart(fig_perf, use_container_width=True)
        
        # ANÁLISIS IA
        st.markdown("---")
        st.markdown("""
        <div class="ai-section">
            <h2>🤖 Análisis Final con IA</h2>
            <p>Evaluación profesional del portafolio generado</p>
        </div>
        """, unsafe_allow_html=True)
        
        with st.spinner("🧠 Analizando..."):
            final_analysis = analyze_portfolio_with_gemini(
                st.session_state.individual_df,
                portfolio_metrics,
                "Perfil seleccionado",  # Podrías guardar esto también en session_state
                st.session_state.justificacion
            )
            st.markdown(final_analysis)
        
        st.success("✅ Análisis completo")

st.markdown("---")
st.caption("🤖 Powered by Gemini 2.5 Flash | © 2025")