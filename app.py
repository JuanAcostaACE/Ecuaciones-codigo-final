import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import io
import json
from datetime import datetime

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SysDyn Analyzer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');

:root {
    --bg-dark: #0a0e1a;
    --bg-card: #111827;
    --bg-card2: #1a2235;
    --accent-cyan: #00d4ff;
    --accent-green: #00ff88;
    --accent-red: #ff3366;
    --accent-amber: #ffaa00;
    --text-primary: #e8edf7;
    --text-muted: #6b7fa3;
    --border: rgba(0,212,255,0.15);
}

/* Global */
.stApp { background: var(--bg-dark); color: var(--text-primary); font-family: 'Syne', sans-serif; }
.main .block-container { padding: 1.5rem 2rem; max-width: 1400px; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: var(--bg-card) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: var(--text-primary) !important; }

/* Sliders */
.stSlider > div > div > div { color: var(--accent-cyan) !important; }

/* Header */
.hero-header {
    text-align: center;
    padding: 1.5rem 0 1rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.5rem;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.4rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    background: linear-gradient(135deg, #00d4ff 0%, #00ff88 50%, #00d4ff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
}
.hero-sub {
    color: var(--text-muted);
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-top: 0.3rem;
}

/* Metric Cards */
.metric-card {
    background: var(--bg-card2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--accent-cyan), transparent);
}
.metric-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--text-muted);
    margin-bottom: 0.4rem;
}
.metric-value {
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem;
    font-weight: 800;
}
.metric-stable { color: var(--accent-green); }
.metric-unstable { color: var(--accent-red); }
.metric-neutral { color: var(--accent-cyan); }
.metric-warn { color: var(--accent-amber); }

/* Section Headers */
.section-header {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    color: var(--accent-cyan);
    border-left: 3px solid var(--accent-cyan);
    padding-left: 0.7rem;
    margin: 1.5rem 0 0.8rem;
}

/* Alert Banners */
.banner-stable {
    background: rgba(0,255,136,0.08);
    border: 1px solid rgba(0,255,136,0.3);
    border-radius: 8px;
    padding: 0.7rem 1rem;
    color: var(--accent-green);
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
}
.banner-unstable {
    background: rgba(255,51,102,0.08);
    border: 1px solid rgba(255,51,102,0.3);
    border-radius: 8px;
    padding: 0.7rem 1rem;
    color: var(--accent-red);
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
}
.formula-box {
    background: var(--bg-card2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin: 0.5rem 0;
}
.sidebar-section {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--accent-cyan);
    margin: 1.2rem 0 0.4rem;
    padding-bottom: 0.3rem;
    border-bottom: 1px solid var(--border);
}
</style>
""", unsafe_allow_html=True)

# ─── Data Computation Functions ──────────────────────────────────────────────

def compute_network_model(t, alpha, beta, omega, A_cos, A_sin, A_base):
    """
    Network load model (ODE-inspired):
    L(t) = A_base * e^(alpha*t) * [A_cos*cos(omega*t) + A_sin*sin(omega*t)] + beta*t
    """
    envelope = A_base * np.exp(alpha * t)
    oscillation = A_cos * np.cos(omega * t) + A_sin * np.sin(omega * t)
    trend = beta * t
    L = envelope * oscillation + trend
    dL = np.gradient(L, t)  # derivative (rate of change)
    return L, dL

def compute_buffer_model(t, C_max, tau, L0, noise_level=0.0):
    """
    Buffer filling model: B(t) = C_max * (1 - e^(-t/tau)) + L0
    Asymptotic convergence to C_max.
    """
    B = C_max * (1 - np.exp(-t / tau)) + L0
    if noise_level > 0:
        rng = np.random.default_rng(42)
        noise = rng.normal(0, noise_level * C_max * 0.05, size=len(t))
        B = B + noise * np.exp(-t / (2 * tau))  # noise decays over time
    B = np.clip(B, 0, C_max * 1.5)
    return B

def routh_hurwitz_stability(alpha, beta):
    """
    Simplified stability criterion based on system parameters.
    For the network model, stability requires:
    - alpha < 0 (envelope must decay)
    - |beta| small (no explosive trend)
    """
    net_stable = alpha < 0
    buffer_stable = beta >= 0  # buffer must be filling, not draining unboundedly
    return net_stable, buffer_stable

def settlement_time(t, B, C_max, threshold=0.98):
    """Find when buffer reaches threshold% of C_max."""
    target = threshold * C_max
    idx = np.where(B >= target)[0]
    if len(idx) == 0:
        return None
    return t[idx[0]]

def compute_poles(alpha, omega):
    """Compute complex poles of the system."""
    # Poles: s = alpha ± j*omega
    return complex(alpha, omega), complex(alpha, -omega)

# ─── Plot Functions ──────────────────────────────────────────────────────────

PLOT_TEMPLATE = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(17,24,39,0.8)',
    font=dict(family='JetBrains Mono, monospace', color='#6b7fa3', size=10),
    xaxis=dict(
        gridcolor='rgba(0,212,255,0.08)',
        linecolor='rgba(0,212,255,0.2)',
        tickcolor='rgba(0,212,255,0.3)',
        title_font=dict(color='#00d4ff', size=11),
    ),
    yaxis=dict(
        gridcolor='rgba(0,212,255,0.08)',
        linecolor='rgba(0,212,255,0.2)',
        tickcolor='rgba(0,212,255,0.3)',
        title_font=dict(color='#00d4ff', size=11),
    ),
    legend=dict(
        bgcolor='rgba(17,24,39,0.9)',
        bordercolor='rgba(0,212,255,0.2)',
        borderwidth=1,
        font=dict(color='#e8edf7', size=10),
    ),
    margin=dict(l=50, r=30, t=50, b=50),
)

def make_network_plot(t, L, dL, net_stable, scenario_b=None):
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Carga de Red L(t)", "Tasa de Cambio dL/dt"),
    )

    color_main = '#00ff88' if net_stable else '#ff3366'
    color_deriv = '#00d4ff'

    fig.add_trace(go.Scatter(
        x=t, y=L, name='L(t) — Escenario A',
        line=dict(color=color_main, width=2.5),
        fill='tozeroy', fillcolor=f'rgba({",".join(str(int(c*255)) for c in ([0,1,0.53] if net_stable else [1,0.2,0.4]))},0.07)',
        hovertemplate='t=%{x:.2f}s<br>L=%{y:.4f}<extra></extra>',
    ), row=1, col=1)

    if scenario_b is not None:
        fig.add_trace(go.Scatter(
            x=t, y=scenario_b, name='L(t) — Escenario B',
            line=dict(color='#ffaa00', width=2, dash='dash'),
            hovertemplate='t=%{x:.2f}s<br>L(B)=%{y:.4f}<extra></extra>',
        ), row=1, col=1)

    fig.add_hline(y=0, line=dict(color='rgba(255,255,255,0.15)', width=1, dash='dot'), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=t, y=dL, name='dL/dt',
        line=dict(color=color_deriv, width=1.8),
        hovertemplate='t=%{x:.2f}s<br>dL/dt=%{y:.4f}<extra></extra>',
    ), row=2, col=1)

    fig.add_hline(y=0, line=dict(color='rgba(255,255,255,0.15)', width=1, dash='dot'), row=2, col=1)

    fig.update_layout(
        height=480,
        title=dict(text='', font=dict(size=13)),
        xaxis2=dict(title='Tiempo (s)', **PLOT_TEMPLATE['xaxis']),
        yaxis=dict(title='Carga L(t)', **PLOT_TEMPLATE['yaxis']),
        yaxis2=dict(title='dL/dt', **PLOT_TEMPLATE['yaxis']),
    )
    for ann in fig.layout.annotations:
        ann.font = dict(color='#00d4ff', size=11, family='JetBrains Mono')
    return fig


def make_buffer_plot(t, B, C_max, tau, settle_t, scenario_b=None):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=t, y=B, name='Buffer B(t) — Escenario A',
        line=dict(color='#00ff88', width=2.5),
        fill='tozeroy', fillcolor='rgba(0,255,136,0.07)',
        hovertemplate='t=%{x:.2f}s<br>B=%{y:.2f} MB<extra></extra>',
    ))

    if scenario_b is not None:
        fig.add_trace(go.Scatter(
            x=t, y=scenario_b, name='Buffer — Escenario B',
            line=dict(color='#ffaa00', width=2, dash='dash'),
            hovertemplate='t=%{x:.2f}s<br>B(B)=%{y:.2f} MB<extra></extra>',
        ))

    # Asymptote line
    fig.add_hline(y=C_max, line=dict(color='rgba(255,170,0,0.6)', width=1.5, dash='dash'),
                  annotation_text=f'C_max = {C_max:.0f} MB',
                  annotation_font=dict(color='#ffaa00', size=10))

    # 98% threshold
    fig.add_hline(y=0.98 * C_max, line=dict(color='rgba(0,212,255,0.4)', width=1, dash='dot'),
                  annotation_text='98%',
                  annotation_font=dict(color='#00d4ff', size=9))

    # Settlement time marker
    if settle_t is not None:
        B_at_settle = C_max * (1 - np.exp(-settle_t / tau))
        fig.add_trace(go.Scatter(
            x=[settle_t], y=[B_at_settle],
            mode='markers',
            marker=dict(color='#ffaa00', size=12, symbol='diamond',
                        line=dict(color='white', width=1.5)),
            name=f'T98% = {settle_t:.1f}s',
            hovertemplate=f't_98% = {settle_t:.2f}s<extra></extra>',
        ))

    fig.update_layout(
        height=380,
        xaxis=dict(title='Tiempo (s)', **PLOT_TEMPLATE['xaxis']),
        yaxis=dict(title='Ocupación Buffer (MB)', **PLOT_TEMPLATE['yaxis']),
    )
    return fig


def make_pole_plot(alpha, omega):
    pole1, pole2 = compute_poles(alpha, omega)
    colors = ['#ff3366' if alpha >= 0 else '#00ff88'] * 2

    fig = go.Figure()

    # Right-half plane shading (unstable region)
    fig.add_shape(type='rect', x0=0, x1=max(2, abs(alpha)+1),
                  y0=-max(2, abs(omega)+1), y1=max(2, abs(omega)+1),
                  fillcolor='rgba(255,51,102,0.05)',
                  line=dict(color='rgba(255,51,102,0.2)', width=1))

    # Imaginary axis
    fig.add_vline(x=0, line=dict(color='rgba(255,255,255,0.2)', width=1.5))
    fig.add_hline(y=0, line=dict(color='rgba(255,255,255,0.2)', width=1.5))

    fig.add_trace(go.Scatter(
        x=[pole1.real, pole2.real],
        y=[pole1.imag, pole2.imag],
        mode='markers+text',
        marker=dict(color=colors, size=14, symbol='x', line=dict(color='white', width=2)),
        text=[f'p₁ = {alpha:.2f}+{omega:.2f}j', f'p₂ = {alpha:.2f}-{omega:.2f}j'],
        textposition='top right',
        textfont=dict(color='#e8edf7', size=9, family='JetBrains Mono'),
        name='Polos del sistema',
        hovertemplate='Re=%{x:.3f}<br>Im=%{y:.3f}<extra></extra>',
    ))

    lim = max(abs(alpha) + 1, abs(omega) + 1, 2)
    fig.update_layout(
        height=280,
        xaxis=dict(title='Parte Real (σ)', range=[-lim, lim], **PLOT_TEMPLATE['xaxis']),
        yaxis=dict(title='Parte Imaginaria (jω)', range=[-lim, lim], **PLOT_TEMPLATE['yaxis']),
    )
    return fig

# ─── Session State ───────────────────────────────────────────────────────────
if 'scenario_a_net' not in st.session_state:
    st.session_state.scenario_a_net = None
if 'scenario_a_buf' not in st.session_state:
    st.session_state.scenario_a_buf = None
if 'noise_seed' not in st.session_state:
    st.session_state.noise_seed = 0

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div style="font-family:Syne,sans-serif;font-size:1.1rem;font-weight:800;color:#00d4ff;padding:0.5rem 0 1rem;">⚡ SysDyn Analyzer</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">🌐 Modelo de Red — Parámetros</div>', unsafe_allow_html=True)

    alpha = st.slider("α — Coeficiente exponencial", -1.5, 0.5, -0.3, 0.05,
                      help="Controla el decaimiento (α<0) o crecimiento (α>0) del envelope")
    beta = st.slider("β — Tendencia lineal", -2.0, 2.0, 0.1, 0.1,
                     help="Pendiente de la componente lineal de la carga")
    omega = st.slider("ω — Frecuencia de oscilación (rad/s)", 0.1, 10.0, 2.0, 0.1)
    A_cos = st.slider("A_cos — Amplitud coseno", 0.0, 5.0, 1.5, 0.1)
    A_sin = st.slider("A_sin — Amplitud seno", 0.0, 5.0, 0.8, 0.1)
    A_base = st.slider("A_base — Ganancia base", 0.1, 5.0, 1.0, 0.1)

    st.markdown('<div class="sidebar-section">💾 Modelo de Buffer — Parámetros</div>', unsafe_allow_html=True)

    C_max = st.slider("C_max — Capacidad máxima (MB)", 50, 1000, 256, 10)
    tau = st.slider("τ — Constante de tiempo (s)", 0.5, 30.0, 5.0, 0.5,
                    help="Tiempo característico de convergencia al 63.2% de C_max")
    L0 = st.slider("L₀ — Nivel inicial (MB)", 0, 100, 0, 5)

    st.markdown('<div class="sidebar-section">⏱ Simulación</div>', unsafe_allow_html=True)
    t_end = st.slider("Tiempo total (s)", 10, 100, 40, 5)
    t_points = st.select_slider("Resolución (puntos)", options=[200, 500, 1000, 2000], value=500)

    st.markdown('<div class="sidebar-section">🔬 Stress Test</div>', unsafe_allow_html=True)
    noise_level = st.slider("Nivel de ruido (%)", 0, 50, 0, 5,
                            help="Añade ruido aleatorio al buffer (0 = sin ruido)")
    if st.button("🎲 Regenerar Ruido", use_container_width=True):
        st.session_state.noise_seed = np.random.randint(0, 9999)

    st.markdown('<div class="sidebar-section">📊 Comparativa de Escenarios</div>', unsafe_allow_html=True)
    if st.button("💾 Guardar Escenario A", use_container_width=True):
        t_tmp = np.linspace(0, t_end, t_points)
        L_tmp, _ = compute_network_model(t_tmp, alpha, beta, omega, A_cos, A_sin, A_base)
        B_tmp = compute_buffer_model(t_tmp, C_max, tau, L0)
        st.session_state.scenario_a_net = L_tmp.copy()
        st.session_state.scenario_a_buf = B_tmp.copy()
        st.success("Escenario A guardado ✓")
    if st.button("🗑 Limpiar Escenario A", use_container_width=True):
        st.session_state.scenario_a_net = None
        st.session_state.scenario_a_buf = None

# ─── Main Computations ───────────────────────────────────────────────────────
t = np.linspace(0.001, t_end, t_points)
L, dL = compute_network_model(t, alpha, beta, omega, A_cos, A_sin, A_base)
B = compute_buffer_model(t, C_max, tau, L0, noise_level / 100)

net_stable, buf_stable = routh_hurwitz_stability(alpha, beta)
settle_t = settlement_time(t, B, C_max)

L_max = float(np.max(np.abs(L)))
L_rms = float(np.sqrt(np.mean(L**2)))
buf_final = float(B[-1])
buf_pct = buf_final / C_max * 100

# Adjust noise seed for reproducibility
np.random.seed(st.session_state.noise_seed)

# ─── Header ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
  <p class="hero-title">SysDyn Analyzer</p>
  <p class="hero-sub">Modelado de Estabilidad · Redes &amp; Buffers · Análisis por EDOs</p>
</div>
""", unsafe_allow_html=True)

# ─── Stability Banner ─────────────────────────────────────────────────────────
overall_stable = net_stable and buf_stable
if overall_stable:
    st.markdown('<div class="banner-stable">✅ SISTEMA ESTABLE — Criterio de Routh-Hurwitz satisfecho. Los polos se encuentran en el semiplano izquierdo (α < 0).</div>', unsafe_allow_html=True)
else:
    reasons = []
    if not net_stable:
        reasons.append(f"α = {alpha} ≥ 0 → envelope divergente")
    if not buf_stable:
        reasons.append(f"β = {beta} < 0 → tendencia de red inestable")
    st.markdown(f'<div class="banner-unstable">⚠️ SISTEMA INESTABLE — {" | ".join(reasons)}. Ajuste los parámetros para estabilizar.</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── Metric Cards ─────────────────────────────────────────────────────────────
col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    stability_class = "metric-stable" if net_stable else "metric-unstable"
    stability_text = "ESTABLE" if net_stable else "INESTABLE"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Red — Estado</div>
        <div class="metric-value {stability_class}">{stability_text}</div>
    </div>""", unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Carga Pico |L|_max</div>
        <div class="metric-value metric-neutral">{L_max:.3f}</div>
    </div>""", unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">RMS Carga</div>
        <div class="metric-value metric-neutral">{L_rms:.3f}</div>
    </div>""", unsafe_allow_html=True)

with col4:
    buf_class = "metric-stable" if buf_pct < 90 else ("metric-warn" if buf_pct < 100 else "metric-unstable")
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Buffer Final</div>
        <div class="metric-value {buf_class}">{buf_pct:.1f}%</div>
    </div>""", unsafe_allow_html=True)

with col5:
    if settle_t is not None:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">T₉₈% Asentamiento</div>
            <div class="metric-value metric-warn">{settle_t:.1f}s</div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">T₉₈% Asentamiento</div>
            <div class="metric-value metric-unstable">>&nbsp;{t_end}s</div>
        </div>""", unsafe_allow_html=True)

with col6:
    p1, _ = compute_poles(alpha, omega)
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Polo p₁</div>
        <div class="metric-value {'metric-stable' if net_stable else 'metric-unstable'}" style="font-size:1.1rem;">
            {alpha:.2f}±{omega:.2f}j
        </div>
    </div>""", unsafe_allow_html=True)

# ─── Network Model ────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">🌐 Modelo de Red — Dinámica de Carga</div>', unsafe_allow_html=True)

net_fig = make_network_plot(t, L, dL, net_stable,
                            scenario_b=st.session_state.scenario_a_net if st.session_state.scenario_a_net is not None and len(st.session_state.scenario_a_net) == len(t) else None)
st.plotly_chart(net_fig, use_container_width=True, config={'displayModeBar': True, 'scrollZoom': True})

# ─── Buffer + Pole-Zero ────────────────────────────────────────────────────────
st.markdown('<div class="section-header">💾 Modelo de Buffer &amp; Mapa de Polos</div>', unsafe_allow_html=True)

col_buf, col_poles = st.columns([2, 1])

with col_buf:
    buf_fig = make_buffer_plot(t, B, C_max, tau, settle_t,
                               scenario_b=st.session_state.scenario_a_buf if st.session_state.scenario_a_buf is not None and len(st.session_state.scenario_a_buf) == len(t) else None)
    st.plotly_chart(buf_fig, use_container_width=True, config={'displayModeBar': True, 'scrollZoom': True})

with col_poles:
    pole_fig = make_pole_plot(alpha, omega)
    st.plotly_chart(pole_fig, use_container_width=True)

# ─── Mathematical Analysis ────────────────────────────────────────────────────
st.markdown('<div class="section-header">🧮 Análisis Matemático</div>', unsafe_allow_html=True)

col_m1, col_m2 = st.columns(2)

with col_m1:
    st.markdown('<div class="formula-box">', unsafe_allow_html=True)
    st.markdown("**Modelo de Red (EDO)**")
    st.latex(
        rf"L(t) = {A_base:.2f} \cdot e^{{{alpha:.2f}t}} "
        rf"\left[ {A_cos:.2f}\cos({omega:.2f}t) + {A_sin:.2f}\sin({omega:.2f}t) \right] "
        rf"+ {beta:.2f}t"
    )
    st.markdown(f"**Función de transferencia:** polos en $s = {alpha:.2f} \\pm {omega:.2f}j$")
    stability_cond = "✅ Estable" if net_stable else "❌ Inestable"
    st.markdown(f"**Condición Routh-Hurwitz:** $\\alpha < 0$ → $\\alpha = {alpha:.2f}$ → **{stability_cond}**")
    st.markdown('</div>', unsafe_allow_html=True)

with col_m2:
    st.markdown('<div class="formula-box">', unsafe_allow_html=True)
    st.markdown("**Modelo de Buffer (EDO de primer orden)**")
    st.latex(
        rf"\frac{{dB}}{{dt}} = \frac{{1}}{{\tau}}\left(C_{{max}} - B(t)\right)"
    )
    st.latex(
        rf"B(t) = {C_max} \cdot \left(1 - e^{{-t/{tau:.1f}}}\right) + {L0}"
    )
    st.markdown(f"**Constante de tiempo:** $\\tau = {tau:.1f}$ s → al $63.2\\%$ en $t=\\tau$")
    settle_str = f"{settle_t:.1f}s" if settle_t else f"> {t_end}s"
    st.markdown(f"**Tiempo de asentamiento al 98%:** $t_{{98}} \\approx {settle_str}$ (≈ $4.6\\tau = {4.6*tau:.1f}$s)")
    st.markdown('</div>', unsafe_allow_html=True)

# ─── Export Section ────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📥 Exportar Datos</div>', unsafe_allow_html=True)

col_e1, col_e2, col_e3 = st.columns(3)

with col_e1:
    df_export = pd.DataFrame({
        'tiempo_s': t,
        'carga_red_L': L,
        'tasa_cambio_dL': dL,
        'buffer_B_MB': B,
    })
    csv_bytes = df_export.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇️ Descargar CSV",
        data=csv_bytes,
        file_name=f"sysdyn_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime='text/csv',
        use_container_width=True,
    )

with col_e2:
    params = {
        'red': {'alpha': alpha, 'beta': beta, 'omega': omega,
                'A_cos': A_cos, 'A_sin': A_sin, 'A_base': A_base},
        'buffer': {'C_max': C_max, 'tau': tau, 'L0': L0},
        'resultados': {
            'red_estable': net_stable,
            'L_max': round(L_max, 4),
            'L_rms': round(L_rms, 4),
            'buf_final_pct': round(buf_pct, 2),
            'T98_s': round(settle_t, 2) if settle_t else None,
        }
    }
    st.download_button(
        label="⬇️ Descargar JSON",
        data=json.dumps(params, indent=2, ensure_ascii=False).encode('utf-8'),
        file_name=f"sysdyn_params_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime='application/json',
        use_container_width=True,
    )

with col_e3:
    summary_lines = [
        "=" * 48,
        "  SYSDYN ANALYZER — Reporte de Simulación",
        "=" * 48,
        f"  Generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "── PARÁMETROS DE RED ──────────────────────────",
        f"  α (exp coef)   = {alpha}",
        f"  β (tendencia)  = {beta}",
        f"  ω (frecuencia) = {omega} rad/s",
        f"  A_cos          = {A_cos}",
        f"  A_sin          = {A_sin}",
        f"  A_base         = {A_base}",
        "",
        "── PARÁMETROS DE BUFFER ───────────────────────",
        f"  C_max          = {C_max} MB",
        f"  τ (tau)        = {tau} s",
        f"  L₀ (inicial)   = {L0} MB",
        "",
        "── RESULTADOS ─────────────────────────────────",
        f"  Red estable    : {'SÍ ✓' if net_stable else 'NO ✗'}",
        f"  Carga máx |L|  : {L_max:.4f}",
        f"  RMS carga      : {L_rms:.4f}",
        f"  Buffer final   : {buf_pct:.1f}%  ({buf_final:.1f} / {C_max} MB)",
        f"  T₉₈%           : {f'{settle_t:.1f}s' if settle_t else f'> {t_end}s'}",
        f"  Polos          : s = {alpha:.3f} ± {omega:.3f}j",
        "=" * 48,
    ]
    report_txt = "\n".join(summary_lines)
    st.download_button(
        label="⬇️ Descargar Reporte TXT",
        data=report_txt.encode('utf-8'),
        file_name=f"sysdyn_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime='text/plain',
        use_container_width=True,
    )

# ─── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:2rem 0 1rem;color:#2d3f5c;font-family:'JetBrains Mono',monospace;font-size:0.65rem;letter-spacing:0.1em;">
    SysDyn Analyzer · Modelado de Sistemas Dinámicos mediante EDOs · Streamlit + Plotly
</div>
""", unsafe_allow_html=True)
