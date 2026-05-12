# ⚡ SysDyn Analyzer

Aplicación web interactiva para modelado de estabilidad de sistemas informáticos
(Redes y Buffers) mediante Ecuaciones Diferenciales Ordinarias (EDOs).

## Características

### Modelos matemáticos
- **Red:** `L(t) = A_base · e^(αt) · [A_cos·cos(ωt) + A_sin·sin(ωt)] + β·t`
- **Buffer:** `B(t) = C_max · (1 - e^(-t/τ)) + L₀`

### Funcionalidades
- ✅ **Sliders interactivos** para todos los parámetros en la barra lateral
- 📊 **Gráficas Plotly** con zoom, hover y scroll interactivo
- 🔬 **Criterio de Routh-Hurwitz** — detección automática de estabilidad
- 📍 **Mapa de Polos** en el plano complejo
- ⏱ **Cálculo de T₉₈%** — tiempo de asentamiento del buffer
- 🎲 **Stress Test** — ruido aleatorio configurable
- 📊 **Comparativa A/B** — guarda y compara dos escenarios
- 🧮 **Análisis matemático** con fórmulas LaTeX dinámicas
- 📥 **Exportación** en CSV, JSON y reporte TXT

## Instalación y ejecución

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Ejecutar la aplicación
streamlit run app.py

# 3. Abrir en el navegador
# → http://localhost:8501
```

## Estructura del código

```
app.py
├── compute_network_model()   # EDO de carga de red
├── compute_buffer_model()    # EDO de buffer con ruido opcional
├── routh_hurwitz_stability() # Criterio de estabilidad
├── settlement_time()         # Cálculo de T₉₈%
├── compute_poles()           # Polos del sistema
├── make_network_plot()       # Gráfica de carga de red (Plotly)
├── make_buffer_plot()        # Gráfica del buffer (Plotly)
├── make_pole_plot()          # Mapa de polos (Plotly)
└── UI (Streamlit)
    ├── Sidebar — sliders y controles
    ├── Banner de estabilidad
    ├── Métricas (6 columnas)
    ├── Gráfica de red
    ├── Gráfica de buffer + mapa de polos
    ├── Análisis matemático (LaTeX)
    └── Exportación (CSV / JSON / TXT)
```

## Parámetros configurables

| Parámetro | Descripción | Rango |
|-----------|-------------|-------|
| α (alpha) | Coeficiente exponencial (α<0 → estable) | [-1.5, 0.5] |
| β (beta)  | Tendencia lineal de la carga | [-2.0, 2.0] |
| ω (omega) | Frecuencia de oscilación (rad/s) | [0.1, 10.0] |
| A_cos     | Amplitud componente coseno | [0.0, 5.0] |
| A_sin     | Amplitud componente seno | [0.0, 5.0] |
| A_base    | Ganancia base del envelope | [0.1, 5.0] |
| C_max     | Capacidad máxima del buffer (MB) | [50, 1000] |
| τ (tau)   | Constante de tiempo de convergencia (s) | [0.5, 30.0] |
| L₀        | Nivel inicial del buffer (MB) | [0, 100] |
