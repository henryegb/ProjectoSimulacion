# Robot del Tipo de Cambio (GTQ/USD) — Banguat + Pronóstico

App **Streamlit** que descarga el histórico oficial del **Tipo de Cambio de Referencia** (GTQ/USD) desde el WebService de **Banguat** (desde 1998) y permite pronosticar fechas futuras. Si seleccionas una **fecha pasada**, devuelve el dato **oficial**; si es **futura**, devuelve un **pronóstico** (con intervalo).

## Estructura
- `app.py`: aplicación Streamlit.
- `requirements.txt`: dependencias completas (incluye `prophet`/`pystan`).
- `requirements-lite.txt`: dependencias **sin** Prophet, por si en tu Mac compilar PyStan es complejo. En la app puedes usar **ARIMA** o **Random Walk**.
- `run.sh`: script para Mac/Linux que crea un entorno y ejecuta la app.
- `Dockerfile`: imagen lista para ejecutar con Docker.

## Ejecutar en macOS (recomendado)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt   # Si falla Prophet, usa: pip install -r requirements-lite.txt
streamlit run app.py
```

Abre el navegador en la URL que te muestre (normalmente `http://localhost:8501`).

### Nota sobre Prophet en Mac (M1/M2/M3)
Si la instalación de `prophet`/`pystan` se complica, ejecuta con `requirements-lite.txt` y en la interfaz selecciona **ARIMA** o **Random Walk** para los pronósticos.

## Deploy en Streamlit Community Cloud
1. Disponible en streamlit para ejecutar en cualquier momento

## Docker (opcional)
```bash
docker build -t tcr-robot .
docker run -p 8501:8501 tcr-robot
# abre http://localhost:8501
```

## Créditos / Curso
Proyecto para Modelación y Simulación — Robot de TCR GTQ/USD.
