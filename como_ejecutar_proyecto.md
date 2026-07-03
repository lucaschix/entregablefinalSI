# paquete de regresion de automoviles

esta carpeta contiene los archivos necesarios para reproducir el analisis:

- `analisis_regresion_automoviles.py`
- `automovil_dataset.csv`
- `requirements.txt`

## como ejecutarlo

desde esta misma carpeta, crear y activar un entorno virtual:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
python analisis_regresion_automoviles.py
```

si la activacion del entorno no se mantiene, usar:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/pip install -r requirements.txt
.venv/bin/python analisis_regresion_automoviles.py
```
