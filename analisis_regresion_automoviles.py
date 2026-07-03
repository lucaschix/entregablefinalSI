import io

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


# En este script voy revisando los datos de automoviles y despues entreno
# una regresion lineal para estimar el precio.

# Primero cargo el archivo con los datos que voy a usar.
df = pd.read_csv("automovil_dataset.csv")

# Antes de modelar, reviso el tamaño, los tipos de datos, las estadisticas
# generales y si hay valores nulos que puedan afectar el analisis.
print("exploracion inicial")
print("shape:", df.shape)

print("info:")
buffer = io.StringIO()
df.info(buf=buffer)
print(buffer.getvalue().lower())

print("estadisticas descriptivas:")
print(df.describe())

print("valores nulos:")
print(df.isnull().sum())

# Despues miro las correlaciones para tener una idea inicial de que variables
# se relacionan mas con el precio.
print("correlacion")
correlacion = df.corr(numeric_only=True)
print(correlacion)
print("correlacion con price:")
print(correlacion["price"].drop("price").sort_values(key=abs, ascending=False).to_string())

# Dejo en X las variables que usare para predecir y en y el precio del auto.
X = df[["horsepower", "age", "mileage", "engine_size"]]
y = df["price"]

# Divido los datos para entrenar con una parte y comprobar el resultado con
# datos que el modelo no vio. Mantengo random_state para repetir el resultado.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42
)

print("variables del modelo")
print("variable dependiente: price")
print("variables independientes: horsepower, age, mileage, engine_size")
print("division de datos: 70% entrenamiento y 30% prueba")

# Con statsmodels reviso la parte estadistica del modelo, por eso agrego la
# constante para que la regresion tambien calcule el intercepto.
modelo_sm = sm.OLS(y_train, sm.add_constant(X_train)).fit()
print("modelo statsmodels ols")
resumen_sm = "\n".join(
    linea
    for linea in str(modelo_sm.summary()).lower().splitlines()
    if set(linea.strip()) not in [{"="}, {"-"}]
)
print(resumen_sm)

# Tambien entreno el modelo con sklearn, que me sirve para hacer predicciones
# de forma mas directa.
modelo = LinearRegression()
modelo.fit(X_train, y_train)

# Calculo predicciones para comparar como responde el modelo en entrenamiento
# y en prueba.
pred_train = modelo.predict(X_train)
pred_test = modelo.predict(X_test)

# Los residuos me muestran cuanto se equivoca el modelo en cada prediccion.
residuos = y_test - pred_test

# Uso estas metricas para entender mejor el rendimiento: r2 indica que parte
# de la variacion se explica, mientras mse, rmse y mae muestran el tamaño del
# error desde distintos puntos de vista.
metricas = pd.DataFrame(
    {
        "conjunto": ["entrenamiento", "prueba"],
        "r2": [r2_score(y_train, pred_train), r2_score(y_test, pred_test)],
        "mse": [
            mean_squared_error(y_train, pred_train),
            mean_squared_error(y_test, pred_test),
        ],
        "rmse": [
            np.sqrt(mean_squared_error(y_train, pred_train)),
            np.sqrt(mean_squared_error(y_test, pred_test)),
        ],
        "mae": [
            mean_absolute_error(y_train, pred_train),
            mean_absolute_error(y_test, pred_test),
        ],
    }
)


print(f"intercepto: {modelo.intercept_:,.4f}")
for variable, coeficiente in zip(X.columns, modelo.coef_):
    print(f"coeficiente {variable}: {coeficiente:,.4f}")

print("metricas")
print(metricas)

print("analisis de residuos en prueba")
print(f"media de residuos: {residuos.mean():,.4f}")
print(f"desviacion estandar de residuos: {residuos.std():,.4f}")
print(f"residuo minimo: {residuos.min():,.4f}")
print(f"residuo maximo: {residuos.max():,.4f}")


# Finalmente pruebo el modelo con un automovil especifico para obtener una
# estimacion concreta de precio.
auto_nuevo = pd.DataFrame(
    {"horsepower": [165], "age": [4], "mileage": [58000], "engine_size": [2.0]}
)
precio_estimado = modelo.predict(auto_nuevo)[0]

# Muestro los datos usados y el precio que entrega el modelo.
print("prediccion solicitada")
print("horsepower = 165 hp")
print("age = 4 anos")
print("mileage = 58.000 km")
print("engine size = 2.0 l")
print(f"precio estimado: {precio_estimado:,.2f}")
