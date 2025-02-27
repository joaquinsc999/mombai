# Mombai

Mombai es una librería de aprendizaje profundo diseñada para implementar y experimentar con capas de redes neuronales avanzadas, basada en investigaciones recientes. Esta librería incluye implementaciones de Kolmogorov-Arnold Networks (KANs) y modelos innovadores como **WAFNet** y **MAXNet**, los cuales exploran nuevas formas de combinar y seleccionar funciones de activación dinámicamente.

## Motivación

El proyecto Mombai nace de la necesidad de explorar y llevar a la práctica conceptos avanzados de redes neuronales presentados en papers de investigación recientes. La librería está en sus primeras fases de desarrollo y busca proporcionar herramientas para experimentar con activaciones híbridas y mecanismos de atención.

A futuro, se espera mejorar la eficiencia de estas implementaciones e incorporar nuevas capas basadas en experimentos innovadores en inteligencia artificial.

## Instalación

Puedes instalar la librería directamente desde PyPI usando pip:

```bash
pip install mombai
```

## Uso

### 1. Uso de `KANLayer`
Ejemplo de cómo usar la capa `KANLayer` para entrenar un modelo simple que ajuste la función y = 3x + 2:

```python
import tensorflow as tf
from mombai.layers.kan import KANLayer

class KANModel(tf.keras.Model):
    def __init__(self, units=1):
        super(KANModel, self).__init__()
        self.kan_layer = KANLayer(units=units, G=5, k=3)
        self.output_layer = tf.keras.layers.Dense(1)

    def call(self, inputs):
        x = self.kan_layer(inputs)
        return self.output_layer(x)

# Generación de datos
x_train = tf.random.uniform((1000, 1), -1, 1)
y_train = 3 * x_train + 2

# Crear y entrenar el modelo
model = KANModel(units=10)
model.compile(optimizer='adam', loss='mse')
model.fit(x_train, y_train, epochs=10, batch_size=32)

# Probar el modelo
x_test = tf.constant([[0.5]], dtype=tf.float32)
y_pred = model.predict(x_test)
print(f"Predicción para x=0.5: {y_pred}")
```

---

## 2. Uso de `WAFLayer`
`WAFLayer` es una capa que permite combinar múltiples funciones de activación en una sola transformación de los datos. Se pueden sumar o promediar las activaciones antes de aplicarlas.

Ejemplo de uso en una red densa con compresor de suma:

```python
import tensorflow as tf
from mombai.layers.multi_activation import WAFLayer

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, input_shape=(10,)),
    WAFLayer(units=32, activations=['relu', 'swish', 'gelu'], compressor="sum"),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.summary()
```

Ejemplo de uso después de una capa convolucional con compresor de promedio:

```python
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    WAFLayer(units=64, activations=['relu', 'swish', 'elu'], compressor="avg"),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()
```

---

## 3. Uso de `MAXLayerWithAttention`
Esta capa permite que la red **seleccione dinámicamente** qué funciones de activación usar en cada entrada mediante un mecanismo de atención.

Ejemplo de uso en una red:

```python
import tensorflow as tf
from mombai.layers.multi_activation import MAXLayerWithAttention

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, input_shape=(10,)),
    MAXLayerWithAttention(units=32, activations=['relu', 'swish', 'gelu']),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.summary()
```

---

## 4. Uso de `MAXLayerWithSelfAttention`
En esta variante, la capa usa **autoatención** para ajustar la combinación de activaciones en función del contexto de la entrada.

Ejemplo en un modelo LSTM para clasificación de texto:

```python
import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dense
from mombai.layers.multi_activation import MAXLayerWithSelfAttention

model = tf.keras.Sequential([
    Embedding(input_dim=5000, output_dim=128, input_length=100),
    LSTM(64, return_sequences=True),
    MAXLayerWithSelfAttention(units=32, activations=['relu', 'swish', 'gelu']),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
```

---

## Funciones consideradas en layers multi_activation
- Nota: No todas las funciones han sido probadas. Puede que exista algún bug con una función en particular.
```json
{
  "relu": "tf.nn.relu",
  "sigmoid": "tf.nn.sigmoid",
  "tanh": "tf.nn.tanh",
  "softmax": "tf.nn.softmax",
  "softplus": "tf.nn.softplus",
  "softsign": "tf.nn.softsign",
  "elu": "tf.nn.elu",
  "selu": "tf.nn.selu",
  "swish": "tf.nn.swish",
  "gelu": "tf.nn.gelu",
  "leaky_relu": "tf.nn.leaky_relu",
  "relu6": "tf.nn.relu6",
  "hard_sigmoid": "tf.keras.activations.hard_sigmoid",
  "exponential": "tf.keras.activations.exponential",
  "linear": "tf.keras.activations.linear",
  "log_softmax": "tf.nn.log_softmax",
  "crelu": "tf.nn.crelu",
}
```


## Estado del Proyecto

Actualmente, Mombai se encuentra en una fase inicial de desarrollo, y las implementaciones están orientadas a experimentar con funciones de activación avanzadas y mecanismos de atención. A futuro, planeamos optimizar el rendimiento y añadir nuevas capas.

Si encuentras problemas o tienes sugerencias, no dudes en abrir un issue o contribuir al proyecto.

## Contribuciones

Las contribuciones son bienvenidas. Si quieres contribuir, revisa `CONTRIBUTING.md` (próximamente) y asegúrate de que tus cambios se alineen con la dirección general del proyecto.

## Licencia

Este proyecto está licenciado bajo la licencia MIT. Consulta el archivo `LICENSE` para más detalles.
