
# Mombai

Mombai es una librería de aprendizaje profundo diseñada para implementar y experimentar con capas de redes neuronales avanzadas, basadas en investigaciones recientes. Esta librería incluye implementaciones de Kolmogorov-Arnold Networks (KANs) y está orientada a explorar nuevas arquitecturas de redes neuronales que capturen mejor las complejidades no lineales en los datos.

## Motivación

El proyecto Mombai nace de la necesidad de explorar y llevar a la práctica conceptos avanzados de redes neuronales presentados en papers de investigación recientes. La librería está en sus primeras fases de desarrollo, con el objetivo de ofrecer implementaciones iniciales de estas arquitecturas. Aunque Mombai aún no está completamente optimizado y algunas partes de la implementación son un primer borrador, es un excelente punto de partida para aquellos interesados en entender y experimentar con estos conceptos.

En futuras versiones, planeamos no solo mejorar la eficiencia de estas implementaciones, sino también añadir nuevas capas y arquitecturas basadas en investigaciones científicas personales y experimentos innovadores en el campo de la inteligencia artificial.

## Instalación

Puedes instalar la librería directamente desde PyPI usando `pip`:

```bash
pip install mombai
```

## Uso

Aquí tienes un ejemplo básico de cómo usar la capa KANLayer para entrenar un modelo simple que ajuste la función `y = 3x + 2`:

```python
import tensorflow as tf
from mombai.layers.kan import KANLayer

# Definición del modelo usando la KANLayer
class KANModel(tf.keras.Model):
    def __init__(self, units=1):
        super(KANModel, self).__init__()
        self.kan_layer = KANLayer(units=units, G=5, k=3)  # Capa KAN
        self.output_layer = tf.keras.layers.Dense(1)  # Capa de salida simple

    def call(self, inputs):
        x = self.kan_layer(inputs)
        return self.output_layer(x)

# Generación de datos para la función y = 3x + 2
def generate_data():
    x = tf.random.uniform((1000, 1), -1, 1)
    y = 3 * x + 2
    return x, y

# Crear los datos de entrenamiento
x_train, y_train = generate_data()

# Crear el modelo
model = KANModel(units=10)

# Compilar el modelo
model.compile(optimizer='adam', loss='mse')

# Entrenar el modelo
model.fit(x_train, y_train, epochs=10, batch_size=32)

# Probar el modelo con un nuevo dato
x_test = tf.constant([[0.5]], dtype=tf.float32)
y_pred = model.predict(x_test)
print(f"Predicción para x=0.5: {y_pred}")
```

## Estado del Proyecto

Esta librería está en una fase inicial y todavía está en desarrollo. Actualmente, las implementaciones están enfocadas en probar los conceptos descritos en los papers de investigación, y se espera que en futuras versiones se mejore la eficiencia y se amplíen las funcionalidades.

En versiones futuras, se incluirán nuevas capas y arquitecturas innovadoras basadas en investigaciones científicas personales. ¡Mantente atento para descubrir estas próximas implementaciones!

Si encuentras problemas o tienes sugerencias, no dudes en abrir un issue o contribuir al proyecto.

## Contribuciones

Las contribuciones son bienvenidas. Si quieres contribuir, por favor, sigue los pasos descritos en `CONTRIBUTING.md` (a crear) y asegúrate de que tus cambios se alineen con la dirección general del proyecto.

## Licencia

Este proyecto está licenciado bajo la licencia MIT. Consulta el archivo `LICENSE` para más detalles.
