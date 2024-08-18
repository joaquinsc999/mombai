import tensorflow as tf
from tensorflow.keras.initializers import GlorotUniform
from ..activations.splines import KANBspline

class KANLayer(tf.keras.layers.Layer):
    def __init__(self, units=3, G=5, k=3):
        super(KANLayer, self).__init__()
        self.units = units
        self.k = k
        self.G = G
        self.n = G + k + 1

    def build(self, input_shape):
        # Generación de nudos (knots) para las splines
        self.knots = tf.Variable(
            initial_value=tf.linspace(-1.0, 1.0, self.n),
            trainable=False  # Los knots generalmente no son entrenables
        )
        
        # Inicialización de coeficientes de las splines cerca de 0
        self.coefs = tf.Variable(
            initial_value=tf.random.normal(
                shape=(input_shape[-1], self.units, self.G + self.k,),
                mean=0.0, stddev=0.01  # Inicialización para mantener spline(x) ≈ 0
            ),
            trainable=True,
            name='coefs'
        )

        # Inicialización Xavier para los pesos de activaciones fijas y splines
        self.fixed_activation_weights = tf.Variable(
            initial_value=GlorotUniform()(shape=(input_shape[-1], self.units)),
            trainable=True,
            name='fixed_activation_weights'
        )

        self.spline_activation_weights = tf.Variable(
            initial_value=GlorotUniform()(shape=(input_shape[-1], self.units)),
            trainable=True,
            name='spline_activation_weights'
        )

    def call(self, inputs):
        # Aplicar las splines a las conexiones (weights)
        activated_inputs = []
        
        # Loop sobre cada unidad (dimensión 2)
        for i in range(self.units):
            coefs_unit = self.coefs[:, i, :]
            weights_fixed_activation = self.fixed_activation_weights[:, i]
            weigths_spline = self.spline_activation_weights[:, i]
            
            # Mapear inputs con los pesos correspondientes para cada feature
            activated_input = tf.map_fn(
                lambda x: KANBspline(x, self.knots, self.k, coefs_unit, weights_fixed_activation, weigths_spline),
                inputs,
                fn_output_signature=tf.float32
            )
            activated_inputs.append(activated_input)
        
        # Concatenar resultados para obtener la salida final de forma [batch_size, units]
        output = tf.stack(activated_inputs, axis=-1)
        
        return tf.reduce_sum(output, axis=-2)
    
