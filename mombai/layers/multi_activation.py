import tensorflow as tf
from tensorflow.keras.layers import Layer

# Since param_swish depends on a trainable parameter, we'll handle it directly in the call method.

@tf.keras.utils.register_keras_serializable()
class WAFLayer(Layer):
    def __init__(self, units=32, activations=[], compressor="sum", apply_father_activation=False, **kwargs):
        super(WAFLayer, self).__init__(**kwargs)
        self.units = units
        self.compressor = compressor
        self.apply_father_activation = apply_father_activation

        # Map activation names to functions
        activation_map = {
            "relu": tf.nn.relu,
            "sigmoid": tf.nn.sigmoid,
            "tanh": tf.nn.tanh,
            "softmax": tf.nn.softmax,
            "softplus": tf.nn.softplus,
            "softsign": tf.nn.softsign,
            "elu": tf.nn.elu,
            "selu": tf.nn.selu,
            "swish": tf.nn.swish,
            "gelu": tf.nn.gelu,
            "leaky_relu": tf.nn.leaky_relu,
            "relu6": tf.nn.relu6,
            "hard_sigmoid": tf.keras.activations.hard_sigmoid,
            "exponential": tf.keras.activations.exponential,
            "linear": tf.keras.activations.linear,
            "log_softmax": tf.nn.log_softmax,
            "crelu": tf.nn.crelu
            }

        self.activation_names = []
        self.activations = []
        for act in activations:
            if isinstance(act, str):
                if act in activation_map:
                    self.activations.append(activation_map[act])
                    self.activation_names.append(act)
                else:
                    raise ValueError(f"Unknown activation function name: {act}")
            else:
                # Assume it's a function
                self.activations.append(act)
                self.activation_names.append(act.__name__)

    def build(self, input_shape):
        self.w = self.add_weight(
            name="kernel",
            shape=(input_shape[-1], self.units),
            initializer="glorot_uniform",
            trainable=True
        )

        self.wf = self.add_weight(
            name="kernel_functions",
            shape=(self.units, len(self.activations)),
            initializer="glorot_uniform",
            trainable=True
        )

        self.b = self.add_weight(
            name="bias",
            shape=(self.units,),
            initializer="zeros",
            trainable=True
        )

        self.bf = self.add_weight(
            name="bias_functions",
            shape=(self.units, len(self.activations)),
            initializer="zeros",
            trainable=True
        )

        # Ensure that softingparam is differentiable
        self.softingparam = self.add_weight(
            name="softingparam",
            shape=(1,),  # Scalar value
            initializer=tf.keras.initializers.Constant(1.0),
            trainable=True
        )

        super(WAFLayer, self).build(input_shape)

    def call(self, inputs):
        # Main operation
        weighted_sum = tf.matmul(inputs, self.w) + self.b

        # Expand weighted_sum for compatibility
        expanded_ws = tf.expand_dims(weighted_sum, axis=-1)
        expanded_ws = expanded_ws * self.wf + self.bf

        # Apply the defined activation functions
        results = [func(expanded_ws[..., i]) for i, func in enumerate(self.activations)]

        # Reduce with sum or avg
        if self.compressor == "sum":
            result = tf.reduce_sum(tf.stack(results, axis=-1), axis=-1)
        elif self.compressor == "avg":
            result = tf.reduce_mean(tf.stack(results, axis=-1), axis=-1)
        else:
            raise ValueError("Invalid compressor. Use 'sum' or 'avg'.")

        # Apply the "father_activation" function if enabled
        if self.apply_father_activation:
            # Implement param_swish directly
            result = result * tf.nn.sigmoid(self.softingparam * result)

        return result

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)

    def get_config(self):
        config = super(WAFLayer, self).get_config()
        config.update({
            "units": self.units,
            "activations": self.activation_names,  # Save the names of the functions
            "compressor": self.compressor,
            "apply_father_activation": self.apply_father_activation,
        })
        return config

    @classmethod
    def from_config(cls, config):
        activations = config.pop('activations')
        instance = cls(**config)

        # Map activation names to functions
        activation_map = {
            "relu": tf.nn.relu,
            "sigmoid": tf.nn.sigmoid,
            "tanh": tf.nn.tanh,
            "softmax": tf.nn.softmax,
            "softplus": tf.nn.softplus,
            "softsign": tf.nn.softsign,
            "elu": tf.nn.elu,
            "selu": tf.nn.selu,
            "swish": tf.nn.swish,
            "gelu": tf.nn.gelu,
            "leaky_relu": tf.nn.leaky_relu,
            "relu6": tf.nn.relu6,
            "hard_sigmoid": tf.keras.activations.hard_sigmoid,
            "exponential": tf.keras.activations.exponential,
            "linear": tf.keras.activations.linear,
            "log_softmax": tf.nn.log_softmax,
            "crelu": tf.nn.crelu
            }

        instance.activations = [activation_map[name] for name in activations]
        instance.activation_names = activations

        return instance


@tf.keras.utils.register_keras_serializable()
class WAFLayerNorm(Layer):
    def __init__(self, units=32, activations=[], compressor="sum", apply_father_activation=False, **kwargs):
        super(WAFLayerNorm, self).__init__(**kwargs)
        self.units = units
        self.compressor = compressor
        self.apply_father_activation = apply_father_activation

        # Map activation names to functions
        activation_map = {
            "relu": tf.nn.relu,
            "sigmoid": tf.nn.sigmoid,
            "tanh": tf.nn.tanh,
            "softmax": tf.nn.softmax,
            "softplus": tf.nn.softplus,
            "softsign": tf.nn.softsign,
            "elu": tf.nn.elu,
            "selu": tf.nn.selu,
            "swish": tf.nn.swish,
            "gelu": tf.nn.gelu,
            "leaky_relu": tf.nn.leaky_relu,
            "relu6": tf.nn.relu6,
            "hard_sigmoid": tf.keras.activations.hard_sigmoid,
            "exponential": tf.keras.activations.exponential,
            "linear": tf.keras.activations.linear,
            "log_softmax": tf.nn.log_softmax,
            "crelu": tf.nn.crelu
            }

        self.activation_names = []
        self.activations = []
        for act in activations:
            if isinstance(act, str):
                if act in activation_map:
                    self.activations.append(activation_map[act])
                    self.activation_names.append(act)
                else:
                    raise ValueError(f"Unknown activation function name: {act}")
            else:
                # Se asume que es una función
                self.activations.append(act)
                self.activation_names.append(act.__name__)

    def build(self, input_shape):
        self.w = self.add_weight(
            name="kernel",
            shape=(input_shape[-1], self.units),
            initializer="glorot_uniform",
            trainable=True
        )

        self.wf = self.add_weight(
            name="kernel_functions",
            shape=(self.units, len(self.activations)),
            initializer="glorot_uniform",
            trainable=True
        )

        self.b = self.add_weight(
            name="bias",
            shape=(self.units,),
            initializer="zeros",
            trainable=True
        )

        self.bf = self.add_weight(
            name="bias_functions",
            shape=(self.units, len(self.activations)),
            initializer="zeros",
            trainable=True
        )

        # softingparam es un parámetro diferenciable para la father activation
        self.softingparam = self.add_weight(
            name="softingparam",
            shape=(1,),  # Valor escalar
            initializer=tf.keras.initializers.Constant(1.0),
            trainable=True
        )

        super(WAFLayerNorm, self).build(input_shape)

    def call(self, inputs):
        # Transformación lineal
        weighted_sum = tf.matmul(inputs, self.w) + self.b  # Forma: (batch, units)

        # Normalización estándar: se resta la media y se divide por la desviación estándar
        mean, variance = tf.nn.moments(weighted_sum, axes=[-1], keepdims=True)
        normalized = (weighted_sum - mean) / tf.sqrt(variance + 1e-5)  # Se agrega epsilon para estabilidad

        # Expandir para compatibilidad y aplicar pesos específicos para cada función
        expanded_norm = tf.expand_dims(normalized, axis=-1)  # Forma: (batch, units, 1)
        transformed = expanded_norm * self.wf + self.bf       # Forma: (batch, units, len(activations))

        # Aplicar cada función de activación a su correspondiente canal
        results = [func(transformed[..., i]) for i, func in enumerate(self.activations)]

        # Combinar resultados según el modo compressor: suma o promedio
        if self.compressor == "sum":
            result = tf.reduce_sum(tf.stack(results, axis=-1), axis=-1)
        elif self.compressor == "avg":
            result = tf.reduce_mean(tf.stack(results, axis=-1), axis=-1)
        else:
            raise ValueError("Invalid compressor. Use 'sum' or 'avg'.")

        # Aplicar la father activation si está habilitada
        if self.apply_father_activation:
            result = result * tf.nn.sigmoid(self.softingparam * result)

        return result

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)

    def get_config(self):
        config = super(WAFLayerNorm, self).get_config()
        config.update({
            "units": self.units,
            "activations": self.activation_names,  # Se guardan los nombres de las funciones
            "compressor": self.compressor,
            "apply_father_activation": self.apply_father_activation,
        })
        return config

    @classmethod
    def from_config(cls, config):
        activations = config.pop('activations')
        instance = cls(**config)

        # Mapear nombres de activaciones a funciones
        activation_map = {
            "relu": tf.nn.relu,
            "sigmoid": tf.nn.sigmoid,
            "tanh": tf.nn.tanh,
            "softmax": tf.nn.softmax,
            "softplus": tf.nn.softplus,
            "softsign": tf.nn.softsign,
            "elu": tf.nn.elu,
            "selu": tf.nn.selu,
            "swish": tf.nn.swish,
            "gelu": tf.nn.gelu,
            "leaky_relu": tf.nn.leaky_relu,
            "relu6": tf.nn.relu6,
            "hard_sigmoid": tf.keras.activations.hard_sigmoid,
            "exponential": tf.keras.activations.exponential,
            "linear": tf.keras.activations.linear,
            "log_softmax": tf.nn.log_softmax,
            "crelu": tf.nn.crelu
            }
        instance.activations = [activation_map[name] for name in activations]
        instance.activation_names = activations

        return instance


@tf.keras.utils.register_keras_serializable()
class MAXLayerWithAttention(Layer):
    def __init__(self, units=32, activations=[], return_sequences=True, **kwargs):
        super(MAXLayerWithAttention, self).__init__(**kwargs)
        self.units = units
        self.return_sequences = return_sequences
        self.activation_names = activations

        # Mapa de activaciones
        activation_map = {
            "relu": tf.nn.relu,
            "sigmoid": tf.nn.sigmoid,
            "tanh": tf.nn.tanh,
            "softmax": tf.nn.softmax,
            "softplus": tf.nn.softplus,
            "softsign": tf.nn.softsign,
            "elu": tf.nn.elu,
            "selu": tf.nn.selu,
            "swish": tf.nn.swish,
            "gelu": tf.nn.gelu,
            "leaky_relu": tf.nn.leaky_relu,
            "relu6": tf.nn.relu6,
            "hard_sigmoid": tf.keras.activations.hard_sigmoid,
            "exponential": tf.keras.activations.exponential,
            "linear": tf.keras.activations.linear,
            "log_softmax": tf.nn.log_softmax,
            "crelu": tf.nn.crelu
            }

        self.activations = []
        for act_name in activations:
            if act_name in activation_map:
                self.activations.append(activation_map[act_name])
            else:
                raise ValueError(f"Unknown activation function name: {act_name}")

        self.attention_dense = Dense(len(self.activations), activation='softmax')

    def build(self, input_shape):
        self.input_dim = input_shape[-1]
        self.w = self.add_weight(
            name="kernel",
            shape=(self.input_dim, self.units),
            initializer="glorot_uniform",
            trainable=True
        )
        self.b = self.add_weight(
            name="bias",
            shape=(self.units,),
            initializer="zeros",
            trainable=True
        )
        self.wf = self.add_weight(
            name="kernel_functions",
            shape=(self.units, len(self.activations)),
            initializer="glorot_uniform",
            trainable=True
        )
        self.bf = self.add_weight(
            name="bias_functions",
            shape=(self.units, len(self.activations)),
            initializer="zeros",
            trainable=True
        )

        self.activation_counts = self.add_weight(
            name="activation_counts",
            shape=(len(self.activations),),
            initializer="zeros",
            trainable=False
        )

        self.attention_dense.build(input_shape)
        super(MAXLayerWithAttention, self).build(input_shape)

    def call(self, inputs):
        # Lógica original de la capa ...
        # Ejemplo simplificado
        input_is_2d = (len(inputs.shape) == 2)
        if input_is_2d:
            inputs = tf.expand_dims(inputs, axis=1)

        weighted_sum = tf.matmul(inputs, self.w) + self.b
        expanded_ws = tf.expand_dims(weighted_sum, axis=-1)
        weighted_activations = expanded_ws * self.wf + self.bf

        results = [func(weighted_activations[..., i]) for i, func in enumerate(self.activations)]
        result_activations = tf.stack(results, axis=-1)

        attention_weights = self.attention_dense(inputs)
        attention_weights_expanded = tf.expand_dims(attention_weights, axis=2)
        weighted_results = result_activations * attention_weights_expanded
        final_results = tf.reduce_sum(weighted_results, axis=-1)

        # Pooling si no return_sequences
        if not self.return_sequences:
            final_results = tf.reduce_mean(final_results, axis=1)

        if input_is_2d and self.return_sequences:
            final_results = tf.squeeze(final_results, axis=1)

        activation_selection = tf.reduce_sum(attention_weights, axis=[0,1])
        self.activation_counts.assign_add(activation_selection)

        return final_results

    def get_config(self):
        config = super(MAXLayerWithAttention, self).get_config()
        # Guardar la config de la attention_dense
        attention_dense_config = self.attention_dense.get_config()
        config.update({
            "units": self.units,
            "activations": self.activation_names,
            "return_sequences": self.return_sequences,
            "attention_dense_config": attention_dense_config,
        })
        return config

    @classmethod
    def from_config(cls, config):
        # Extraer la config de attention_dense antes de crear la instancia
        attention_dense_config = config.pop('attention_dense_config')
        instance = cls(**config)

        # Reconstruir la attention_dense capa
        instance.attention_dense = Dense.from_config(attention_dense_config)

        # Reconstruir las activaciones
        activation_map = {
            "relu": tf.nn.relu,
            "sigmoid": tf.nn.sigmoid,
            "tanh": tf.nn.tanh,
            "softmax": tf.nn.softmax,
            "softplus": tf.nn.softplus,
            "softsign": tf.nn.softsign,
            "elu": tf.nn.elu,
            "selu": tf.nn.selu,
            "swish": tf.nn.swish,
            "gelu": tf.nn.gelu,
            "leaky_relu": tf.nn.leaky_relu,
            "relu6": tf.nn.relu6,
            "hard_sigmoid": tf.keras.activations.hard_sigmoid,
            "exponential": tf.keras.activations.exponential,
            "linear": tf.keras.activations.linear,
            "log_softmax": tf.nn.log_softmax,
            "crelu": tf.nn.crelu
            }
        instance.activations = [activation_map[name] for name in instance.activation_names]

        return instance



@tf.keras.utils.register_keras_serializable()
class MAXLayerWithSelfAttention(Layer):
    def __init__(self, units=32, activations=[], return_sequences=True, **kwargs):
        super(MAXLayerWithSelfAttention, self).__init__(**kwargs)
        self.units = units
        self.return_sequences = return_sequences

        # Mapeo de activaciones
        activation_map = {
            "relu": tf.nn.relu,
            "sigmoid": tf.nn.sigmoid,
            "tanh": tf.nn.tanh,
            "softmax": tf.nn.softmax,
            "softplus": tf.nn.softplus,
            "softsign": tf.nn.softsign,
            "elu": tf.nn.elu,
            "selu": tf.nn.selu,
            "swish": tf.nn.swish,
            "gelu": tf.nn.gelu,
            "leaky_relu": tf.nn.leaky_relu,
            "relu6": tf.nn.relu6,
            "hard_sigmoid": tf.keras.activations.hard_sigmoid,
            "exponential": tf.keras.activations.exponential,
            "linear": tf.keras.activations.linear,
            "log_softmax": tf.nn.log_softmax,
            "crelu": tf.nn.crelu
            }

        self.activation_names = []
        self.activations = []
        for act in activations:
            if act in activation_map:
                self.activations.append(activation_map[act])
                self.activation_names.append(act)
            else:
                raise ValueError(f"Unknown activation: {act}")

        # Subcapas densas
        self.q_dense = Dense(units)
        self.k_dense = Dense(units)
        self.v_dense = Dense(units)
        self.activation_weight_dense = Dense(len(self.activations), activation='softmax')

    def build(self, input_shape):
        self.input_dim = input_shape[-1]

        # Pesos principales
        self.w = self.add_weight(
            name="kernel",
            shape=(self.input_dim, self.units),
            initializer="glorot_uniform",
            trainable=True
        )
        self.b = self.add_weight(
            name="bias",
            shape=(self.units,),
            initializer="zeros",
            trainable=True
        )

        # Contador de activaciones
        self.activation_counts = self.add_weight(
            name="activation_counts",
            shape=(len(self.activations),),
            initializer="zeros",
            trainable=False
        )

        # Construir subcapas con shapes simbólicas
        # q_dense, k_dense, v_dense recibirán (batch, timesteps, units)
        # Por lo tanto, input_dim de estas capas es self.units
        # Podemos usar (None, None, self.units) para permitir batch y timesteps variables.
        self.q_dense.build((None, None, self.units))
        self.k_dense.build((None, None, self.units))
        self.v_dense.build((None, None, self.units))

        # activation_weight_dense recibe (batch, units)
        self.activation_weight_dense.build((None, self.units))

        super(MAXLayerWithSelfAttention, self).build(input_shape)

    def call(self, inputs):
        # Detectar si la entrada es 2D (batch, input_dim) o 3D (batch, timesteps, input_dim)
        input_is_2d = (len(inputs.shape) == 2)
        if input_is_2d:
            # Añadir dimensión de tiempo ficticia
            inputs = tf.expand_dims(inputs, axis=1)  # (batch, 1, input_dim)

        # Multiplicación lineal principal
        x = tf.matmul(inputs, self.w) + self.b  # (batch, timesteps, units)

        # Aplicar Q, K, V
        Q = self.q_dense(x)
        K = self.k_dense(x)
        V = self.v_dense(x)

        # Atención
        K_t = tf.transpose(K, [0, 2, 1])
        scores = tf.matmul(Q, K_t) / tf.sqrt(tf.cast(self.units, tf.float32))
        attention_weights = tf.nn.softmax(scores, axis=-1)  # (batch, timesteps, timesteps)
        context = tf.matmul(attention_weights, V)  # (batch, timesteps, units)

        # Vector global
        context_vector = tf.reduce_mean(context, axis=1)  # (batch, units)
        activation_weights = self.activation_weight_dense(context_vector)  # (batch, num_activations)

        # Expandir pesos de activación
        activation_weights_expanded = tf.expand_dims(activation_weights, axis=1)  # (batch, 1, num_activations)
        activation_weights_expanded = tf.expand_dims(activation_weights_expanded, axis=2)  # (batch, 1, 1, num_activations)

        # Aplicar activaciones a x
        acts = [f(x) for f in self.activations]
        acts = tf.stack(acts, axis=-1)  # (batch, timesteps, units, num_activations)
        weighted_activations = acts * activation_weights_expanded
        combined = tf.reduce_sum(weighted_activations, axis=-1)  # (batch, timesteps, units)

        # Si no queremos secuencias, promediamos sobre tiempo
        if not self.return_sequences:
            combined = tf.reduce_mean(combined, axis=1)  # (batch, units)

        # Si originalmente no había secuencia pero return_sequences=True
        # combined tendría (batch, 1, units). Podés quitar el eje de tiempo.
        if input_is_2d and self.return_sequences:
            combined = tf.squeeze(combined, axis=1)

        # Actualizar contador de activaciones
        activation_sum = tf.reduce_sum(activation_weights, axis=0)
        self.activation_counts.assign_add(activation_sum)

        return combined

    def compute_output_shape(self, input_shape):
        if self.return_sequences:
            return (input_shape[0], input_shape[1], self.units)
        else:
            return (input_shape[0], self.units)

    def get_config(self):
        config = super(MAXLayerWithSelfAttention, self).get_config()
        q_dense_config = self.q_dense.get_config()
        k_dense_config = self.k_dense.get_config()
        v_dense_config = self.v_dense.get_config()
        activation_weight_dense_config = self.activation_weight_dense.get_config()

        config.update({
            'units': self.units,
            'activations': self.activation_names,
            'return_sequences': self.return_sequences,
            'q_dense_config': q_dense_config,
            'k_dense_config': k_dense_config,
            'v_dense_config': v_dense_config,
            'activation_weight_dense_config': activation_weight_dense_config,
        })
        return config

    @classmethod
    def from_config(cls, config):
        q_dense_config = config.pop('q_dense_config')
        k_dense_config = config.pop('k_dense_config')
        v_dense_config = config.pop('v_dense_config')
        activation_weight_dense_config = config.pop('activation_weight_dense_config')
        activations = config.pop('activations')

        instance = cls(activations=activations, **config)

        # Reconstruir subcapas
        instance.q_dense = Dense.from_config(q_dense_config)
        instance.k_dense = Dense.from_config(k_dense_config)
        instance.v_dense = Dense.from_config(v_dense_config)
        instance.activation_weight_dense = Dense.from_config(activation_weight_dense_config)

        # Reconstruir activaciones
        activation_map = {
            "relu": tf.nn.relu,
            "sigmoid": tf.nn.sigmoid,
            "tanh": tf.nn.tanh,
            "softmax": tf.nn.softmax,
            "softplus": tf.nn.softplus,
            "softsign": tf.nn.softsign,
            "elu": tf.nn.elu,
            "selu": tf.nn.selu,
            "swish": tf.nn.swish,
            "gelu": tf.nn.gelu,
            "leaky_relu": tf.nn.leaky_relu,
            "relu6": tf.nn.relu6,
            "hard_sigmoid": tf.keras.activations.hard_sigmoid,
            "exponential": tf.keras.activations.exponential,
            "linear": tf.keras.activations.linear,
            "log_softmax": tf.nn.log_softmax,
            "crelu": tf.nn.crelu
            }
        instance.activations = [activation_map[name] for name in activations]
        instance.activation_names = activations

        return instance
