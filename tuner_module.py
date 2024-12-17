
import tensorflow as tf
import keras_tuner as kt
from tensorflow_transform.tf_metadata import schema_utils
from tfx.components.trainer.fn_args_utils import FnArgs
from tensorflow_transform import TFTransformOutput

FEATURE_KEYS = ["AAPL", "MSFT", "AMZN", "BRK_B"]
LABEL_KEY = "IXIC"

def input_fn(file_pattern, tf_transform_output, num_epochs=None, batch_size=32):
    file_pattern = tf.io.gfile.glob(file_pattern)
    feature_spec = tf_transform_output.raw_feature_spec()

    def parse_fn(serialized_example):
        parsed_features = tf.io.parse_single_example(serialized_example, feature_spec)
        features = {key: parsed_features[key] for key in FEATURE_KEYS}
        label = parsed_features[LABEL_KEY]
        return features, label

    dataset = tf.data.TFRecordDataset(file_pattern, compression_type="GZIP")
    dataset = dataset.map(parse_fn)
    dataset = dataset.shuffle(buffer_size=1000).repeat(num_epochs).batch(batch_size)
    return dataset

class TunerFnResult:
    def __init__(self, tuner, fit_kwargs):
        self.tuner = tuner
        self.fit_kwargs = fit_kwargs

def tuner_fn(fn_args: FnArgs):
    tf_transform_output = TFTransformOutput(fn_args.transform_graph_path)

    def build_model(hp):
        inputs = {key: tf.keras.Input(shape=(1,), name=key) for key in FEATURE_KEYS}
        concatenated = tf.keras.layers.Concatenate()(list(inputs.values()))
        x = tf.keras.layers.Dense(
            units=hp.Int("units_1", min_value=64, max_value=256, step=64),
            activation='relu'
        )(concatenated)
        x = tf.keras.layers.Dense(
            units=hp.Int("units_2", min_value=32, max_value=128, step=32),
            activation='relu'
        )(x)
        outputs = tf.keras.layers.Dense(1)(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model

    tuner = kt.RandomSearch(
        hypermodel=build_model,
        objective='val_mae',
        max_trials=50,
        directory=fn_args.working_dir,
        project_name="nasdaq_hyperband"
    )

    train_dataset = input_fn(fn_args.train_files, tf_transform_output)
    validation_dataset = input_fn(fn_args.eval_files, tf_transform_output)

    return TunerFnResult(
        tuner=tuner,
        fit_kwargs={
            "x": train_dataset,
            "validation_data": validation_dataset,
            "steps_per_epoch": fn_args.train_steps,
            "validation_steps": fn_args.eval_steps
        }
    )
