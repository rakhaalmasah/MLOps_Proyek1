
import os
import tensorflow as tf
from tensorflow_transform import TFTransformOutput
from tfx.components.trainer.fn_args_utils import FnArgs
import json

FEATURE_KEYS = ["AAPL", "MSFT", "AMZN", "BRK_B"]
LABEL_KEY = "IXIC"

def input_fn(file_pattern, tf_transform_output, num_epochs=None, batch_size=32):
    file_pattern = tf.io.gfile.glob(file_pattern)
    feature_spec = tf_transform_output.raw_feature_spec()

    def parse_fn(serialized_example):
        parsed_features = tf.io.parse_single_example(serialized_example, feature_spec)
        features = {key: tf.cast(parsed_features[key], tf.float32) for key in FEATURE_KEYS}
        label = tf.cast(parsed_features[LABEL_KEY], tf.float32)
        return features, label

    dataset = tf.data.TFRecordDataset(file_pattern, compression_type="GZIP")
    dataset = dataset.map(parse_fn)
    dataset = dataset.shuffle(buffer_size=1000).repeat(num_epochs).batch(batch_size)
    return dataset

def build_model(hparams):
    inputs = {
        key: tf.keras.Input(shape=(1,), name=key)
        for key in FEATURE_KEYS
    }
    concatenated = tf.keras.layers.Concatenate()(list(inputs.values()))

    x = tf.keras.layers.Dense(hparams["units_1"], activation='relu')(concatenated)
    x = tf.keras.layers.Dense(hparams["units_2"], activation='relu')(x)
    outputs = tf.keras.layers.Dense(1)(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def load_best_hyperparameters(tuner_directory):
    hparams_file = os.path.join(tuner_directory, "best_hyperparameters.txt")
    if not os.path.exists(hparams_file):
        raise FileNotFoundError(f"Hyperparameter tuning file not found at {hparams_file}")

    with open(hparams_file, "r") as f:
        hparams_data = json.load(f)

    hparams = hparams_data["values"]
    return hparams

def run_fn(fn_args):
    tf_transform_output = TFTransformOutput(fn_args.transform_graph_path)

    train_dataset = input_fn(fn_args.train_files, tf_transform_output, num_epochs=fn_args.train_steps)
    eval_dataset = input_fn(fn_args.eval_files, tf_transform_output, num_epochs=1)

    tuner_uri = fn_args.custom_config["tuner_hyperparameters_path"]
    hparams = load_best_hyperparameters(tuner_uri)
    model = build_model(hparams)

    model.fit(
        train_dataset,
        validation_data=eval_dataset,
        steps_per_epoch=fn_args.train_steps,
        validation_steps=fn_args.eval_steps,
        epochs=100
    )

    os.makedirs(fn_args.serving_model_dir, exist_ok=True)

    @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')])
    def serve_tf_examples_fn(serialized_examples):
        raw_feature_spec = tf_transform_output.raw_feature_spec()
        parsed_features = tf.io.parse_example(serialized_examples, raw_feature_spec)

        model_inputs = {key: parsed_features[key] for key in FEATURE_KEYS}
        return model(model_inputs)

    tf.saved_model.save(
        model,
        fn_args.serving_model_dir,
        signatures={'serving_default': serve_tf_examples_fn}
    )
