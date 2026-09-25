"""
Higher-accuracy training for the Xception + ResNet50 autism image classifier.

Drop-in replacement for the notebook training cells. Expects tf.data datasets
that yield (image, label). Images should be float32 in [0, 255] (the default
from image_dataset_from_directory). Labels may be one-hot or integer class ids.

Accuracy changes versus the original loop:
- Each backbone gets its own ImageNet preprocessing. Xception wants [-1, 1];
  ResNet50 wants BGR mean subtraction. Sharing one input scaling trains both
  nets on the wrong colors.
- The single-step LSTM is not used. One pooled vector is not a time sequence,
  so the LSTM only added parameters.
- Stage 1 trains the new head with the backbones frozen. Stage 2 unfreezes the
  top of each backbone (BatchNorm stays frozen) at a lower learning rate.
- Augmentation, label smoothing, class weights, early stopping, and learning-rate
  reduction. The saved weights are the best validation epoch, not the last one.
- Hyperparameter search scores the best epoch of each trial, not the final epoch.
"""

import random

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.applications import ResNet50, Xception
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.xception import preprocess_input as xception_preprocess
from tensorflow.keras.layers import (
    BatchNormalization,
    Concatenate,
    Dense,
    Dropout,
    Input,
    Layer,
    RandomBrightness,
    RandomContrast,
    RandomFlip,
    RandomRotation,
    RandomZoom,
)
from tensorflow.keras.models import Model


class XceptionPreprocess(Layer):
    def call(self, inputs):
        return xception_preprocess(inputs)


class ResNetPreprocess(Layer):
    def call(self, inputs):
        return resnet_preprocess(inputs)

OUTPUT_DIR = "training_outputs"


def to_class_indices(labels):
    labels = np.asarray(labels)
    if labels.ndim > 1 and labels.shape[-1] > 1:
        return np.argmax(labels, axis=-1).astype(int)
    return labels.reshape(-1).astype(int)


def compute_class_weights(dataset, class_count=2):
    """Balanced class weights from a tf.data dataset of (image, label)."""
    counts = np.zeros(class_count, dtype=np.float64)
    for _, labels in dataset:
        indices = to_class_indices(labels.numpy())
        for class_id in indices:
            if 0 <= class_id < class_count:
                counts[class_id] += 1
    total = counts.sum()
    if total == 0:
        return {i: 1.0 for i in range(class_count)}
    return {
        i: float(total / (class_count * max(counts[i], 1.0)))
        for i in range(class_count)
    }


def _augmentation():
    return tf.keras.Sequential(
        [
            RandomFlip("horizontal"),
            RandomRotation(0.04),
            RandomZoom(0.1),
            RandomContrast(0.1),
            RandomBrightness(0.1),
        ],
        name="augmentation",
    )


def create_model(
    input_shape=(224, 224, 3),
    class_count=2,
    learning_rate=1e-4,
    dropout_rate=0.3,
    dense_units=256,
    label_smoothing=0.05,
    use_lstm=False,
    weights="imagenet",
):
    """
    Dual-backbone classifier. use_lstm is accepted so older call sites still run;
    a one-step LSTM is not built because it does not improve single-image accuracy.
    """
    del use_lstm  # single images have no time axis to model

    xception = Xception(
        weights=weights, include_top=False, input_shape=input_shape, pooling="avg"
    )
    resnet = ResNet50(
        weights=weights, include_top=False, input_shape=input_shape, pooling="avg"
    )
    xception.trainable = False
    resnet.trainable = False

    inputs = Input(shape=input_shape, name="image")
    augmented = _augmentation()(inputs)
    xception_features = xception(
        XceptionPreprocess(name="xception_preprocess")(augmented),
        training=False,
    )
    resnet_features = resnet(
        ResNetPreprocess(name="resnet_preprocess")(augmented),
        training=False,
    )

    features = Concatenate(name="backbone_features")([xception_features, resnet_features])
    features = BatchNormalization()(features)
    features = Dropout(dropout_rate)(features)
    features = Dense(dense_units, activation="relu")(features)
    features = Dropout(dropout_rate)(features)
    output = Dense(class_count, activation="softmax", name="predictions")(features)

    model = Model(inputs=inputs, outputs=output, name="xception_resnet_autism")
    _compile(model, learning_rate, label_smoothing)
    return model


def _compile(model, learning_rate, label_smoothing):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate, clipnorm=1.0),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing),
        metrics=["accuracy"],
    )


def _set_backbone_trainable(backbone, unfreeze_layers):
    backbone.trainable = True
    frozen_until = max(len(backbone.layers) - unfreeze_layers, 0)
    for layer in backbone.layers[:frozen_until]:
        layer.trainable = False
    for layer in backbone.layers:
        if isinstance(layer, BatchNormalization):
            layer.trainable = False


def unfreeze_for_finetune(model, unfreeze_layers=40, fine_tune_lr=1e-5, label_smoothing=0.05):
    """Unfreeze the top of Xception and ResNet50 and recompile with a smaller LR."""
    named = {layer.name: layer for layer in model.layers}
    for name in ("xception", "resnet50"):
        backbone = named.get(name)
        if backbone is not None:
            _set_backbone_trainable(backbone, unfreeze_layers)
    _compile(model, fine_tune_lr, label_smoothing)
    return model


def _callbacks(model_name, stage):
    import os

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    checkpoint = os.path.join(OUTPUT_DIR, f"{model_name}_{stage}.keras")
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=6,
            restore_best_weights=True,
            mode="max",
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
        ),
    ]


def _best_epoch_accuracy(history):
    values = history.history.get("val_accuracy", [])
    if not values:
        return 0.0
    return float(max(values))


def train_and_evaluate_model(
    model,
    train_data,
    val_data,
    test_data,
    model_name,
    class_names,
    class_weights=None,
    head_epochs=25,
    fine_tune_epochs=20,
    class_count=2,
    label_smoothing=0.05,
):
    if class_weights is None:
        class_weights = compute_class_weights(train_data, class_count=class_count)

    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=head_epochs,
        class_weight=class_weights,
        callbacks=_callbacks(model_name, "head"),
    )
    unfreeze_for_finetune(model, fine_tune_lr=1e-5, label_smoothing=label_smoothing)
    fine_tune_history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=fine_tune_epochs,
        class_weight=class_weights,
        callbacks=_callbacks(model_name, "finetune"),
    )

    test_loss, test_accuracy = model.evaluate(test_data, verbose=0)
    print(f"\n--- Model Evaluation: {model_name} ---")
    print(f"Best head validation accuracy: {_best_epoch_accuracy(history):.4f}")
    print(f"Best fine-tune validation accuracy: {_best_epoch_accuracy(fine_tune_history):.4f}")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")

    y_true, y_pred, y_prob = [], [], []
    for images, labels in test_data:
        probabilities = model.predict(images, verbose=0)
        y_prob.extend(probabilities[:, 1] if probabilities.shape[-1] > 1 else probabilities.reshape(-1))
        y_pred.extend(np.argmax(probabilities, axis=-1))
        y_true.extend(to_class_indices(labels.numpy()))

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=list(class_names)))

    cm = confusion_matrix(y_true, y_pred)
    _save_plots(history, fine_tune_history, cm, class_names, model_name)
    return {
        "history": history,
        "fine_tune_history": fine_tune_history,
        "test_accuracy": float(test_accuracy),
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": np.asarray(y_prob),
    }


def _save_plots(head_history, fine_tune_history, cm, class_names, model_name):
    import os

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    accuracy = head_history.history["accuracy"] + fine_tune_history.history["accuracy"]
    val_accuracy = head_history.history["val_accuracy"] + fine_tune_history.history["val_accuracy"]
    loss = head_history.history["loss"] + fine_tune_history.history["loss"]
    val_loss = head_history.history["val_loss"] + fine_tune_history.history["val_loss"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(accuracy, label="Training Accuracy", marker="o")
    axes[0].plot(val_accuracy, label="Validation Accuracy", marker="s")
    axes[0].axvline(len(head_history.history["accuracy"]) - 0.5, color="gray", linestyle="--", label="Fine-tune start")
    axes[0].set_title(f"Accuracy vs Epochs for {model_name}")
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(loss, label="Training Loss", marker="o", color="r")
    axes[1].plot(val_loss, label="Validation Loss", marker="s", color="g")
    axes[1].set_title(f"Loss vs Epochs for {model_name}")
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, f"{model_name}_curves.png"), dpi=120)
    plt.close(fig)

    fig = plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        cmap="crest",
        annot=True,
        fmt="d",
        xticklabels=list(class_names),
        yticklabels=list(class_names),
    )
    plt.title(f"Confusion Matrix for {model_name}")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, f"{model_name}_confusion.png"), dpi=120)
    plt.close(fig)


def chaos_butterfly_optimization(
    train_data,
    val_data,
    iterations=5,
    proxy_epochs=4,
    class_count=2,
    weights="imagenet",
):
    """
    Short random search. The original loop scored the last epoch, which can be
    worse than an earlier one. Each trial is scored by its best validation accuracy.
    """
    class_weights = compute_class_weights(train_data, class_count=class_count)
    best_accuracy = -1.0
    best_params = {
        "learning_rate": 1e-4,
        "dropout_rate": 0.3,
        "dense_units": 256,
        "label_smoothing": 0.05,
        "use_lstm": False,
    }

    for i in range(iterations):
        learning_rate = 10 ** np.random.uniform(-5, -3)
        dropout_rate = float(np.random.uniform(0.2, 0.5))
        dense_units = int(random.choice([128, 256, 512]))
        label_smoothing = float(random.choice([0.0, 0.05, 0.1]))
        print(
            f"\nIteration {i + 1}: LR={learning_rate:.6f}, Dropout={dropout_rate:.2f}, "
            f"Dense={dense_units}, Smoothing={label_smoothing:.2f}"
        )
        model = create_model(
            learning_rate=learning_rate,
            dropout_rate=dropout_rate,
            dense_units=dense_units,
            label_smoothing=label_smoothing,
            class_count=class_count,
            weights=weights,
        )
        history = model.fit(
            train_data,
            validation_data=val_data,
            epochs=proxy_epochs,
            class_weight=class_weights,
            verbose=0,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_accuracy",
                    patience=2,
                    restore_best_weights=True,
                    mode="max",
                )
            ],
        )
        val_acc = _best_epoch_accuracy(history)
        print(f"Best validation accuracy this trial: {val_acc:.4f}")
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            best_params = {
                "learning_rate": float(learning_rate),
                "dropout_rate": dropout_rate,
                "dense_units": dense_units,
                "label_smoothing": label_smoothing,
                "use_lstm": False,
            }
        tf.keras.backend.clear_session()

    print("\nBest Hyperparameters:", best_params)
    return best_params


def run_colab(train_data, val_data, test_data, class_names, search_iterations=5):
    """Call this from Colab after train_data, val_data, test_data, and class_names exist."""
    best_params = chaos_butterfly_optimization(
        train_data, val_data, iterations=search_iterations
    )
    final_model = create_model(
        learning_rate=best_params["learning_rate"],
        dropout_rate=best_params["dropout_rate"],
        dense_units=best_params["dense_units"],
        label_smoothing=best_params["label_smoothing"],
        use_lstm=False,
    )
    results = train_and_evaluate_model(
        final_model,
        train_data,
        val_data,
        test_data,
        model_name="xception_resnet",
        class_names=class_names,
        label_smoothing=best_params["label_smoothing"],
    )
    final_model.save("autism_eye_model.keras")
    print("Saved autism_eye_model.keras")
    print(f"Test accuracy: {results['test_accuracy']:.4f}")
    return final_model, results
