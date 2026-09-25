import numpy as np
import tensorflow as tf

from train_model import (
    ResNetPreprocess,
    XceptionPreprocess,
    compute_class_weights,
    to_class_indices,
)


def test_to_class_indices_one_hot_and_sparse():
    one_hot = np.array([[1, 0], [0, 1], [1, 0]])
    sparse = np.array([0, 1, 0])
    assert list(to_class_indices(one_hot)) == [0, 1, 0]
    assert list(to_class_indices(sparse)) == [0, 1, 0]


def test_class_weights_balance_the_smaller_class():
    images = tf.zeros((4, 8, 8, 3))
    labels = tf.constant([[1, 0], [1, 0], [1, 0], [0, 1]])
    dataset = tf.data.Dataset.from_tensor_slices((images, labels)).batch(2)
    weights = compute_class_weights(dataset, class_count=2)
    assert weights[1] > weights[0]


def test_backbones_do_not_share_preprocessing():
    image = tf.constant(np.full((1, 4, 4, 3), 128.0, dtype=np.float32))
    xception = XceptionPreprocess()(image)
    resnet = ResNetPreprocess()(image)
    assert not np.allclose(xception.numpy(), resnet.numpy())


if __name__ == "__main__":
    test_to_class_indices_one_hot_and_sparse()
    test_class_weights_balance_the_smaller_class()
    test_backbones_do_not_share_preprocessing()
    print("ok")
