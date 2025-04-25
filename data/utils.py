import os
import os.path
import copy
import hashlib
import errno
import numpy as np
from numpy.testing import assert_array_almost_equal

def multiclass_noisify(y, P, random_state=0):
    assert P.shape[0] == P.shape[1]
    assert np.max(y) < P.shape[0]
    assert_array_almost_equal(P.sum(axis=1), np.ones(P.shape[1]))
    assert (P >= 0.0).all()

    m = y.shape[0]
    new_y = y.copy()
    flipper = np.random.RandomState(random_state)

    for idx in np.arange(m):
        i = y[idx]
        flipped = flipper.multinomial(1, P[int(i)], 1)[0]
        new_y[idx] = np.where(flipped == 1)[0][0]

    return new_y

def noisify_pairflip(y_train, noise, random_state=None, nb_classes=10):
    P = np.eye(nb_classes)
    if noise > 0.0:
        P[0, 0], P[0, 1] = 1. - noise, noise
        for i in range(1, nb_classes - 1):
            P[i, i], P[i, i + 1] = 1. - noise, noise
        P[nb_classes - 1, nb_classes - 1], P[nb_classes - 1, 0] = 1. - noise, noise

        y_train_noisy = multiclass_noisify(y_train, P=P, random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()
        print(f'Actual noise {actual_noise:.2f}')
        return y_train_noisy, actual_noise
    return y_train, 0

def noisify_multiclass_symmetric(y_train, noise, random_state=None, nb_classes=10):
    P = np.ones((nb_classes, nb_classes)) * (noise / (nb_classes - 1))
    np.fill_diagonal(P, 1. - noise)

    y_train_noisy = multiclass_noisify(y_train, P=P, random_state=random_state)
    actual_noise = (y_train_noisy != y_train).mean()
    print(f'Actual noise {actual_noise:.2f}')
    return y_train_noisy, actual_noise

def noisify(dataset, train_labels, noise_type, noise_rate, random_state=0, nb_classes=10):
    if noise_type == "pairflip":
        train_noisy_labels, actual_noise_rate = noisify_pairflip(train_labels, noise_rate, random_state=random_state, nb_classes=nb_classes)
    elif noise_type == "symmetric":
        train_noisy_labels, actual_noise_rate = noisify_multiclass_symmetric(train_labels, noise_rate, random_state=random_state, nb_classes=nb_classes)
    elif noise_type == "clean":
        train_noisy_labels = train_labels.copy()
        actual_noise_rate = 0
    else:
        raise ValueError(f"Unsupported noise_type: {noise_type}")
    return train_noisy_labels, actual_noise_rate
