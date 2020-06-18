"""
Format of entries is `test name: (autolab problem name, score)`
"""
problems = {
    'test_mcq':('Multiple Choice Questions', 5),

    'test_sigmoid_forward': ('Sigmoid Non-Linearity (forward)', 2),
    'test_sigmoid_derivative': ('Sigmoid Non-Linearity (derivative)', 2),
    'test_tanh_forward': ('Tanh Non-Linearity (forward)', 2),
    'test_tanh_derivative': ('Tanh Non-Linearity (derivative)', 2),
    'test_relu_forward': ('ReLU Non-Linearity (forward)', 2),
    'test_relu_derivative': ('ReLU Non-Linearity (derivative)', 2),

    'test_softmax_cross_entropy_forward': ('Softmax Cross Entropy (forward)', 2),
    'test_softmax_cross_entropy_derivative': ('Softmax Cross Entropy (derivative)', 2),

    'test_batch_norm_train': ('Batch Normalization (training time)', 10),
    'test_batch_norm_inference': ('Batch Normalization (inference time)', 5),

    'test_linear_layer_forward': ('Linear Layer (Forward)',2),
    'test_linear_layer_backward': ('Linear Layer (Backward)',2),

    'test_linear_classifier_forward': ('Linear Classifier (forward)', 2),
    'test_linear_classifier_backward': ('Linear Classifier (backward)', 2),
    'test_linear_classifier_step': ('Linear Classifier (step)', 1),

    'test_single_hidden_forward': ('Single Hidden Layer (forward)', 5),
    'test_single_hidden_backward': ('Single Hidden Layer (backward)', 5),
    'test_mystery_hidden_forward1': ('N Hidden Layer (forward) 1', 5),
    'test_mystery_hidden_forward2': ('N Hidden Layer (forward) 2', 5),
    'test_mystery_hidden_forward3': ('N Hidden Layer (forward) 3', 5),
    'test_mystery_hidden_backward1': ('N Hidden Layer (backward) 1', 5),
    'test_mystery_hidden_backward2': ('N Hidden Layer (backward) 2', 5),
    'test_mystery_hidden_backward3': ('N Hidden Layer (backward) 3', 5),

    'test_momentum': ('Momentum', 10),
    # 'test_train_statistics' :('Train statistics', 5)
}