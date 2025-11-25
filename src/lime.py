
"""
LIME utilities (tabular). This expects flat inputs; for sequence inputs, it will flatten (time*features).
"""
import numpy as np
try:
    from lime import lime_tabular
except Exception as e:
    lime_tabular = None

def explain_instance(model, X, feature_names=None, class_names=None, num_features: int = 10, sample_index: int = 0):
    if lime_tabular is None:
        raise ImportError("LIME is not installed. Please `pip install lime`.")
    X = np.asarray(X)
    if X.ndim == 3:
        n, t, f = X.shape
        X_flat = X.reshape(n, t * f)
        if feature_names is None:
            feature_names = [f"f{name}_t-{tidx}" for tidx in range(t-1, -1, -1) for name in range(f)]
    else:
        X_flat = X
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(X_flat.shape[1])]

    explainer = lime_tabular.LimeTabularExplainer(
        X_flat, feature_names=feature_names, class_names=class_names, mode='regression'
    )
    def predict_fn(x):
        if X.ndim == 3:
            t = X.shape[1]; f = X.shape[2]
            x = x.reshape(-1, t, f)
        return model.predict(x).reshape(-1)

    exp = explainer.explain_instance(X_flat[sample_index], predict_fn, num_features=num_features)
    return exp
