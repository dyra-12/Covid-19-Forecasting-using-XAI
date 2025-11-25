
"""
SHAP utilities compatible with sequence and flat inputs.
"""
import shap
import numpy as np
import matplotlib.pyplot as plt

def _flatten_seq(X):
    if X.ndim == 3:
        n, t, f = X.shape
        X_flat = X.reshape(n, t * f)
        names = [f"f{name}_t-{tidx}" for tidx in range(t-1, -1, -1) for name in range(f)]
        return X_flat, names
    return X, [f"Feature_{i}" for i in range(X.shape[1])]

def explain_model(model, X, feature_names=None, max_background: int = 100, title: str = "SHAP Summary"):
    X_flat, auto_names = _flatten_seq(np.asarray(X))
    if feature_names is None:
        feature_names = auto_names
    # Ensure small background for performance
    background = X_flat[:max_background]

    def model_wrapper(x):
        # reshape back if needed
        if X.ndim == 3:
            t = X.shape[1]
            f = X.shape[2]
            x = x.reshape(-1, t, f)
        return model.predict(x)

    explainer = shap.Explainer(model_wrapper, background)
    shap_values = explainer(X_flat[:max_background])

    plt.figure(figsize=(14, 8))
    shap.summary_plot(shap_values, X_flat[:max_background], feature_names=feature_names, show=False)
    plt.title(title)
    plt.tight_layout()
    plt.show()
