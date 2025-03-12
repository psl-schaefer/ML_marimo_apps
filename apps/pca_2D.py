import marimo

__generated_with = "0.11.17"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt

    plt.style.use("dark_background")
    return mo, np, plt


@app.cell
def _(np):
    np.random.seed(42)

    N = 100

    mean = [0, 0]        # Mean vector
    std_dev = [1, 1]     # Standard deviations: [sigma_1, sigma_2]
    rho = 0.8          # Correlation coefficient

    # Construct covariance matrix
    cov = [[std_dev[0]**2, rho * std_dev[0] * std_dev[1]], 
           [rho * std_dev[0] * std_dev[1], std_dev[1]**2]]

    X = np.random.multivariate_normal(mean, cov, size=N)

    X_center = X - X.mean(axis=0, keepdims=True)
    assert np.all(np.isclose(X_center.mean(axis=0), 0))

    S = 1/N * np.dot(X.T, X)
    eigenval, eigenvec = np.linalg.eigh(S) # since S is symmetric
    sorted_indices = np.argsort(eigenval*-1)
    eigenval = eigenval[sorted_indices]
    eigenvec = eigenvec[:, sorted_indices]
    eigenvec *= -1
    first_eigenvec, second_eigenvec = eigenvec[:, 0], eigenvec[:, 1]
    return (
        N,
        S,
        X,
        X_center,
        cov,
        eigenval,
        eigenvec,
        first_eigenvec,
        mean,
        rho,
        second_eigenvec,
        sorted_indices,
        std_dev,
    )


@app.cell
def _(mo):
    theta_slider = mo.ui.slider(0, 180, value=90, label="Rotation Angle (θ in degrees)")
    eigen_checkbox = mo.ui.checkbox(value=False, label="Display Eigenvectors")
    projection_checkbox = mo.ui.checkbox(value=True, label="Display Projection")
    mo.hstack([theta_slider, eigen_checkbox, projection_checkbox], justify="start", align="start")
    return eigen_checkbox, projection_checkbox, theta_slider


@app.cell
def _(X, np, theta_slider):
    theta = np.radians(theta_slider.value)
    b = np.array([np.cos(theta), np.sin(theta)])
    assert np.isclose(np.dot(b,b), 1)
    Z = np.dot(X, b)
    X_hat = np.outer(Z, b)
    X_res = X - X_hat
    V = np.round(np.dot(Z.T, Z), 2)
    SSE = np.round(np.sum((X - X_hat)**2))
    return SSE, V, X_hat, X_res, Z, b, theta


@app.cell
def _(
    N,
    SSE,
    V,
    X,
    X_hat,
    X_res,
    Z,
    b,
    eigen_checkbox,
    eigenval,
    first_eigenvec,
    np,
    plt,
    projection_checkbox,
    second_eigenvec,
):
    # Plot the samples
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    delta = 0.5
    x_lim = (-3.5, 3.5)
    y_lim = (-3.5, 3.5)

    axes[0].scatter(X[:, 0], X[:, 1], alpha=0.5)
    axes[0].set_xlabel("X1")
    axes[0].set_ylabel("X2")
    axes[0].set_xlim(x_lim)
    axes[0].set_ylim(y_lim)
    axes[0].set_title("Original Data $X$")

    axes[0].axline((0, 0), (b[0], b[1]), color='red', linestyle='--', alpha=0.5, 
                   label="Direction Line")
    axes[0].quiver(0, 0, b[0]*2, b[1]*2,  angles='xy', scale_units='xy', width=0.015,
                   scale=1, color='red', label="Rotating Vector")

    if eigen_checkbox.value:
        axes[0].quiver(0, 0, first_eigenvec[0]*eigenval[0], first_eigenvec[1]*eigenval[0], width=0.018,
                       angles='xy', scale_units='xy', scale=0.3, color='purple', label="First Eigenvector")
        axes[0].quiver(0, 0, second_eigenvec[0]*eigenval[1], second_eigenvec[1]*eigenval[1], width=0.018,
                       angles='xy', scale_units='xy', scale=0.3, color='purple', label="Second Eigenvector")

    axes[1].scatter(Z, np.repeat(0, N), alpha=0.5)
    axes[1].set_title(f"Compressed Data $Z = X b$ | V= {V}")
    axes[1].set_ylim((-1e-1, 1e-1))
    axes[1].set_xlim((-4, 4))
    axes[1].set_yticks([0]) 
    axes[1].set_yticklabels([]) 
    axes[1].set_xlabel("PC 1")

    axes[2].scatter(X_hat[:, 0], X_hat[:, 1], alpha=0.50)
    axes[2].set_xlabel("X1")
    axes[2].set_ylabel("X2")
    axes[2].set_xlim(x_lim)
    axes[2].set_ylim(y_lim)
    X_hat_string = "$\hat{X} = Z b^T$"
    axes[2].set_title(rf"Reconstructed Data {X_hat_string} | SSE = {SSE}")

    axes[3].scatter(X_res[:, 0], X_res[:, 1], alpha=0.50)
    axes[3].set_xlabel("X1")
    axes[3].set_ylabel("X2")
    axes[3].set_xlim(x_lim)
    axes[3].set_ylim(y_lim)
    axes[3].set_title("Residual Data $X - \hat{X}$")

    if projection_checkbox.value:
        for i in range(len(X)):
            axes[0].plot([X[i, 0], X_hat[i, 0]], [X[i, 1], X_hat[i, 1]], alpha=0.5, linewidth=1.5, c="grey")

    fig
    return X_hat_string, axes, delta, fig, i, x_lim, y_lim


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
