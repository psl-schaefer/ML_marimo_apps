import marimo

__generated_with = "0.11.17"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import scipy.stats as stats
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.style.use('dark_background')
    return mo, np, plt, sns, stats


@app.cell
def _(mo):
    mo.md("""### Fisher Information""")
    return


@app.cell
def _(np):
    true_mu = 5
    mus_tested = np.linspace(-6, 16, 100)

    def log_lik_grad(x, mu, sigma):
        return (x - mu) / sigma**2
    return log_lik_grad, mus_tested, true_mu


@app.cell
def _(mo):
    mo.md("""#### Univariate Normal with known variance, unknown mean (and true mean=5)""")
    return


@app.cell
def _(mo):
    MAX_N = 400
    N_number = mo.ui.number(1, MAX_N, 1, 20)
    mo.md(f"N: {N_number}")
    return MAX_N, N_number


@app.cell
def _(mo):
    S_slider = mo.ui.slider(0.25, 5, 0.25, 2)
    mo.md(f"Sigma: {S_slider}")
    return (S_slider,)


@app.cell
def _(
    MAX_N,
    N_number,
    S_slider,
    log_lik_grad,
    mus_tested,
    np,
    plt,
    sns,
    stats,
    true_mu,
):
    sigma = S_slider.value
    normal = stats.norm(true_mu, sigma)

    N = N_number.value
    np.random.seed(42)
    samples = normal.rvs(size=MAX_N)
    samples = samples[:N]

    log_lik = np.stack([
        stats.norm(mu, sigma).logpdf(samples) for mu in mus_tested
    ]).T

    log_lik_gradients = np.stack([
        log_lik_grad(samples, mu, sigma) for mu in mus_tested
    ]).T

    empirical_F = np.mean(log_lik_grad(normal.rvs(size=1000), true_mu, sigma)**2)

    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    ax[0].hist(samples, bins=100, density=False, range=(-15, 15), alpha=0.5 if N > 1 else 1)

    for i in range(1, 3): 
        ax[i].axvline(true_mu, color='white', linestyle='--')
    ax[2].axhline(0, color='white', linestyle='--')

    for i in range(N):
        ax[1].plot(mus_tested, log_lik[i, :], alpha=0.2 if N > 1 else 1)
        ax[2].plot(mus_tested, log_lik_gradients[i, :], alpha=0.2 if N > 1 else 1)

    sns.kdeplot(log_lik_grad(samples, true_mu, sigma), ax=ax[3], fill=True)

    for i in range(1, 3):
        ax[i].set_xlabel('$\mu$')
    ax[0].set_xlabel('$x_i$')
    ax[0].set_ylabel('Count')
    ax[3].set_xlabel("Score: $\\nabla_{\mu} \\log p(x_i | \mu, \sigma)$ at $\mu^*$")

    ax[1].set_ylim(-20, 0)
    ax[2].set_ylim(-5, 5)
    ax[3].set_ylim(0, 2)

    for i in range(3):
        ax[i].set_xlim(-5, 15)
    ax[3].set_xlim(-5, 5)

    ax[1].set_title("Log Likelihood: $\\log p(x_i | \mu, \sigma)$")
    ax[2].set_title("Score: $\\nabla_{\mu} \\log p(x_i | \mu, \sigma)$")
    ax[3].set_title("Score Distribution at $\mu^*$")

    fig.suptitle(f"$\\sigma$={sigma} || Empirical $F$={empirical_F:.3f}")
    return (
        N,
        ax,
        empirical_F,
        fig,
        i,
        log_lik,
        log_lik_gradients,
        normal,
        samples,
        sigma,
    )


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


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
