import marimo

__generated_with = "0.11.17"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import torch
    import torch.distributions as dist
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.style.use('dark_background')
    return dist, mo, np, plt, sns, torch


@app.cell
def _(mo):
    mo.md("""### Fisher Information""")
    return


@app.cell
def _(torch):
    true_mu = 5
    mus_tested = torch.linspace(-6, 16, 100)

    log_lik_grad = lambda x, mu, sigma: (x - mu) / sigma**2
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
    dist,
    log_lik_grad,
    mus_tested,
    plt,
    sns,
    torch,
    true_mu,
):
    sigma = S_slider.value
    normal = dist.Normal(true_mu, sigma)

    N = N_number.value
    torch.manual_seed(42)
    samples = normal.sample((MAX_N,))
    samples = samples[0:N]
    log_lik = torch.cat([dist.Normal(mu, sigma).log_prob(samples) for mu in mus_tested]).reshape(-1, N).numpy()
    log_lik_gradients = torch.cat([log_lik_grad(x=samples, mu=mu, sigma=sigma) for mu in mus_tested]).reshape(-1, N).numpy()
    empirical_F = (log_lik_grad(x=normal.sample((1000, )), mu=true_mu, sigma=sigma)**2).mean().item()
    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    ax[0].hist(samples, bins=100, density=False, range=(-15, 15), alpha=0.5 if N > 1 else 1)
    for i in range(1, 3): ax[i].axvline(true_mu, color='white', linestyle='--')
    ax[2].axhline(0, color='white', linestyle='--')
    for i in range(N):
        ax[1].plot(mus_tested.numpy(), log_lik[:, i], alpha=0.2 if N > 1 else 1)
        ax[2].plot(mus_tested.numpy(), log_lik_gradients[:, i], alpha=0.2 if N > 1 else 1)
    sns.kdeplot(log_lik_grad(x=samples, mu=true_mu, sigma=sigma), ax=ax[3], fill=True)
    for i in range(1, 3): ax[i].set_xlabel('$\mu$')
    ax[0].set_xlabel('$x_i$')
    ax[0].set_ylabel('Count')
    ax[3].set_xlabel("Score: $\\nabla_{\mu} \log p(x_i | \mu, \sigma)$ at $\mu^*$")
    ax[1].set_ylim(-20, 0)
    ax[2].set_ylim(-5, 5)
    ax[3].set_ylim(0, 2)
    for i in range(3): ax[i].set_xlim(-5, 15)
    ax[3].set_xlim(-5, 5)
    ax[1].set_title("Log Likelihood: $\log p(x_i | \mu, \sigma)$")
    ax[2].set_title("Score: $\\nabla_{\mu} \log p(x_i | \mu, \sigma)$")
    ax[3].set_title("Score Distribution at $\mu^*$")
    fig.suptitle(f"$\sigma$={sigma} || Empirical $F=${empirical_F:.3f}")
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
