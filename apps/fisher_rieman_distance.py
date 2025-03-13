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
def _():
    def matprint(mat, fmt="g"):
        col_maxes = [max([len(("{:"+fmt+"}").format(x)) for x in col]) for col in mat.T]
        for x in mat:
            for i, y in enumerate(x):
                print(("{:"+str(col_maxes[i])+fmt+"}").format(y), end="  ")
            print("")
    return (matprint,)


@app.cell
def _(mo):
    mo.md("""### Fisher Information""")
    return


@app.cell
def _(mo):
    mo.md("""#### Multivariate Normal with known covariance, unknown mean""")
    return


@app.cell
def _(mo):
    slider_sigma = mo.ui.slider(0.25, 5, 0.25, 2)
    mo.md(f"Slider Sigma: {slider_sigma}")
    return (slider_sigma,)


@app.cell
def _(mo):
    slider_corr = mo.ui.slider(-0.9, 0.9, 0.1, 0)
    mo.md(f"Slider Correlation: {slider_corr}")
    return (slider_corr,)


@app.cell
def _(torch):
    def log_lik_grad(x, mu, precision):
        return torch.matmul(precision, (x - mu).T).T
    return (log_lik_grad,)


@app.cell
def _(slider_corr, slider_sigma, torch):
    N = 40
    true_mu = torch.tensor([5., 5.])
    s_diag = torch.diag(torch.tensor((slider_sigma.value, slider_sigma.value)))
    rho_mtx = torch.tensor([[1., slider_corr.value], [slider_corr.value, 1]])
    sigma = s_diag @ rho_mtx @ s_diag
    prec = torch.inverse(sigma)
    return N, prec, rho_mtx, s_diag, sigma, true_mu


@app.cell
def _(N, dist, log_lik_grad, prec, sigma, torch, true_mu):
    mvn = dist.MultivariateNormal(true_mu, sigma)
    torch.manual_seed(42)
    samples = mvn.sample((N,))
    log_like_contour = -10.

    # create a 2D grid of mus to test where I want to evaluate each combination of mus in the range of -6 to 16
    mus_tested = torch.linspace(-6, 16, 20)
    mus_tested_1, mus_tested_2 = torch.meshgrid(mus_tested, mus_tested)
    mus_tested_1 = mus_tested_1.reshape(-1)
    mus_tested_2 = mus_tested_2.reshape(-1)

    log_probs = torch.cat([dist.MultivariateNormal(torch.tensor([mus_tested_1[i].item(), mus_tested_2[i].item()]), sigma).log_prob(samples) 
                           for i in range(len(mus_tested_1))]).reshape(-1, N).numpy()

    grads_at_true = log_lik_grad(x=samples, mu=true_mu, precision=prec).numpy()

    empirical_grads = log_lik_grad(x=mvn.sample((1000, )), mu=true_mu, precision=prec)
    empirical_F = 1/1000 * empirical_grads.T@empirical_grads
    empirical_F_inverse = torch.inverse(empirical_F)
    F_probs = dist.MultivariateNormal(true_mu, empirical_F_inverse).log_prob(torch.cat((mus_tested_1, mus_tested_2), dim=0).reshape(2, -1).T)
    return (
        F_probs,
        empirical_F,
        empirical_F_inverse,
        empirical_grads,
        grads_at_true,
        log_like_contour,
        log_probs,
        mus_tested,
        mus_tested_1,
        mus_tested_2,
        mvn,
        samples,
    )


@app.cell
def _(mo, sigma):
    mo.md(
        f"""
        Sigma:\n
        | {sigma[0, 0].item():.2f} | {sigma[0, 1].item():.2f} | \n
        | {sigma[1, 0].item():.2f} | {sigma[1, 1].item():.2f} | \n
        """
    )
    return


@app.cell
def _(empirical_F, mo):
    mo.md(
        f"""
        Fisher Information:\n
        | {empirical_F[0, 0].item():.3f} | {empirical_F[0, 1].item():.3f} | \n
        | {empirical_F[1, 0].item():.3f} | {empirical_F[1, 1].item():.3f} | \n
        """
    )
    return


@app.cell
def _(
    F_probs,
    empirical_F,
    mus_tested,
    mus_tested_1,
    mus_tested_2,
    plt,
    torch,
):
    fig_2, ax_2 = plt.subplots(1, 3, figsize=(15, 5))

    XX, YY, ZZ = mus_tested_1.reshape((mus_tested.shape[0], mus_tested.shape[0])).numpy(), mus_tested_2.reshape((mus_tested.shape[0], mus_tested.shape[0])).numpy(), F_probs.reshape((mus_tested.shape[0], mus_tested.shape[0]))

    for p_idx in range(3):
        ax_2[p_idx].contour(XX, YY, ZZ, levels=20, alpha=0.5)

    p_1, p_2, p_3, p_4 = torch.tensor([0., 0.]), torch.tensor([10., 10.]), torch.tensor([10., 0.]), torch.tensor([0., 10.])

    # compute euclidean distance between p_1 and p_2
    p_1_p_2_dist = torch.sqrt(torch.sum((p_1 - p_2)**2))
    p_1_p_3_dist = torch.sqrt(torch.sum((p_3 - p_4)**2))

    ax_2[0].scatter(p_1[0], p_1[1], marker='x', color='red', s=100)
    ax_2[0].scatter(p_2[0], p_2[1], marker='x', color='red', s=100)
    ax_2[1].scatter(p_3[0], p_3[1], marker='x', color='blue', s=100)
    ax_2[1].scatter(p_4[0], p_4[1], marker='x', color='blue', s=100)
    ax_2[2].scatter(p_1[0], p_1[1], marker='x', color='green', s=100)
    ax_2[2].scatter(p_3[0], p_3[1], marker='x', color='green', s=100)

    ax_2[0].set_title(f"Euclidean Distance: {torch.sqrt(torch.sum((p_1 - p_2)**2)):.2f}\nFisher Information Distance: {torch.sqrt((p_1 - p_2)@empirical_F@(p_1 - p_2)):.2f}")
    ax_2[0].set_xlabel("$\mu_1$")
    ax_2[0].set_ylabel("$\mu_2$")

    ax_2[1].set_title(f"Euclidean Distance: {torch.sqrt(torch.sum((p_3 - p_4)**2)):.2f}\nFisher Information Distance: {torch.sqrt((p_3 - p_4)@empirical_F@(p_3 - p_4)):.2f}")
    ax_2[1].set_xlabel("$\mu_1$")
    ax_2[1].set_ylabel("$\mu_2$")

    ax_2[2].set_title(f"Euclidean Distance: {torch.sqrt(torch.sum((p_1 - p_3)**2)):.2f}\nFisher Information Distance: {torch.sqrt((p_1 - p_3)@empirical_F@(p_1 - p_3)):.2f}")
    ax_2[2].set_xlabel("$\mu_1$")
    ax_2[2].set_ylabel("$\mu_2$")

    plt.gca()
    return (
        XX,
        YY,
        ZZ,
        ax_2,
        fig_2,
        p_1,
        p_1_p_2_dist,
        p_1_p_3_dist,
        p_2,
        p_3,
        p_4,
        p_idx,
    )


@app.cell
def _(
    N,
    XX,
    YY,
    ZZ,
    empirical_F,
    grads_at_true,
    log_like_contour,
    log_probs,
    mus_tested,
    mus_tested_1,
    mus_tested_2,
    np,
    plt,
    samples,
    true_mu,
):
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    ax[0].axvline(true_mu[0], color='white', linestyle='--', alpha=0.3)
    ax[0].axhline(true_mu[1], color='white', linestyle='--', alpha=0.3)
    ax[0].scatter(samples[:, 0], samples[:, 1], alpha=0.2)
    ax[0].set_title(f"Contours represent\nLog Likelihood, i.e. $\log p(x_i | \mu, \sigma)=${log_like_contour}")
    ax[0].set_xlabel("$\mu_1$")
    ax[0].set_ylabel("$\mu_2$")

    for i in range(N):
        X, Y, Z = mus_tested_1.reshape((mus_tested.shape[0], mus_tested.shape[0])).numpy(), mus_tested_2.reshape((mus_tested.shape[0], mus_tested.shape[0])).numpy(), log_probs[:, i].reshape((mus_tested.shape[0], mus_tested.shape[0]))
        ax[0].contour(X, Y, Z, levels=np.array([log_like_contour]), alpha=0.2, cmap="Blues")

    ax[1].quiver(np.repeat(true_mu[0], N), np.repeat(true_mu[1], N), grads_at_true[:, 0], grads_at_true[:, 1], 
              angles='xy', scale_units='xy', scale=10, color='white')
    ax[1].scatter(true_mu[0], true_mu[1], marker='x', color='red', s=100)
    ax[1].set_xlim(4.8, 5.2)
    ax[1].set_ylim(4.8, 5.2)
    ax[1].set_title("Scores (Log Likelihood), i.e. $\\nabla_x \log p(x_i | \mu, \sigma)$\nGradients evaluated at $\mu^*$")
    ax[1].set_xlabel("$\mu_1$")
    ax[1].set_ylabel("$\mu_2$")

    ax[2].contour(XX, YY, ZZ, levels=20, alpha=0.5)

    eig_vals, eig_vecs = np.linalg.eig(empirical_F)
    for i in range(2):
        ax[2].arrow(true_mu[0], true_mu[1], eig_vals[i]*eig_vecs[0, i]*2, eig_vals[i]*eig_vecs[1, i]*2, head_width=0.5, head_length=0.5)

    ax[2].set_title("Fisher Information Matrix")
    ax[2].set_xlabel("$\mu_1$")
    ax[2].set_ylabel("$\mu_2$")

    plt.gca()
    return X, Y, Z, ax, eig_vals, eig_vecs, fig, i


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
