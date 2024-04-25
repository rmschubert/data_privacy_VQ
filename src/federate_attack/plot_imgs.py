import os

import matplotlib.pyplot as plt
import pandas as pd


def create_visible_results(res_path):
    results = pd.read_pickle(res_path)
    results_path = os.path.split(res_path)[0]

    ## make results for assumed p^+ and p^-
    mses_p = []
    mses_m = []
    ssims_p = []
    ssims_m = []
    for i, res in enumerate(results["Approximations"]):
        for j, app in enumerate(res):
            sign, beta = app[1], app[2]
            if sign * beta > 0 and sign < 0:
                ## guessing p^+
                mses_p.append(results["MSE"][i][j])
                ssims_p.append(results["SSIM"][i][j])
            elif sign * beta < 0 and sign < 0:
                ## guessing p^-
                mses_m.append(results["MSE"][i][j])
                ssims_m.append(results["SSIM"][i][j])
            ## Note: one could here also distinguish
            ## when the guess fails, i.e. s * b < 0 and s > 0
            ## or s * b > 0 and s > 0 (wrong update scheme)
    plt.rcParams["font.weight"] = "bold"
    f, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].plot(mses_p, label=r"guessed $p^+$")
    axs[0].plot(mses_m, label=r"guessed $p^-$")
    axs[0].set_title("MSE", fontweight="bold")
    axs[0].set_xlabel("Epochs", fontweight="bold")
    axs[0].set_ylabel("MSE (lower is better)", fontweight="bold")
    axs[1].plot(ssims_p, label=r"guessed $p^+$")
    axs[1].plot(ssims_m, label=r"guessed $p^-$")
    axs[1].set_title("SSIM", fontweight="bold")
    axs[1].set_xlabel("Epochs", fontweight="bold")
    axs[1].set_ylabel("SSIM (higher is better)", fontweight="bold")
    axs[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, "plots.pdf"), dpi=400)
    plt.close()

    cherry_pick = [
        (i, r) for i, r in enumerate(results["SSIM"]) if r[0] > 0.95 or r[1] > 0.95
    ][-1]
    real_img = results["Images"][cherry_pick[0]]
    rec_img1 = (
        results["Approximations"][cherry_pick[0]][0][-1].reshape((28, 28)).clip(0, 1)
    )
    rec_img2 = (
        results["Approximations"][cherry_pick[0]][1][-1].reshape((28, 28)).clip(0, 1)
    )
    mse1 = results["MSE"][cherry_pick[0]][0]
    mse2 = results["MSE"][cherry_pick[0]][1]
    ssim1 = results["SSIM"][cherry_pick[0]][0]
    ssim2 = results["SSIM"][cherry_pick[0]][1]

    bad_pick = [
        (i, r)
        for i, r in enumerate(results["SSIM"])
        if (r[0] < 0.15 or r[1] < 0.15) and i < 20
    ][-1]

    breal_img = results["Images"][bad_pick[0]]
    brec_img1 = (
        results["Approximations"][bad_pick[0]][0][-1].reshape((28, 28)).clip(0, 1)
    )
    brec_img2 = (
        results["Approximations"][bad_pick[0]][1][-1].reshape((28, 28)).clip(0, 1)
    )
    bmse1 = results["MSE"][bad_pick[0]][0]
    bmse2 = results["MSE"][bad_pick[0]][1]
    bssim1 = results["SSIM"][bad_pick[0]][0]
    bssim2 = results["SSIM"][bad_pick[0]][1]

    f, axs = plt.subplots(2, 3, figsize=(20, 6))
    axs[0][0].imshow(breal_img, cmap="gray")
    axs[0][0].set_title(
        f"Real Image Epoch {bad_pick[0]}", fontsize=20, fontweight="bold"
    )
    axs[0][0].axis("off")
    axs[0][1].imshow(brec_img1, cmap="gray")
    axs[0][1].set_title(
        f"MSE: {bmse1:.4f} and SSIM: {bssim1:.4f}", fontsize=20, fontweight="bold"
    )
    axs[0][1].axis("off")
    axs[0][2].imshow(brec_img2, cmap="gray")
    axs[0][2].set_title(
        f"MSE: {bmse2:.4f} and SSIM: {bssim2:.4f}", fontsize=20, fontweight="bold"
    )
    axs[0][2].axis("off")

    axs[1][0].imshow(real_img, cmap="gray")
    axs[1][0].set_title(
        f"Real Image Epoch {cherry_pick[0]}", fontsize=20, fontweight="bold"
    )
    axs[1][0].axis("off")
    axs[1][1].imshow(rec_img1, cmap="gray")
    axs[1][1].set_title(
        f"MSE: {mse1:.4f} and SSIM: {ssim1:.4f}", fontsize=20, fontweight="bold"
    )
    axs[1][1].axis("off")
    axs[1][2].imshow(rec_img2, cmap="gray")
    axs[1][2].set_title(
        f"MSE: {mse2:.4f} and SSIM: {ssim2:.4f}", fontsize=20, fontweight="bold"
    )
    axs[1][2].axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, "cherry_pick.pdf"), dpi=400)
    plt.close()
