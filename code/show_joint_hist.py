import cv2
import numpy as np
import matplotlib.pyplot as plt

DATA_FOLDER = "data"
COUPLES = [
    (f"{DATA_FOLDER}/I1.png", f"{DATA_FOLDER}/J1.png")
] + [
    (f"{DATA_FOLDER}/I{n}.jpg", f"{DATA_FOLDER}/J{n}.jpg") for n in range(2, 7)
]

def np_joint_hist(I, J, bin):
    return np.histogram2d(I.flatten(), J.flatten(), bins=bin, range=[[0, 256], [0, 256]])[0].astype(int)

def joint_hist(I,  J,  bin):
    I_bin = np.floor(I / (256 / bin)).astype(int).flatten()
    J_bin = np.floor(J / (256 / bin)).astype(int).flatten()

    hist = np.zeros((bin, bin), dtype=int)
    np.add.at(hist, (I_bin, J_bin), 1)

    return hist

def show_hist_with_margin(hist, bin=256):
    hist_I = np.sum(hist, axis=1)
    hist_J = np.sum(hist, axis=0)

    fig = plt.figure(figsize=(8, 8))
    gs = fig.add_gridspec(2, 3, width_ratios=(1, 4, 0.15), height_ratios=(1, 4), wspace=0.05, hspace=0.05)

    ax_joint  = fig.add_subplot(gs[1, 1])
    ax_marg_I = fig.add_subplot(gs[0, 1], sharex=ax_joint)
    ax_marg_J = fig.add_subplot(gs[1, 0], sharey=ax_joint)
    ax_colorbar = fig.add_subplot(gs[1, 2])

    im = ax_joint.imshow(hist, origin="lower")
    ax_joint.set_xlabel("I")
    ax_marg_J.set_ylabel("J")

    ax_marg_I.bar(range(bin), hist_I)
    ax_marg_J.barh(range(bin), hist_J)
    ax_marg_J.invert_xaxis()

    for ax in (ax_marg_I, ax_marg_J):
        for spine in ax.spines.values():
            spine.set_visible(False)
            
    fig.colorbar(im, cax=ax_colorbar, label="Fréquence")

    plt.show()

def check_sum(I, J, hist):
    hist_sum = np.sum(hist)
    assert I.size == J.size == hist_sum, "Somme hist != n*p"

if __name__ == "__main__":
    for i_path, j_path in COUPLES:
        i = cv2.imread(i_path, cv2.IMREAD_GRAYSCALE).astype(int)
        j = cv2.imread(j_path, cv2.IMREAD_GRAYSCALE).astype(int)

        print(f"{i_path.split('/')[-1]} - {j_path.split('/')[-1]}")

        hist = joint_hist(i, j, 256)
        show_hist_with_margin(hist, 256)
        # show_hist_with_margin(np_joint_hist(i, j, 256), 256) # Pour comparer
        check_sum(i, j, hist)