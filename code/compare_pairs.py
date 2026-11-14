import cv2
import numpy as np
from show_joint_hist import joint_hist

DATA_FOLDER = "data"
COUPLES = [
    (f"{DATA_FOLDER}/I1.png", f"{DATA_FOLDER}/J1.png")
] + [
    (f"{DATA_FOLDER}/I{n}.jpg", f"{DATA_FOLDER}/J{n}.jpg") for n in range(2, 7)
]


def ssd(I, J):
    return np.sum((I - J) ** 2)

def cr(I, J):
    mean_I = np.mean(I)
    mean_J = np.mean(J)

    center_I = I - mean_I
    center_J = J - mean_J

    var_I = np.sum(center_I ** 2)
    var_J = np.sum(center_J ** 2)

    return np.sum(center_I * center_J) / np.sqrt(var_I * var_J)

def IM(I, J):
    hist = joint_hist(I, J, 256)

    P_ij = hist / np.sum(hist)
    P_i = np.sum(P_ij, axis=1)
    P_j = np.sum(P_ij, axis=0)

    P_i_P_j = P_i[:, None] * P_j[None, :]
    mask = P_ij > 0 # Pour log
    return np.sum(P_ij[mask] * np.log2(P_ij[mask] / P_i_P_j[mask]))


if __name__ == "__main__":
    for i_path, j_path in COUPLES:
        i = cv2.imread(i_path, cv2.IMREAD_GRAYSCALE).astype(int)
        j = cv2.imread(j_path, cv2.IMREAD_GRAYSCALE).astype(int)

        print(f"{i_path.split('/')[-1]} - {j_path.split('/')[-1]}")

        ssd_ij = ssd(i, j)
        cr_ij = cr(i, j)
        im_ij = IM(i, j)
        print(f"SSD: {ssd_ij}, CR: {cr_ij}, IM: {im_ij}")
