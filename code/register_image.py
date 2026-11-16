import numpy as np
import cv2

def ssd(I, J):
    """
    Calcule la somme des carrés des différences entre deux images I et J.
    """
    return np.sum((I - J) ** 2)

def translate_image(I, p, q):
    """
    Fait une translation de l'image I de p pixels horizontalement et q pixels verticalement.
    """
    rows, cols = I.shape
    M = np.float32([[1, 0, p], [0, 1, q]])
    print(M.shape)
    translated = cv2.warpAffine(I, M, (cols, rows), flags=cv2.INTER_LINEAR)
    return translated

if __name__ == "__main__":
    I = cv2.imread("./data/I1.png", cv2.IMREAD_GRAYSCALE).astype(float)
    J = cv2.imread("./data/J1.png", cv2.IMREAD_GRAYSCALE).astype(float)

    ssd_value = ssd(I, J)
    print(f"SSD entre I et J: {ssd_value}")
    print(I.shape, J.shape)
    translated_I = translate_image(I, 10, 5)
    cv2.imwrite("./data/I1_translated.png", translated_I)