import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import map_coordinates

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
    translated = cv2.warpAffine(I, M, (cols, rows), flags=cv2.INTER_LINEAR)
    return translated

def rotate_image(I, theta):
    """
    Fait une rotation de l'image I d'un angle theta (en degrés).
    """
    rows, cols = I.shape
    M = cv2.getRotationMatrix2D((0, 0), theta, 1)
    rotated = cv2.warpAffine(I, M, (cols, rows), flags=cv2.INTER_LINEAR)
    return rotated

def register_translation_ssd(I, J, horizontal_range=(-10, 10), vertical_range=(-10, 10)):
    """
    Recale l'image J par rapport à l'image I en utilisant la méthode de translation et le critère SSD.
    """
    best_ssd = float('inf')
    best_translation = (0, 0)
    registery_ssd = []
    
    # Parcourir un certain intervalle de translations possibles
    for p in range(horizontal_range[0], horizontal_range[1] + 1):  # Translation horizontale
        for q in range(vertical_range[0], vertical_range[1] + 1):  # Translation verticale
            translated_I = translate_image(I, p, q)
            current_ssd = ssd(translated_I, J)
            if current_ssd < best_ssd:
                best_ssd = current_ssd
                best_translation = (p, q)
            registery_ssd.append((p, q, current_ssd))
    
    return best_translation, best_ssd, registery_ssd

def register_rotation_ssd(I, J, angle_range=(-180, 180), step=1):
    """
    Recale l'image J par rapport à l'image I en utilisant la méthode de rotation et le critère SSD.
    """
    best_ssd = float('inf')
    best_angle = 0
    registery_ssd = []
    
    # Parcourir un certain intervalle d'angles possibles
    for theta in range(angle_range[0], angle_range[1] + 1, step):
        rotated_I = rotate_image(I, theta)
        current_ssd = ssd(rotated_I, J)
        if current_ssd < best_ssd:
            best_ssd = current_ssd
            best_angle = theta
        registery_ssd.append((theta, current_ssd))
    
    return best_angle, best_ssd, registery_ssd

def show_ssd_graph_3D(registery_ssd):
    """
    Affiche un graphique de la somme des carrés des différences (SSD) en fonction des translations.
    """

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    p_values = [item[0] for item in registery_ssd]
    q_values = [item[1] for item in registery_ssd]
    ssd_values = [item[2] for item in registery_ssd]

    ax.scatter(p_values, q_values, ssd_values)
    ax.set_xlabel('Translation horizontale (p)')
    ax.set_ylabel('Translation verticale (q)')
    ax.set_zlabel('SSD')
    plt.title('Graphique SSD en fonction des translations')
    plt.show()

def show_ssd_graph_2d(registery_ssd):
    """
    Affiche un graphique de la somme des carrés des différences (SSD) en fonction des angles.
    """
    angles = [item[0] for item in registery_ssd]
    ssd_values = [item[1] for item in registery_ssd]

    plt.plot(angles, ssd_values)
    plt.xlabel('Angle (degrés)')
    plt.ylabel('SSD')
    plt.title('Graphique SSD en fonction des angles')
    plt.grid()
    plt.show()

def rotate_translate_image(I, p, q, theta):
    """
    Applique la translation et la rotation en UNE SEULE opération (OpenCV).
    Évite la double interpolation et accélère massivement le calcul.
    """
    rows, cols = I.shape[:2]
    M = cv2.getRotationMatrix2D((0, 0), np.degrees(theta), 1.0)
    
    M[0, 2] += p
    M[1, 2] += q
    
    return cv2.warpAffine(np.float32(I), M, (cols, rows), flags=cv2.INTER_LINEAR)

def register_rigid_ssd(I, J, max_iterations=350, epsilon_t=1e-7, epsilon_r=1e-8):
    """
    Recalage rigide (translation + rotation) minimisant la SSD, 
    basé sur le calcul analytique du gradient de l'image.
    """
    I = np.float32(I)
    J = np.float32(J)

    dI_dy, dI_dx = np.gradient(I)

    rows, cols = I.shape

    x_lin = np.arange(cols)
    y_lin = np.arange(rows)
    x_grid, y_grid = np.meshgrid(x_lin, y_lin)

    # On centre les coordonnées autour de (0,0) pour faire tourner l'image autour de son centre
    x_c = x_grid
    y_c = y_grid
    p, q, theta = 0.0, 0.0, 0.0

    for iteration in range(max_iterations):
        x_rot = x_c * np.cos(theta) - y_c * np.sin(theta)
        y_rot = x_c * np.sin(theta) + y_c * np.cos(theta)
        x_new = x_rot - p
        y_new = y_rot - q

        coords = np.array([y_new.ravel(), x_new.ravel()])

        #interpolation bilinéaire
        I_moved = map_coordinates(I, coords, order=1, mode='constant', cval=0).reshape((rows, cols))
        
        dI_dx_moved = map_coordinates(dI_dx, coords, order=1, mode='constant', cval=0).reshape((rows, cols))
        dI_dy_moved = map_coordinates(dI_dy, coords, order=1, mode='constant', cval=0).reshape((rows, cols))
        diff = I_moved - J

        grad_p = 2 * np.sum(diff * (-dI_dx_moved))
        grad_q = 2 * np.sum(diff * (-dI_dy_moved))

        term_x = dI_dx_moved * (-x_c * np.sin(theta) - y_c * np.cos(theta))
        term_y = dI_dy_moved * (x_c * np.cos(theta) - y_c * np.sin(theta))
        
        grad_theta = 2 * np.sum(diff * (term_x + term_y))
        p -= epsilon_t * grad_p
        q -= epsilon_t * grad_q
        theta -= epsilon_r * grad_theta
        if abs(grad_p) < 1e-3 and abs(grad_q) < 1e-3 and abs(grad_theta) < 1e-3:
            print(f"Convergence atteinte à l'itération {iteration}")
            break

    final_ssd = np.sum((I_moved - J)**2)

    return (p, q, theta), final_ssd


if __name__ == "__main__":

    # # Partie Barbare
    # I = cv2.imread("./data/I1.png", cv2.IMREAD_GRAYSCALE).astype(float)
    # J = cv2.imread("./data/J1.png", cv2.IMREAD_GRAYSCALE).astype(float)

    # translated_I = translate_image(I, 2, 5)
    # cv2.imwrite("./data/I1_translated.png", translated_I)

    # best_translation, best_ssd, registery_ssd = register_translation_ssd(I, translated_I, horizontal_range=(-10, 10), vertical_range=(-10, 10))
    # print(f"Best translation: {best_translation}, Best SSD: {best_ssd}, number of ssd: {len(registery_ssd)}")
    # translated_best_I = translate_image(I, best_translation[0], best_translation[1])
    # cv2.imwrite("./data/I1_best_translated.png", translated_best_I)
    # # show_ssd_graph(registery_ssd)
    # rotated_I = rotate_image(I, 30)
    # cv2.imwrite("./data/I1_rotated.png", rotated_I)
    # best_angle, best_ssd_rot, registery_ssd_rot = register_rotation_ssd(I, rotated_I, angle_range=(-180, 180), step=1)
    # best_rotated_I = rotate_image(I, best_angle)
    # cv2.imwrite("./data/I1_best_rotated.png", best_rotated_I)
    # print(f"Best angle: {best_angle}, Best SSD: {best_ssd_rot}, number of ssd: {len(registery_ssd_rot)}")
    # show_ssd_graph_2d(registery_ssd_rot)

    # Partie Rigid
    Brain = cv2.imread("./data/BrainMRI_1.jpg", cv2.IMREAD_GRAYSCALE)
    Brain_translated = cv2.imread("./data/BrainMRI_2.jpg", cv2.IMREAD_GRAYSCALE)
    Brain_rotated_left = cv2.imread("./data/BrainMRI_3.jpg", cv2.IMREAD_GRAYSCALE)
    Brain_rotated_right = cv2.imread("./data/BrainMRI_4.jpg", cv2.IMREAD_GRAYSCALE)

    rigid_translation, rigid_ssd =register_rigid_ssd(Brain, Brain_rotated_left, max_iterations=5000, epsilon_t=1e-9, epsilon_r=1e-10)
    print(f"Rigid registration (translation): {rigid_translation}, SSD: {rigid_ssd}")
    Best_rigid_translated = rotate_translate_image(Brain, rigid_translation[0], rigid_translation[1], rigid_translation[2])
    cv2.imwrite("./data/Brain_best_translated.png", Best_rigid_translated)
