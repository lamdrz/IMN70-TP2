import numpy as np
import matplotlib.pyplot as plt
import argparse

def generate_3D_grid(x_range, y_range, z_range, x_step, y_step, z_step):
    """
    Génère une grille régulière de points 3D dans les plages spécifiées pour chaque axe.

    Parameters:
    x_range (tuple): Un tuple contenant les valeurs minimales et maximales pour l'axe x (min_x, max_x).
    y_range (tuple): Un tuple contenant les valeurs minimales et maximales pour l'axe y (min_y, max_y).
    z_range (tuple): Un tuple contenant les valeurs minimales et maximales pour l'axe z (min_z, max_z).
    x_step (int): Le nombre de points à générer le long de l'axe x.
    y_step (int): Le nombre de points à générer le long de l'axe y.
    z_step (int): Le nombre de points à générer le long de l'axe z.

    Returns:
    np.ndarray: Un tableau numpy représentant les points de la grille 3D.
    """
    grid_points = []

    x = np.linspace(x_range[0], x_range[1], x_step)
    y = np.linspace(y_range[0], y_range[1], y_step)
    z = np.linspace(z_range[0], z_range[1], z_step)
    
    X, Y, Z = np.meshgrid(x, y, z)
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            for k in range(X.shape[2]):
                grid_points.append((X[i, j, k], Y[i, j, k], Z[i, j, k]))
    
    return np.array(grid_points)

def trans_rigide(theta, omega, phi, p, q, r):
    """
    Applies a rigid body transformation to a point in 3D space.

    Parameters:
    theta (float): Rotation angle around the x-axis in radians.
    omega (float): Rotation angle around the y-axis in radians.
    phi (float): Rotation angle around the z-axis in radians.
    p (float): Translation along the x-axis.
    q (float): Translation along the y-axis.
    r (float): Translation along the z-axis.

    Returns:
    function: A function that takes a point (x, y, z) and returns the transformed point.
    """
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(theta), -np.sin(theta)],
                    [0, np.sin(theta), np.cos(theta)]])
    
    R_y = np.array([[np.cos(omega), 0, np.sin(omega)],
                    [0, 1, 0],
                    [-np.sin(omega), 0, np.cos(omega)]])
    
    R_z = np.array([[np.cos(phi), -np.sin(phi), 0],
                    [np.sin(phi), np.cos(phi), 0],
                    [0, 0, 1]])
    
    R = R_z @ R_y @ R_x

    M=np.array([[R[0, 0], R[0, 1], R[0, 2], p],
                [R[1, 0], R[1, 1], R[1, 2], q],
                [R[2, 0], R[2, 1], R[2, 2], r],
                [0, 0, 0, 1]])
    return M

def show_3d_grid(grid_points):
    """
    Affiche une grille 3D de points en utilisant Matplotlib.

    Parameters:
    grid_points (np.ndarray): Un tableau numpy représentant les points de la grille 3D.
    """
    x_points = grid_points[:, 0]
    y_points = grid_points[:, 1]
    z_points = grid_points[:, 2]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x_points, y_points, z_points, c='blue', marker='o', alpha=0.6)

    ax.set_xlabel('Axe X')
    ax.set_ylabel('Axe Y')
    ax.set_zlabel('Axe Z')
    ax.set_title('Grille régulière de points 3D')

    plt.show()

def show_3d_grid_compare(grid_points, transformed_grid_points):
    """
    Affiche deux grilles 3D de points en utilisant Matplotlib. Nous pouvons comparer la grille originale et la grille transformée.

    Parameters:
    grid_points (np.ndarray): Un tableau numpy représentant les points de la grille 3D.
    transformed_grid_points (np.ndarray): Un tableau numpy représentant les points de la grille 3D transformée.
    """
    x_points = grid_points[:, 0]
    y_points = grid_points[:, 1]
    z_points = grid_points[:, 2]

    x_transformed_points = transformed_grid_points[:, 0]
    y_transformed_points = transformed_grid_points[:, 1]
    z_transformed_points = transformed_grid_points[:, 2]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x_points, y_points, z_points, c='blue', marker='o', alpha=0.6, label='grille originale')
    ax.scatter(x_transformed_points, y_transformed_points, z_transformed_points, c='red', marker='x', alpha=0.6, label='grille transformée')

    ax.set_xlabel('Axe X')
    ax.set_ylabel('Axe Y')
    ax.set_zlabel('Axe Z')
    ax.set_title('Grille régulière de points 3D')

    ax.legend()

#On force les axes à avoir la même échelle pour une meilleure visualisation
    all_x = np.concatenate((x_points, x_transformed_points))
    all_y = np.concatenate((y_points, y_transformed_points))
    all_z = np.concatenate((z_points, z_transformed_points))

    max_range = np.array([all_x.max()-all_x.min(), all_y.max()-all_y.min(), all_z.max()-all_z.min()]).max()
    mid_x = (all_x.max()+all_x.min()) * 0.5
    mid_y = (all_y.max()+all_y.min()) * 0.5
    mid_z = (all_z.max()+all_z.min()) * 0.5
    
    ax.set_xlim(mid_x - max_range*0.5, mid_x + max_range*0.5)
    ax.set_ylim(mid_y - max_range*0.5, mid_y + max_range*0.5)
    ax.set_zlim(mid_z - max_range*0.5, mid_z + max_range*0.5)
    ax.set_box_aspect([1, 1, 1])

    plt.show()


def similitude(theta, omega, phi, p, q, r, s):
    """
    Renvoie la matrice 4x4 d'une similitude : 
    Rotations (theta, omega, phi), translation (p, q, r) et homothétie (s).
    """
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta), np.cos(theta)]
    ])
    
    Ry = np.array([
        [np.cos(omega), 0, np.sin(omega)],
        [0, 1, 0],
        [-np.sin(omega), 0, np.cos(omega)]
    ])
    
    Rz = np.array([
        [np.cos(phi), -np.sin(phi), 0],
        [np.sin(phi), np.cos(phi), 0],
        [0, 0, 1]
    ])
    
    #matrice de rotation combinée
    R_combine = Rz @ Ry @ Rx
    
    #matrice homogène 4x4
    M = np.eye(4)
    
    #homothétie s sur la rotation
    M[0:3, 0:3] = s * R_combine
    
    #translation
    M[0:3, 3] = [p, q, r]
    
    return M


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='générer une grille régulière de points 3D et appliquer une transformation rigide ou une similitude.')
    parser.add_argument('theta', type=float, help='Angle de rotation autour de l\'axe x en radians.')
    parser.add_argument('omega', type=float, help='Angle de rotation autour de l\'axe y en radians.')
    parser.add_argument('phi', type=float, help='Angle de rotation autour de l\'axe z en radians.')
    parser.add_argument('p', type=float, help='Translation le long de l\'axe x.')
    parser.add_argument('q', type=float, help='Translation le long de l\'axe y.')
    parser.add_argument('r', type=float, help='Translation le long de l\'axe z.')
    parser.add_argument('s', type=float, help='Facteur d\'échelle pour la transformation de similitude.')
    
    args = parser.parse_args()
    theta = args.theta  # Rotation x
    omega = args.omega  # Rotation y
    phi = args.phi      # Rotation z
    p = args.p          # Translation x
    q = args.q          # Translation y
    r = args.r          # Translation z
    s = args.s          # Scaling factor

  
    grid = generate_3D_grid((-5, 5), (-5, 5), (-5, 5), 10, 10, 10)
    colonne_homogene = np.ones((grid.shape[0], 1))
    grid_homogene = np.hstack((grid, colonne_homogene))

    M = trans_rigide(theta, omega, phi, p, q, r)
    transformed_grid_homogene = grid_homogene @ M.T
    transformed_grid = transformed_grid_homogene[:, :3]

    show_3d_grid_compare(grid, transformed_grid)

    M_similitude = similitude(theta, omega, phi, p, q, r, s)
    transformed_grid_similitude = (grid_homogene @ M_similitude.T)[:, :3]
    show_3d_grid_compare(grid, transformed_grid_similitude)
