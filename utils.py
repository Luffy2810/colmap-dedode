import numpy as np
import cv2
import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def visualize_matches(img0, img1, mkpts0, mkpts1, save_name, timetaken, show_keypoints=True, title="Keypoint Matches"):
    """
    Visualize the matches between two images by plotting keypoints and the matches.
    If there are more than 100 matches, show a random 100 of them.

    Parameters:
    - img0: First image (equirectangular or cubemap).
    - img1: Second image (equirectangular or cubemap).
    - mkpts0: Matched keypoints in the first image (Nx2 array).
    - mkpts1: Matched keypoints in the second image (Nx2 array).
    - save_name: Name of the file to save the visualized matches.
    - timetaken: Time taken to compute the matches.
    - title: Title of the plot (default: 'Keypoint Matches').
    - show_keypoints: Whether to display keypoints (default: True).
    """
    # Limit to 100 matches if there are more than 100
    num_matches = len(mkpts0)
    if len(mkpts0) > 250:
        idxs = random.sample(range(len(mkpts0)), 250)
        mkpts0 = mkpts0[idxs]
        mkpts1 = mkpts1[idxs]

    # Convert images to RGB if they are in grayscale
    if len(img0.shape) == 2:  # Grayscale
        img0 = cv2.cvtColor(img0, cv2.COLOR_GRAY2RGB)
    if len(img1.shape) == 2:  # Grayscale
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)

    # Create a figure to display the images and the matches
    combined_height = img0.shape[0] + img1.shape[0]
    fig, ax = plt.subplots(1, 1, figsize=(10, 15))

    # Create a combined image by stacking the two images vertically
    combined_img = np.vstack((img0, img1))
    ax.imshow(combined_img)
    ax.set_title(title)
    ax.axis('off')

    # Set colormap for the matches
    color = cm.jet(np.linspace(0, 1, len(mkpts0)))

    # Extract coordinates of keypoints
    x0, y0 = mkpts0[:, 0], mkpts0[:, 1]  # Points in image 1
    x1, y1 = mkpts1[:, 0], mkpts1[:, 1] + img0.shape[0]  # Points in image 2 (shift y1 by image height of img0)

    # Draw matches (lines connecting keypoints)
    for i in range(len(x0)):
        ax.plot([x0[i], x1[i]], [y0[i], y1[i]], color=color[i], linewidth=1)

    # Optionally, draw the keypoints
    if show_keypoints:
        ax.scatter(x0, y0, 25, marker='o', facecolors='none', edgecolors='r')  # Keypoints in image 1
        ax.scatter(x1, y1, 25, marker='o', facecolors='none', edgecolors='r')  # Keypoints in image 2

    # Add text for time taken and number of matches at the top left corner
    
    ax.text(10, 10, f'Time taken: {timetaken:.2f} sec\nMatches: {num_matches}', 
            fontsize=12, color='white', backgroundcolor='black', verticalalignment='top')

    plt.tight_layout()
    plt.savefig(save_name)
    plt.show()


def generate_mapping_data(image_width):
    in_size = [image_width, int(image_width * 3 / 4)]
    edge = int(in_size[0] / 4)

    out_pix = np.zeros((in_size[1], in_size[0], 2), dtype="f4")
    xyz = np.zeros((in_size[1] * in_size[0] // 2, 3), dtype="f4")
    vals = np.zeros((in_size[1] * in_size[0] // 2, 3), dtype="i4")

    start, end = 0, 0
    rng_1 = np.arange(0, edge * 3)
    rng_2 = np.arange(edge, edge * 2)
    for i in range(in_size[0]):
        face = i // edge
        rng = rng_1 if face == 2 else rng_2

        end += len(rng)
        vals[start:end, 0] = rng
        vals[start:end, 1] = i
        vals[start:end, 2] = face
        start = end

    j, i, face = vals.T
    face[j < edge] = 4
    face[j >= 2 * edge] = 5

    a = 2.0 * i / edge
    b = 2.0 * j / edge
    one_arr = np.ones(len(a))
    for k in range(6):
        face_idx = face == k
        one_arr_idx = one_arr[face_idx]
        a_idx = a[face_idx]
        b_idx = b[face_idx]

        if k == 0:
            vals_to_use = [-one_arr_idx, 1.0 - a_idx, 3.0 - b_idx]
        elif k == 1:
            vals_to_use = [a_idx - 3.0, -one_arr_idx, 3.0 - b_idx]
        elif k == 2:
            vals_to_use = [one_arr_idx, a_idx - 5.0, 3.0 - b_idx]
        elif k == 3:
            vals_to_use = [7.0 - a_idx, one_arr_idx, 3.0 - b_idx]
        elif k == 4:
            vals_to_use = [b_idx - 1.0, a_idx - 5.0, one_arr_idx]
        elif k == 5:
            vals_to_use = [5.0 - b_idx, a_idx - 5.0, -one_arr_idx]

        xyz[face_idx] = np.array(vals_to_use).T

    x, y, z = xyz.T
    theta = np.arctan2(y, x)
    r = np.sqrt(x**2 + y**2)
    phi = np.arctan2(z, r)

    uf = (2.0 * edge * (theta + np.pi) / np.pi) % in_size[0]
    uf[uf == in_size[0]] = 0.0
    vf = (2.0 * edge * (np.pi / 2 - phi) / np.pi)

    out_pix[j, i, 0] = vf
    out_pix[j, i, 1] = uf

    map_x_32 = out_pix[:, :, 1]
    map_y_32 = out_pix[:, :, 0]
    return map_x_32, map_y_32

def spherical_to_cubemap(img, map_x_32, map_y_32):
    cubemap = cv2.remap(img, map_x_32, map_y_32, cv2.INTER_LINEAR)
    return cubemap

def extract_face_from_cubemap(cubemap_img, face):
    face_size = cubemap_img.shape[0] // 3

    if face == 0:  # Positive X (right)
        return cubemap_img[face_size:2 * face_size, face_size * 0:face_size * 1]  # Middle-right
    elif face == 1:  # Negative X (left)
        return cubemap_img[face_size:2 * face_size, face_size * 1:face_size*2]  # Middle-left
    elif face == 2:  # Negative Z (back)
        return cubemap_img[face_size:2 * face_size, face_size * 2:face_size * 3]  # Middle-center
    elif face == 3:  # Positive Z (front)
        return cubemap_img[face_size:2 * face_size, face_size * 3:face_size * 4]  # Middle-second from left
    elif face == 4:  # Positive Y (up)
        return cubemap_img[0:face_size, face_size * 2:face_size * 3]  # Top-center
    elif face == 5:  # Negative Y (down)
        return cubemap_img[2 * face_size:, face_size * 2:face_size * 3]  # Bottom-center

def cubemap_to_equirectangular(points, face, cubemap_size, img_width, img_height):
    points = np.array(points)
    if points.ndim == 1:
        points = points.reshape(1, -1)
    
    x, y = points.T
    x = (x / cubemap_size) * 2 - 1
    y = (y / cubemap_size) * 2 - 1
    
    face_mapping = {
        '0': lambda: np.column_stack((np.ones_like(x), x, -y)),
        '1': lambda: np.column_stack((-x, np.ones_like(x), -y)),
        '2': lambda: np.column_stack((-np.ones_like(x), -x, -y)),
        '3': lambda: np.column_stack((x, -np.ones_like(x), -y)),
        '4': lambda: np.column_stack((y, x, np.ones_like(x))),
        '5': lambda: np.column_stack((-y, x, -np.ones_like(x)))
    }
    
    vec = face_mapping[face]()
    vec = vec / np.linalg.norm(vec, axis=1, keepdims=True)
    theta = np.arctan2(vec[:, 1], vec[:, 0])
    phi = -np.arcsin(vec[:, 2])
    u = (theta + np.pi) / (2 * np.pi)
    v = (phi + np.pi/2) / np.pi
    result = np.column_stack((u * img_width, v * img_height))
    
    return result[0] if len(result) == 1 else result

def convert_cubemap_matches_to_equirectangular( mkpts0, mkpts1, cubemap_size,img_width,img_height):
    edge_size = cubemap_size // 4
    
    faces0 = [get_face(x, y, edge_size) for x, y in mkpts0]
    faces1 = [get_face(x, y, edge_size) for x, y in mkpts1]
    
    local_mkpts0 = np.array([(x % edge_size, y % edge_size) for x, y in mkpts0])
    local_mkpts1 = np.array([(x % edge_size, y % edge_size) for x, y in mkpts1])
    
    eq_mkpts0 = np.array([cubemap_to_equirectangular(pts, face, edge_size,img_width,img_height) 
                            for pts, face in zip(local_mkpts0, faces0)])
    eq_mkpts1 = np.array([cubemap_to_equirectangular(pts, face, edge_size,img_width,img_height) 
                            for pts, face in zip(local_mkpts1, faces1)])
    
    return eq_mkpts0, eq_mkpts1


def get_face(x, y, edge_size):
    face_x = int(x // edge_size)
    face_y = int(y // edge_size)
    if face_y == 0:
        return '4'  # top face
    elif face_y == 2:
        return '5'  # bottom face
    else:
        return str(face_x)  # side faces



import numpy as np

def vectorized_equirectangular_to_cubemap(points, img_width, img_height, cubemap_size):
    """
    Vectorized conversion of points from equirectangular coordinates to cubemap coordinates.
    
    Parameters:
    points (np.ndarray): Array of points in equirectangular coordinates (N x 2)
    img_width (int): Width of the equirectangular image
    img_height (int): Height of the equirectangular image
    cubemap_size (int): Size of one face of the cubemap
    
    Returns:
    tuple: (cubemap_points, faces)
        - cubemap_points: Array of points in cubemap coordinates (N x 2)
        - faces: Array of face indices for each point (N,)
    """
    points = np.asarray(points)
    if points.ndim == 1:
        points = points.reshape(1, -1)
    
    # Convert to spherical coordinates (vectorized)
    theta = (points[:, 0] / img_width) * 2 * np.pi - np.pi
    phi = (points[:, 1] / img_height) * np.pi - np.pi/2
    
    # Convert to 3D cartesian coordinates (vectorized)
    cos_phi = np.cos(phi)
    xyz = np.stack([
        cos_phi * np.cos(theta),  # x
        cos_phi * np.sin(theta),  # y
        np.sin(phi)               # z
    ], axis=-1)
    
    # Find dominant axis and face (vectorized)
    abs_xyz = np.abs(xyz)
    max_indices = np.argmax(abs_xyz, axis=1)
    max_values = np.take_along_axis(xyz, max_indices[:, None], axis=1).squeeze()
    signs = np.sign(max_values)
    
    # Initialize output arrays
    edge_size = cubemap_size // 4
    N = len(points)
    cubemap_points = np.zeros((N, 2))
    faces = np.zeros(N, dtype=str)
    
    # Prepare masks for each face
    masks = {
        '0': (max_indices == 0) & (signs > 0),   # Positive X (right)
        '1': (max_indices == 0) & (signs < 0),   # Negative X (left)
        '2': (max_indices == 2) & (signs < 0),   # Negative Z (back)
        '3': (max_indices == 2) & (signs > 0),   # Positive Z (front)
        '4': (max_indices == 1) & (signs > 0),   # Positive Y (up)
        '5': (max_indices == 1) & (signs < 0),   # Negative Y (down)
    }
    
    # Face coordinate calculations (vectorized for each face)
    face_coords = {
        '0': {'u': -xyz[:, 2] / xyz[:, 0], 'v': -xyz[:, 1] / xyz[:, 0], 'fx': 0, 'fy': 1},
        '1': {'u': xyz[:, 2] / -xyz[:, 0], 'v': -xyz[:, 1] / -xyz[:, 0], 'fx': 1, 'fy': 1},
        '2': {'u': -xyz[:, 0] / -xyz[:, 2], 'v': -xyz[:, 1] / -xyz[:, 2], 'fx': 2, 'fy': 1},
        '3': {'u': xyz[:, 0] / xyz[:, 2], 'v': -xyz[:, 1] / xyz[:, 2], 'fx': 3, 'fy': 1},
        '4': {'u': xyz[:, 0] / xyz[:, 1], 'v': xyz[:, 2] / xyz[:, 1], 'fx': 2, 'fy': 0},
        '5': {'u': xyz[:, 0] / -xyz[:, 1], 'v': -xyz[:, 2] / -xyz[:, 1], 'fx': 2, 'fy': 2}
    }
    
    # Process each face (still vectorized within each face)
    for face, mask in masks.items():
        if not np.any(mask):
            continue
            
        coords = face_coords[face]
        
        # Calculate local coordinates
        u = np.zeros(N)
        v = np.zeros(N)
        u[mask] = coords['u'][mask]
        v[mask] = coords['v'][mask]
        
        # Convert to pixel coordinates
        u = (u + 1) * edge_size / 2
        v = (v + 1) * edge_size / 2
        
        # Clamp values
        u = np.clip(u, 0, edge_size - 1)
        v = np.clip(v, 0, edge_size - 1)
        
        # Set global cubemap coordinates
        cubemap_points[mask, 0] = coords['fx'] * edge_size + u[mask]
        cubemap_points[mask, 1] = coords['fy'] * edge_size + v[mask]
        faces[mask] = face
    
    return cubemap_points, faces

def batch_convert_equirect_to_cubemap(points, img_width, img_height, cubemap_size, batch_size=10000):
    """
    Process large point sets in batches to avoid memory issues.
    
    Parameters:
    points (np.ndarray): Array of points to convert
    img_width (int): Width of the equirectangular image
    img_height (int): Height of the equirectangular image
    cubemap_size (int): Size of one face of the cubemap
    batch_size (int): Number of points to process in each batch
    
    Returns:
    tuple: (cubemap_points, faces)
    """
    points = np.asarray(points)
    total_points = len(points)
    
    # Initialize output arrays
    cubemap_points = np.zeros_like(points)
    faces = np.zeros(total_points, dtype=str)
    
    # Process in batches
    for i in range(0, total_points, batch_size):
        batch_end = min(i + batch_size, total_points)
        batch_points = points[i:batch_end]
        
        # Convert batch
        batch_cubemap_points, batch_faces = vectorized_equirectangular_to_cubemap(
            batch_points, img_width, img_height, cubemap_size)
        
        # Store results
        cubemap_points[i:batch_end] = batch_cubemap_points
        faces[i:batch_end] = batch_faces
    
    return cubemap_points, faces