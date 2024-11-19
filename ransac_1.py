import numpy as np
from .utils1 import *
import time

def compute_triangulation_angles_and_depth(x1, x2, cam_pose):
    """
    Compute triangulation angles between rays and depth values from two spherical cameras.
    
    Args:
        x1: 3xN array of normalized bearing vectors from first camera
        x2: 3xN array of normalized bearing vectors from second camera
        cam_pose: 4x4 transformation matrix from camera 1 to camera 2
    
    Returns:
        angles: Array of acute triangulation angles in degrees
        depths: Array of depth values for points from first camera
    """
    # Get camera centers
    C1 = np.array([0, 0, 0])
    C2 = cam_pose[0:3, 3]
    baseline = C2 / np.linalg.norm(C2)
    
    # Triangulate points to get depths
    solver = EightPointAlgorithmGeneralGeometry()
    points_3d = solver.triangulate_points_from_cam_pose(cam_pose, x1, x2)
    depths = np.linalg.norm(points_3d[0:3, :], axis=0)
    
    # Rotate second camera rays to world frame
    x2_world = (cam_pose[0:3, 0:3] @ x2).T
    x1_world = x1.T
    
    # Compute dot products between corresponding vectors
    dots = np.einsum('ij,ij->i', x1_world, x2_world)
    
    # Get acute angles
    angles = np.arccos(np.clip(dots, -1.0, 1.0))
    obtuse_mask = angles > np.pi/2
    angles[obtuse_mask] = np.pi - angles[obtuse_mask]
    
    # Check angles with baseline
    baseline_dots1 = np.abs(np.einsum('ij,j->i', x1_world, baseline))
    baseline_dots2 = np.abs(np.einsum('ij,j->i', x2_world, baseline))
    
    # Filter points parallel to baseline
    parallel_to_baseline = (baseline_dots1 > 0.95) | (baseline_dots2 > 0.95)
    angles[parallel_to_baseline] = 0
    
    return np.degrees(angles), depths

class EightPointAlgorithmGeneralGeometry:
    def compute_essential_matrix(self, x1, x2):
        A = self.building_matrix_A(x1, x2)
        U, Sigma, V = np.linalg.svd(A)
        E = V[-1].reshape(3, 3)
        U, S, V = np.linalg.svd(E)
        S[2] = 0
        E = np.dot(U, np.dot(np.diag(S), V))
        return E / np.linalg.norm(E)

    @staticmethod
    def building_matrix_A(x1, x2):
        A = np.array([
            x1[0, :] * x2[0, :], x1[0, :] * x2[1, :], x1[0, :] * x2[2, :],
            x1[1, :] * x2[0, :], x1[1, :] * x2[1, :], x1[1, :] * x2[2, :],
            x1[2, :] * x2[0, :], x1[2, :] * x2[1, :], x1[2, :] * x2[2, :]
        ]).T

        return A

    @staticmethod
    def get_the_four_cam_solutions_from_e(E, x1, x2):
        U, S, V = np.linalg.svd(E)
        if np.linalg.det(np.dot(U, V)) < 0:
            V = -V
        W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        t = U[:, 2].reshape(1, -1).T
        transformations = [
            np.vstack((np.hstack((U @ W.T @ V, t)), [0, 0, 0, 1])),
            np.vstack((np.hstack((U @ W.T @ V, -t)), [0, 0, 0, 1])),
            np.vstack((np.hstack((U @ W @ V, t)), [0, 0, 0, 1])),
            np.vstack((np.hstack((U @ W @ V, -t)), [0, 0, 0, 1])),
        ]
        return transformations

    def recover_pose_from_e(self, E, x1, x2):
        transformations = self.get_the_four_cam_solutions_from_e(E, x1, x2)
        return self.select_camera_pose(transformations, x1=x1, x2=x2)

    def select_camera_pose(self, transformations, x1, x2):
        residuals = np.zeros((len(transformations),))
        for idx, M in enumerate(transformations):
            pt1_3d = self.triangulate_points_from_cam_pose(cam_pose=M,
                                                           x1=x1,
                                                           x2=x2)
            pt2_3d = M @ pt1_3d
            x1_hat = spherical_normalization(pt1_3d)
            x2_hat = spherical_normalization(pt2_3d)
            closest_projections_cam2 = (np.sum(x2 * x2_hat, axis=0)) > 0.98
            closest_projections_cam1 = (np.sum(x1 * x1_hat, axis=0)) > 0.98
            residuals[idx] = np.sum(closest_projections_cam1) + np.sum(
                closest_projections_cam2)
        return transformations[residuals.argmax()]

    @staticmethod
    def get_e_from_cam_pose(cam_pose):

        t_x = vector2skew_matrix(cam_pose[0:3, 3] /
                                 np.linalg.norm(cam_pose[0:3, 3]))
        e = t_x.dot(cam_pose[0:3, 0:3])
        return e / np.linalg.norm(e)

    def recover_pose_from_matches(self, x1, x2):
        e = self.compute_essential_matrix(x1, x2)
        return self.recover_pose_from_e(e, x1, x2)

    @staticmethod
    def triangulate_points_from_cam_pose(cam_pose, x1, x2):
        assert x1.shape[0] == 3
        assert x1.shape == x2.shape

        cam_pose = np.linalg.inv(cam_pose)
        landmarks_x1 = []
        for p1, p2 in zip(x1.T, x2.T):
            p1x = vector2skew_matrix(p1.ravel())
            p2x = vector2skew_matrix(p2.ravel())

            A = np.vstack(
                (np.dot(p1x,
                        np.eye(4)[0:3, :]), np.dot(p2x, cam_pose[0:3, :])))
            U, D, V = np.linalg.svd(A)
            landmarks_x1.append(V[-1])

        landmarks_x1 = np.asarray(landmarks_x1).T
        landmarks_x1 = landmarks_x1 / landmarks_x1[3, :]
        return landmarks_x1

    @staticmethod
    def projected_error(**kwargs):
        E_dot_x1 = np.matmul(kwargs["e"].T, kwargs["x1"])
        E_dot_x2 = np.matmul(kwargs["e"], kwargs["x2"])
        dst = np.sum(kwargs["x1"] * E_dot_x2, axis=0)
        return dst / (np.linalg.norm(kwargs["x1"]) *
                    np.linalg.norm(E_dot_x2))

    @staticmethod
    def algebraic_error(**kwargs):
        E_dot_x2 = np.matmul(kwargs["e"], kwargs["x2"])
        dst = np.sum(kwargs["x1"] * E_dot_x2, axis=0)
        return dst


def get_ransac_iterations(p_success=0.99,
                          outliers=0.5,
                          min_constraint=8):
    return int(
        np.log(1 - p_success) / np.log(1 - (1 - outliers) ** min_constraint)) + 1


class RANSAC_8PA:

    def __init__(self):
        self.residual_threshold = 5.e-6
        self.probability_success = 0.99
        self.expected_inliers = 0.5
        self.solver = EightPointAlgorithmGeneralGeometry()
        self.max_trials = get_ransac_iterations(
            p_success=self.probability_success,
            outliers=1 - self.expected_inliers,
            min_constraint=8
        )
        self.num_samples = 0
        self.best_model = None
        self.best_evaluation = np.inf
        self.best_inliers = None
        self.best_inliers_num = 0
        self.counter_trials = 0
        self.time_evaluation = np.inf
        self.post_function_evaluation = None
        self.min_super_set = 8
        self.depth_percentile = 90
        self.min_triangulation_angle = 10
    
    def estimate_essential_matrix(self, sample_bearings1, sample_bearings2, function=None):
        if function is None:
            # If no post-evaluation function is provided, compute E directly
            return self.solver.compute_essential_matrix(
                x1=sample_bearings1,
                x2=sample_bearings2
            )
        else:
            # Use the provided function to get camera pose and then compute E
            bearings = dict(
                x1=sample_bearings1,
                x2=sample_bearings2
            )
            return self.solver.get_e_from_cam_pose(function(**bearings))


    # def get_inliers(self,bearings_1, bearings_2,e_hat):
    #     sample_residuals = self.solver.projected_error(
    #                             e=e_hat,
    #                             x1=bearings_1,
    #                             x2=bearings_2)
            
    #     sample_evaluation = np.sum(sample_residuals ** 2)
    #     sample_inliers = np.abs(sample_residuals) < self.residual_threshold
    #     sample_inliers_num = np.sum(sample_inliers)
    #     return sample_inliers


    def run(self, bearings_1, bearings_2):
        assert bearings_1.shape == bearings_2.shape
        assert bearings_1.shape[0] == 3
        self.num_samples = bearings_1.shape[1]

        random_state = np.random.RandomState(1000)
        self.time_evaluation = 0
        aux_time = time.time()
        total_time_residual= 0
        for self.counter_trials in range(self.max_trials):
            t3 = time.time()
            initial_inliers = random_state.choice(self.num_samples, self.min_super_set, replace=False)
            sample_bearings1 = bearings_1[:, initial_inliers]
            sample_bearings2 = bearings_2[:, initial_inliers]

            # Estimation
            e_hat = self.solver.compute_essential_matrix(
                x1=sample_bearings1,
                x2=sample_bearings2,
            )

            # Get camera pose for triangulation angle check
            # cam_pose = self.solver.recover_pose_from_e(
            #     E=e_hat,
            #     x1=bearings_1,
            #     x2=bearings_2
            # )

            # # Compute triangulation angles
            # tri_angles = compute_triangulation_angles(bearings_1, bearings_2, cam_pose)

            # angle_mask = tri_angles >= self.min_triangulation_angle

            # Evaluation
            
            sample_residuals = self.solver.projected_error(
                e=e_hat,
                x1=bearings_1,
                x2=bearings_2
            )
            # Combine geometric error and angle constraints
            sample_inliers = (np.abs(sample_residuals) < self.residual_threshold)
            sample_inliers_num = np.sum(sample_inliers)
            sample_evaluation = np.sum(sample_residuals[sample_inliers] ** 2)

            t4 = time.time()
            total_time_residual += (t4-t3)
            # Loop Control
            lc_1 = sample_inliers_num > self.best_inliers_num
            lc_2 = sample_inliers_num == self.best_inliers_num
            lc_3 = sample_evaluation < self.best_evaluation
            if lc_1 or (lc_2 and lc_3):
                self.best_model = e_hat.copy()
                self.best_inliers_num = sample_inliers_num.copy()
                self.best_evaluation = sample_evaluation.copy()
                self.best_inliers = sample_inliers.copy()

            if self.counter_trials >= self._dynamic_max_trials():
                break

        # Final refinement using only inliers
        # print ("Residual Time: ",total_time_residual)
        if self.best_inliers is not None and np.sum(self.best_inliers) >= 8:
            best_bearings_1 = bearings_1[:, self.best_inliers]
            best_bearings_2 = bearings_2[:, self.best_inliers]
            
            self.best_model = self.estimate_essential_matrix(
                sample_bearings1=best_bearings_1,
                sample_bearings2=best_bearings_2,
                function=self.post_function_evaluation
            )
            
            cam_pose = self.solver.recover_pose_from_e(
                E=self.best_model,
                x1=best_bearings_1,
                x2=best_bearings_2
            )
            
            # Recompute angles and depths for final refinement
            tri_angles, depths = compute_triangulation_angles_and_depth(best_bearings_1, best_bearings_2, cam_pose)
            depth_threshold = np.percentile(depths, self.depth_percentile)
            valid_points = (tri_angles >= self.min_triangulation_angle) & (depths <= depth_threshold)
            
            sample_residuals = self.solver.projected_error(
                e=self.best_model,
                x1=best_bearings_1,
                x2=best_bearings_2
            )
            
            final_inliers = (np.abs(sample_residuals) < self.residual_threshold) & valid_points
            
            # Update inlier mask for all points
            updated_inliers = np.zeros(bearings_1.shape[1], dtype=bool)
            updated_inliers[self.best_inliers] = final_inliers
            
            self.best_inliers = updated_inliers
            self.best_inliers_num = np.sum(final_inliers)
            self.best_evaluation = np.sum(sample_residuals[final_inliers] ** 2)

        self.time_evaluation = time.time() - aux_time
        return self.best_model, self.best_inliers

    def get_inliers(self, bearings_1, bearings_2):
        self.run(bearings_1, bearings_2)
        return self.best_inliers, self.best_inliers_num


    def get_cam_pose(self, bearings_1, bearings_2):
        self.run(
            bearings_1=bearings_1,
            bearings_2=bearings_2
        )
        cam_pose = self.solver.recover_pose_from_e(
            E=self.best_model,
            x1=bearings_1[:, self.best_inliers],
            x2=bearings_2[:, self.best_inliers]
        )
        return cam_pose

    def reset(self):
        self.max_trials = get_ransac_iterations(
            p_success=self.probability_success,
            outliers=1 - self.expected_inliers,
            min_constraint=8
        )
        self.num_samples = 0
        # self.best_model = None
        self.best_evaluation = np.inf
        # self.best_inliers = None
        self.best_inliers_num = 0
        self.counter_trials = 0
        self.time_evaluation = np.inf
        self.min_super_set = 8
        
    def _dynamic_max_trials(self):
        if self.best_inliers_num == 0:
            return np.inf

        nom = 1 - self.probability_success
        if nom == 0:
            return np.inf

        inlier_ratio = self.best_inliers_num / float(self.num_samples)
        denom = 1 - inlier_ratio ** 8
        if denom == 0:
            return 1
        elif denom == 1:
            return np.inf
        # print ("nom: ",nom)
        nom = np.log(nom)
        # print ("denom: ",denom)
        denom = np.log(denom)
        if denom == 0:
            return 0
        try:
            return int(np.ceil(nom / denom))
        except:
            return np.inf