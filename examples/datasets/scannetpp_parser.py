import os
import copy
import cv2
import torch
import numpy as np
from typing import Dict, assert_never
from pycolmap import SceneManager
from pyquaternion import Quaternion
from .colmap import Parser as BaseParser
from .normalize import (
    align_principal_axes,
    similarity_from_cameras,
    transform_cameras,
    transform_points,
)


class ScannetppParser(BaseParser):

    def __init__(
        self,
        data_dir: str,
        factor: int = 1,
        normalize: bool = False,
        test_every: int = 8,
        use_undistort: bool = False,
        use_roll_shutter: bool = False,
    ) -> None:
        """Initialize the parser for ScanNet++ dataset."""
        self.data_dir = data_dir
        self.factor = factor
        self.normalize = normalize
        self.test_every = test_every
        self.use_undistort = use_undistort
        self.use_roll_shutter = use_roll_shutter

        self.colmap_dir = self.init_colmap_dir()
        self.imgs_dir = self.init_images_dir()
        self.masks_dir = self.init_masks_dir()

        self.manager = SceneManager(self.colmap_dir)
        self.manager.load_cameras()
        self.manager.load_images()
        self.manager.load_points3D()

        if self.use_roll_shutter:
            self.manager_rs = SceneManager(self.colmap_dir)
            self.manager_rs.load_images(os.path.join(self.colmap_dir, "images_rs.txt"))
        else:
            self.manager_rs = None
        
        self.parse()

    def init_colmap_dir(self) -> str:
        """Get COLMAP directory path."""
        return os.path.join(self.data_dir, "colmap")

    def init_images_dir(self) -> str:
        """Get images directory path."""
        imgs_tag = (
            "resized_undistorted_images" if self.use_undistort else "resized_images"
        )
        return os.path.join(
            self.data_dir,
            imgs_tag if self.factor == 1 else f"{imgs_tag}_{self.factor}",
        )

    def init_masks_dir(self) -> str:
        """Get masks directory path."""
        masks_tag = (
            "resized_undistorted_masks" if self.use_undistort else "resized_anon_masks"
        )
        return os.path.join(
            self.data_dir,
            masks_tag if self.factor == 1 else f"{masks_tag}_{self.factor}",
        )

    def parse_cameras(self) -> Dict[int, Dict[str, np.ndarray]]:
        """Parse the dataset and save to COLMAP format."""
        cam_dict = dict()
        for cam_id in self.manager.cameras:
            cam = self.manager.cameras[cam_id]

            # 1. parse scaled camera matrix
            fx, fy, cx, cy = cam.fx, cam.fy, cam.cx, cam.cy
            cam_mat = np.array(
                [
                    [fx, 0, cx],
                    [0, fy, cy],
                    [0, 0, 1],
                ],
                dtype=np.float64,
            )
            cam_mat[:2, :] /= self.factor

            # 2. parse distortion parameters
            cam_model = cam.camera_type
            if cam_model == 0 or cam_model == "SIMPLE_PINHOLE":
                dist_coeffs = np.empty(0, dtype=np.float32)
                cam_type = "perspective"
            elif cam_model == 1 or cam_model == "PINHOLE":
                dist_coeffs = np.empty(0, dtype=np.float32)
                cam_type = "perspective"
            if cam_model == 2 or cam_model == "SIMPLE_RADIAL":
                dist_coeffs = np.array([cam.k1, 0.0, 0.0, 0.0], dtype=np.float32)
                cam_type = "perspective"
            elif cam_model == 3 or cam_model == "RADIAL":
                dist_coeffs = np.array([cam.k1, cam.k2, 0.0, 0.0], dtype=np.float32)
                cam_type = "perspective"
            elif cam_model == 4 or cam_model == "OPENCV":
                dist_coeffs = np.array(
                    [cam.k1, cam.k2, cam.p1, cam.p2], dtype=np.float32
                )
                cam_type = "perspective"
            elif cam_model == 5 or cam_model == "OPENCV_FISHEYE":
                dist_coeffs = np.array(
                    [cam.k1, cam.k2, cam.k3, cam.k4], dtype=np.float32
                )
                cam_type = "fisheye"
            assert (
                cam_type == "perspective" or cam_type == "fisheye"
            ), f"Only perspective and fisheye cameras are supported, got {cam_model}"

            # 3. parse scaled image size
            img_size = (cam.width // self.factor, cam.height // self.factor)

            cam_dict[cam_id] = {
                "camera_matrix": cam_mat,
                "distortion_coefficients": dist_coeffs,
                "image_size": img_size,
                "camera_type": cam_type,
                "camera_mask": None,
            }
        return cam_dict

    def parse_images(self) -> Dict[int, Dict[str, np.ndarray]]:
        """Parse scaled image files and camera extrinsics, i.e. poses."""
        img_dict = dict()
        for img_id in self.manager.images:
            img = self.manager.images[img_id]
            rot_mat = img.R()
            t_vec = img.tvec.reshape(3, 1)
            w2c_mat = np.concatenate(
                [
                    np.concatenate([rot_mat, t_vec], 1),
                    np.array([0, 0, 0, 1]).reshape(1, 4),
                ],
                axis=0,
            )

            w2c_mat_rs = copy.deepcopy(w2c_mat)
            if self.use_roll_shutter:
                img_rs = self.manager_rs.images[img_id]
                rot_mat_rs = img_rs.R()
                t_vec_rs = img_rs.tvec.reshape(3, 1)
                w2c_mat_rs = np.concatenate(
                    [
                        np.concatenate([rot_mat_rs, t_vec_rs], 1),
                        np.array([0, 0, 0, 1]).reshape(1, 4),
                    ],
                    axis=0,
                )

            cam_id = img.camera_id
            img_name = img.name
            img_dict[img_id] = {
                "image_id": img_id,
                "image_name": img_name,
                "camera_id": cam_id,
                "world_to_camera": w2c_mat,
                "world_to_camera_rs": w2c_mat_rs,
            }
        return img_dict

    def parse_points_3d(self) -> np.ndarray:
        """Parse 3D points XYZ (3) + RGB (3) + Error (1), i.e. Dim=[N, 7] point cloud."""
        dtype = [
            ("XYZ", np.float32, (3,)),
            ("RGB", np.uint8, (3,)),
            ("Error", np.float32, (1,)),
        ]
        pnts_xyz = self.manager.points3D.astype(np.float32)
        pnts_rgb = self.manager.point3D_colors.astype(np.uint8)
        pnts_err = self.manager.point3D_errors.astype(np.float32)[:, None]
        pnts_vals = np.array(
            [row for row in zip(pnts_xyz, pnts_rgb, pnts_err)], dtype=dtype
        )
        return pnts_vals

    def parse_image_to_point_indices(self) -> Dict[str, np.ndarray]:
        """Parse the mapping from each image name to its corresponding indices of 3D points."""
        img_to_pnt_idxs = dict()
        img_id_to_name = {v: k for k, v in self.manager.name_to_image_id.items()}
        for pnt_id, img_info in self.manager.point3D_id_to_images.items():
            for img_id, _ in img_info:
                img_name = img_id_to_name[img_id]
                pnt_idx = self.manager.point3D_id_to_point3D_idx[pnt_id]
                img_to_pnt_idxs.setdefault(img_name, []).append(pnt_idx)
        img_to_pnt_idxs = {
            k: np.array(v).astype(np.int32) for k, v in img_to_pnt_idxs.items()
        }
        return img_to_pnt_idxs

    def find_mask_name_by_image(self, img_name: str):
        """Find corresponding mask name by image name."""
        return img_name.replace(".JPG", ".png")

    def parse(self):
        """Parse COLMAP format cameras, images and points3D."""
        cam_dict = self.parse_cameras()
        img_dict = self.parse_images()
        print(
            f"[ScannetppParser] Parsed {len(img_dict)} images, taken by {len(cam_dict)} cameras."
        )

        pnts_vals = self.parse_points_3d()
        print(f"[ScannetppParser] Parsed {len(pnts_vals)} 3D points.")

        img_to_pnt_idxs = self.parse_image_to_point_indices()
        num_pnts_per_img = np.mean([len(v) for v in img_to_pnt_idxs.values()])
        print(
            f"[ScannetppParser] Average number of points per image is {num_pnts_per_img:.2f}."
        )

        # 1. split camera intrinsic parameters
        self.Ks_dict = {
            cam_id: cam_dict[cam_id]["camera_matrix"] for cam_id in cam_dict
        }
        self.params_dict = {
            cam_id: cam_dict[cam_id]["distortion_coefficients"] for cam_id in cam_dict
        }
        self.mask_dict = {
            cam_id: cam_dict[cam_id]["camera_mask"] for cam_id in cam_dict
        }
        self.imsize_dict = {
            cam_id: cam_dict[cam_id]["image_size"] for cam_id in cam_dict
        }
        self.cam_type_dict = {
            cam_id: cam_dict[cam_id]["camera_type"] for cam_id in cam_dict
        }

        # 2. sort images by their names
        img_name_to_id = {img_dict[img_id]["image_name"]: img_id for img_id in img_dict}
        self.image_names = sorted(list(img_name_to_id.keys()))
        self.image_ids = [img_name_to_id[img_name] for img_name in self.image_names]
        self.camera_ids = [img_dict[img_id]["camera_id"] for img_id in self.image_ids]
        self.image_paths = [
            os.path.join(self.imgs_dir, img_name) for img_name in self.image_names
        ]
        self.mask_paths = [
            os.path.join(self.masks_dir, self.find_mask_name_by_image(img_name))
            for img_name in self.image_names
        ]
        self.camtoworlds = np.linalg.inv(
            np.stack(
                [img_dict[img_id]["world_to_camera"] for img_id in self.image_ids],
                axis=0,
            )
        )
        self.camtoworlds_rs = np.linalg.inv(
            np.stack(
                [img_dict[img_id]["world_to_camera_rs"] for img_id in self.image_ids],
                axis=0,
            )
        )
        self.camtoworlds_mid = np.stack(
            [
                self.interpolate_transform_by_linear(c2w_mat_0, c2w_mat_1, ratio=0.5)
                for c2w_mat_0, c2w_mat_1 in zip(self.camtoworlds, self.camtoworlds_rs)
            ],
            axis=0,
        )

        # 3. split 3D points into XYZ, RGB and Error parts
        self.points = pnts_vals["XYZ"]
        self.points_rgb = pnts_vals["RGB"]
        self.points_err = pnts_vals["Error"]
        self.point_indices = img_to_pnt_idxs

        # size of the scene measured by cameras
        cam_locs = self.camtoworlds_mid[:, :3, 3]
        scene_center = np.mean(cam_locs, axis=0)
        cam_dists = np.linalg.norm(cam_locs - scene_center, axis=1)
        self.scene_scale = np.max(cam_dists)
        self.bounds = np.array([0.01, 1.0], dtype=np.float32)

        self.check_points_normalize()
        self.check_image_scale()
        self.check_camera_undistort()

    def interpolate_transform_by_linear(
        self, tf_mat_0: np.ndarray, tf_mat_1: np.ndarray, ratio: float = 0.5
    ):
        """Interpolates between two transformations according to linear ratio."""
        quat_0 = Quaternion(matrix=tf_mat_0[:3, :3])
        quat_1 = Quaternion(matrix=tf_mat_1[:3, :3])
        quat = Quaternion.slerp(quat_0, quat_1, amount=ratio)
        rot_mat = quat.rotation_matrix

        t_vec_0 = tf_mat_0[:3, 3]
        t_vec_1 = tf_mat_1[:3, 3]
        t_vec = (1 - ratio) * t_vec_0 + ratio * t_vec_1

        tf_mat = np.eye(4, dtype=tf_mat_0.dtype)
        tf_mat[:3, :3] = rot_mat
        tf_mat[:3, 3] = t_vec
        return tf_mat

    def check_points_normalize(self):
        if self.normalize:
            sim_tf = similarity_from_cameras(self.camtoworlds_mid)
            self.camtoworlds = transform_cameras(sim_tf, self.camtoworlds)
            self.camtoworlds_rs = transform_cameras(sim_tf, self.camtoworlds_rs)
            self.points = transform_points(sim_tf, self.points)

            axis_tf = align_principal_axes(self.points)
            self.camtoworlds = transform_cameras(axis_tf, self.camtoworlds)
            self.camtoworlds_rs = transform_cameras(axis_tf, self.camtoworlds_rs)
            self.points = transform_points(axis_tf, self.points)

            self.transform = axis_tf @ sim_tf

            # Fix for up side down. We assume more points towards
            # the bottom of the scene which is true when ground floor is
            # present in the images.
            if np.median(self.points[:, 2]) > np.mean(self.points[:, 2]):
                # rotate 180 degrees around x axis such that z is flipped
                flip_tf = np.array(
                    [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, -1.0, 0.0, 0.0],
                        [0.0, 0.0, -1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                )
                self.camtoworlds = transform_cameras(flip_tf, self.camtoworlds)
                self.camtoworlds_rs = transform_cameras(flip_tf, self.camtoworlds_rs)
                self.points = transform_points(flip_tf, self.points)
                self.transform = flip_tf @ self.transform
        else:
            self.transform = np.eye(4, dtype=np.float32)

    def check_image_scale(self):
        """Check scale of images compared with the original ones from COLMAP."""
        cam_id_to_img_files = dict()
        for cam_id, img_file in zip(self.camera_ids, self.image_paths):
            cam_id_to_img_files.setdefault(cam_id, []).append(img_file)

        for cam_id in cam_id_to_img_files:
            img_file = cam_id_to_img_files[cam_id][0]
            assert os.path.exists(img_file), f"File not exists: {img_file}!"
            img = cv2.imread(img_file)[..., :3]
            actual_h, actual_w = img.shape[:2]
            colmap_w, colmap_h = self.imsize_dict[cam_id]

            if actual_w == colmap_w and actual_h == colmap_h:
                continue

            scale_h = actual_h / colmap_h
            scale_w = actual_w / colmap_w
            self.Ks_dict[cam_id][0, :] *= scale_w
            self.Ks_dict[cam_id][1, :] *= scale_h
            self.imsize_dict[cam_id] = (int(actual_w), int(actual_h))

    def check_camera_undistort(self):
        """Check undistortion parameters applied on each camera."""
        self.mapx_dict = dict()
        self.mapy_dict = dict()
        self.roi_undist_dict = dict()
        for cam_id in self.params_dict.keys():
            dist_coeffs = self.params_dict[cam_id]
            if len(dist_coeffs) == 0:
                continue  # no distortion
            assert cam_id in self.Ks_dict, f"Missing K for camera {cam_id}!"
            cam_mat = self.Ks_dict[cam_id]
            cam_w, cam_h = self.imsize_dict[cam_id]
            cam_type = self.cam_type_dict[cam_id]

            if cam_type == "perspective":
                cam_mat_undist, roi_undist = cv2.getOptimalNewCameraMatrix(
                    cam_mat, dist_coeffs, (cam_w, cam_h), 0
                )
                mapx, mapy = cv2.initUndistortRectifyMap(
                    cam_mat,
                    dist_coeffs,
                    None,
                    cam_mat_undist,
                    (cam_w, cam_h),
                    cv2.CV_32FC1,
                )
            elif cam_type == "fisheye":
                fx = cam_mat[0, 0]
                fy = cam_mat[1, 1]
                cx = cam_mat[0, 2]
                cy = cam_mat[1, 2]
                grid_x, grid_y = np.meshgrid(
                    np.arange(cam_w, dtype=np.float32),
                    np.arange(cam_h, dtype=np.float32),
                    indexing="xy",
                )
                x1 = (grid_x - cx) / fx
                y1 = (grid_y - cy) / fy
                theta = np.sqrt(x1**2 + y1**2)
                r = (
                    1.0
                    + dist_coeffs[0] * theta**2
                    + dist_coeffs[1] * theta**4
                    + dist_coeffs[2] * theta**6
                    + dist_coeffs[3] * theta**8
                )
                mapx = (fx * x1 * r + cam_w // 2).astype(np.float32)
                mapy = (fy * y1 * r + cam_h // 2).astype(np.float32)

                # Use mask to define ROI
                mask = np.logical_and(
                    np.logical_and(mapx > 0, mapy > 0),
                    np.logical_and(mapx < cam_w - 1, mapy < cam_h - 1),
                )
                y_indices, x_indices = np.nonzero(mask)
                y_min, y_max = y_indices.min(), y_indices.max() + 1
                x_min, x_max = x_indices.min(), x_indices.max() + 1
                mask = mask[y_min:y_max, x_min:x_max]
                cam_mat_undist = cam_mat.copy()
                cam_mat_undist[0, 2] -= x_min
                cam_mat_undist[1, 2] -= y_min
                roi_undist = [x_min, y_min, x_max - x_min, y_max - y_min]
            else:
                assert_never(cam_type)

            self.mapx_dict[cam_id] = mapx
            self.mapy_dict[cam_id] = mapy
            self.Ks_dict[cam_id] = cam_mat_undist
            self.imsize_dict[cam_id] = (roi_undist[2], roi_undist[3])
            self.mask_dict[cam_id] = (
                mask
                if self.mask_dict[cam_id] is None
                else np.logical_and(self.mask_dict[cam_id], mask)
            )
            self.roi_undist_dict[cam_id] = roi_undist
