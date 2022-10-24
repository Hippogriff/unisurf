import pyrender
from open3d import geometry
import pyrr
import trimesh
from matplotlib import pyplot as plt
from render import render_mesh
from model.render_utils import normalize_mesh
import pyrender
import numpy as np
import torch

class Loss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = torch.nn.L1Loss()

    def forward(self, pred, gt):
        return self.loss(pred, gt)


class GenerateRays():
    def __init__(self, mesh_path):
        mesh = trimesh.load(mesh_path, force="mesh")
        normalize_mesh(mesh)
        self.mesh = mesh
        self.camera_poses = self.create_random_uniform_camera_poses()
        self.vectors = {}
        self.origins = {}
        self.index_rays = {}
        self.pixels = {}
        self.points = {}
        self.depths = {}
        self.close_cameras()
        seq = np.arange(len(self.camera_poses))

        np.random.shuffle(seq)

        self.train_set = seq[0:int(len(self.camera_poses) * 0.8)]
        self.test_set = seq[int(len(self.camera_poses) * 0.8):]
        print("Training set: ", self.train_set)
        print("Testing set: ", self.test_set)

    def generate(self):
        rayintersector = trimesh.ray.ray_pyembree.RayMeshIntersector(self.mesh)

        for index in range(self.camera_poses.shape[0]):
            extra = np.eye(4)
            extra[0, 0] = 0
            extra[0, 1] = 1
            extra[1, 0] = -1
            extra[1, 1] = 0
            scene = self.mesh.scene()

            # np.linalg.inv(create_look_at(frontVector, np.zeros(3), np.array([0, 1, 0])))
            scene.camera_transform = self.camera_poses[index] @ extra  # @ np.diag([1, -1,-1, 1]
            # scene.camera_transform = camera_transform_matrix(frontVector, np.zeros(3), np.array([0, 1, 0])) @ e

            # any of the automatically generated values can be overridden
            # set resolution, in pixels
            scene.camera.resolution = [256, 256]

            # set field of view, in degrees
            # make it relative to resolution so pixels per degree is same
            scene.camera.fov = 60, 60

            # convert the camera to rays with one ray per pixel
            origins, vectors, pixels = scene.camera_rays()

            # camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.)
            # camera = scene.add(camera, pose=self.camera_poses[index])
            # origins, vectors, pixels = scene.camera_rays()

            index_tri, index_ray, points = rayintersector.intersects_id(
                origins, vectors, multiple_hits=False, return_locations=True)

            depths = trimesh.util.diagonal_dot(points - origins[0], vectors[index_ray])

            self.vectors[index] = vectors
            self.origins[index] = origins
            self.index_rays[index] = index_ray
            self.pixels[index] = pixels
            self.depths[index] = depths
            self.points[index] = points

    def create_random_uniform_camera_poses(self, distance=2, low_scale=0.51):
        mesh = geometry.TriangleMesh()
        frontvectors = np.array(mesh.create_sphere(distance, 7).vertices)

        # frontvectors = self.rotate_perturbation_point_cloud(
        #     np.expand_dims(frontvectors, 1), angle_clip=0.25
        # )
        # frontvectors = self.random_scale_point_cloud(
        #     frontvectors, scale_low=low_scale, scale_high=1.0
        # )[:, 0, :]

        camera_poses = []
        for i in range(frontvectors.shape[0]):
            camera_pose = np.array(
                pyrr.Matrix44.look_at(
                    eye=frontvectors[i], target=np.zeros(3), up=np.array([0.0, 1.0, 0])
                ).T
            )
            camera_pose = np.linalg.inv(np.array(camera_pose))
            camera_poses.append(camera_pose)
        return np.stack(camera_poses, 0)

    def rotate_perturbation_point_cloud(self, batch_data, angle_sigma=0.06, angle_clip=0.18):
        """ Randomly perturb the point clouds by small rotations
            Input:
              BxNx3 array, original batch of point clouds
            Return:
              BxNx3 array, rotated batch of point clouds
        """
        rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
        for k in range(batch_data.shape[0]):
            angles = np.clip(angle_sigma * np.random.randn(3), -angle_clip, angle_clip)
            Rx = np.array([[1, 0, 0],
                           [0, np.cos(angles[0]), -np.sin(angles[0])],
                           [0, np.sin(angles[0]), np.cos(angles[0])]])
            Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                           [0, 1, 0],
                           [-np.sin(angles[1]), 0, np.cos(angles[1])]])
            Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                           [np.sin(angles[2]), np.cos(angles[2]), 0],
                           [0, 0, 1]])
            R = np.dot(Rz, np.dot(Ry, Rx))
            shape_pc = batch_data[k, ...]
            rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), R)
        return rotated_data

    def random_scale_point_cloud(self, batch_data, scale_low=0.8, scale_high=1.25):
        """ Randomly scale the point cloud. Scale is per point cloud.
            Input:
                BxNx3 array, original batch of point clouds
            Return:
                BxNx3 array, scaled batch of point clouds
        """
        B, N, C = batch_data.shape
        scales = np.random.uniform(scale_low, scale_high, B)
        for batch_index in range(B):
            batch_data[batch_index, :, :] *= scales[batch_index]
        return batch_data

    def generate_data(self, batch_size, conver_to_cuda=True, invert_depth=True, set_type="train"):
        if set_type == "train":
            random_index = np.random.choice(self.train_set)
        else:
            random_index = np.random.choice(self.test_set)

        rays = None
        centers = None
        depth = None

        # Find out-sample of shape pixels
        all_valid_indices = self.index_rays[random_index]
        random_indices_on_surface = np.random.choice(len(all_valid_indices), batch_size // 2)
        on_surface_indices = all_valid_indices[random_indices_on_surface]

        # points = self.points[random_index][indices]
        depth_on_surface = self.depths[random_index][random_indices_on_surface]

        off_surface_indices = np.ones((len(self.pixels[random_index])), dtype=np.int32)
        off_surface_indices[self.index_rays[random_index]] = 0
        off_surface_indices = np.where(off_surface_indices)[0]

        random_indices_off_surface = np.random.choice(len(off_surface_indices), batch_size // 2)
        off_surface_indices = off_surface_indices[random_indices_off_surface]

        # Find in-sample shape pixels
        depth_off_surface = np.ones((batch_size // 2), dtype=np.float32) * 10

        on_surface_vectors = self.vectors[random_index][on_surface_indices]
        off_surface_vectors = self.vectors[random_index][off_surface_indices]

        depths = np.concatenate([depth_on_surface, depth_off_surface], 0)
        vectors = np.concatenate([on_surface_vectors, off_surface_vectors], 0)

        on_surface_points = self.points[random_index][random_indices_on_surface]
        origin = self.origins[random_index][0:batch_size]
        data = {"depths": depths, "vectors": vectors, "origins": origin,
                "points": on_surface_points, "index": random_index}
        if conver_to_cuda:
            self.convert_to_cuda(data, invert_depth=invert_depth)
        return data

    def convert_to_cuda(self, data, invert_depth=False):
        data["depths"] = torch.from_numpy(np.expand_dims(data["depths"], 0).astype(np.float32)).cuda()
        if invert_depth:
            data["depths"] = 1 / (data["depths"] + 1e-6)
        data["origins"] = torch.from_numpy(np.expand_dims(data["origins"].astype(np.float32), axis=0)).cuda()
        data["vectors"] = torch.from_numpy(np.expand_dims(data["vectors"].astype(np.float32), axis=0)).cuda()

    def close_cameras(self):
        camera_poses_t = self.camera_poses[:, :, -1][:, 0:3]
        dist = (np.expand_dims(camera_poses_t, 0) - np.expand_dims(camera_poses_t, 1)) ** 2
        dist = np.sum(dist, 2)
        indices = np.argsort(dist, axis=1)[:, 1:]
        self.close_camera_indices = indices

    def generate_depths_from_other_views(self, pred_disp, origins, vectors, ref_view, closest_index=0):
        points = origins + 1 / (pred_disp.reshape(-1, 1) + 1e-6) * vectors
        close_cameras = self.camera_poses[self.close_camera_indices[ref_view][closest_index]]
        close_camera_center = close_cameras[:, -1][0:3]
        close_camera_center = torch.from_numpy(close_camera_center.astype(np.float32)).cuda().reshape((1, 3))
        close_camera_center = close_camera_center.repeat(points.shape[1], 1)

        new_vectors = points - close_camera_center

        depth = new_vectors.norm(p=2, dim=-1, keepdim=True) + 1e-6
        new_vectors = new_vectors / (depth)

        close_camera_center = torch.unsqueeze(close_camera_center, 0)
        return depth, new_vectors.detach(), close_camera_center.detach(), points.detach()


def smoothplot(x, sigma=10):
    from scipy.ndimage.filters import gaussian_filter1d
    ysmoothed = gaussian_filter1d(x, sigma=sigma)
    return ysmoothed
