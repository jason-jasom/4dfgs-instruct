#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from utils.data_utils import CameraDataset

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0], ply_path=None):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        self.white_background = args.white_background

        direct_ply_path = getattr(args, "ply_path", "")
        direct_mlp_dir = getattr(args, "mlp_dir", "")
        direct_checkpoint_dir = getattr(args, "checkpoint_dir", "")
        if direct_checkpoint_dir:
            direct_ply_path = direct_ply_path or os.path.join(direct_checkpoint_dir, "point_cloud.ply")
            direct_mlp_dir = direct_mlp_dir or direct_checkpoint_dir

        if direct_ply_path and direct_mlp_dir and load_iteration is None:
            self.loaded_iter = "custom"
            print("Loading trained model from {}".format(direct_mlp_dir))
        elif load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
                
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        print(args.source_path)

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval, args.lod)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args, args.source_path, args.white_background, args.eval, ply_path=ply_path)
        elif os.path.exists(os.path.join(args.source_path, "colmap")):
            duration = 50 # technicolor
            scene_info = sceneLoadTypeCallbacks["Technicolor"](args, args.source_path, None, args.eval, duration=duration)
        else:
            assert False, "Could not recognize scene type!"

        self.gaussians.set_appearance(len(scene_info.train_cameras))
        
        if not self.loaded_iter:
            if ply_path is not None:
                with open(ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                    dest_file.write(src_file.read())
            else:
                with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                    dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        # print(f'self.cameras_extent: {self.cameras_extent}')

        skip_test_cameras = getattr(args, "skip_test_cameras", False)
        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            if skip_test_cameras:
                print("Skipping Test Cameras")
                self.test_cameras[resolution_scale] = []
            else:
                print("Loading Test Cameras")
                self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        if self.loaded_iter:
            if direct_ply_path and direct_mlp_dir:
                self.gaussians.load_ply_sparse_gaussian(direct_ply_path)
                self.gaussians.load_mlp_checkpoints(direct_mlp_dir)
            else:
                self.gaussians.load_ply_sparse_gaussian(os.path.join(self.model_path,
                                                               "point_cloud",
                                                               "iteration_" + str(self.loaded_iter),
                                                               "point_cloud.ply"))
                self.gaussians.load_mlp_checkpoints(os.path.join(self.model_path,
                                                               "point_cloud",
                                                               "iteration_" + str(self.loaded_iter)))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent, args.init_voxel_scale)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        self.gaussians.save_mlp_checkpoints(point_cloud_path)
    def save_3dedit(self, iteration, prompt):
        point_cloud_path = os.path.join(self.model_path, "point_cloud_3dedit/{}/iteration_{}".format(prompt, iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        self.gaussians.save_mlp_checkpoints(point_cloud_path)
    def save_refine(self, iteration, prompt):
        point_cloud_path = os.path.join(self.model_path, "point_cloud_refine/{}/iteration_{}".format(prompt, iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        self.gaussians.save_mlp_checkpoints(point_cloud_path)

    def getTrainCameras(self, scale=1.0, view_only=False, rgba=False):
        return CameraDataset(self.train_cameras[scale].copy(), self.white_background, view_only, rgba)
        
    def getTestCameras(self, scale=1.0, view_only=False, rgba=False):
        return CameraDataset(self.test_cameras[scale].copy(), self.white_background, view_only, rgba)
