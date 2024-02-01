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

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from training.gaussian_splatting.gaussian_model import GaussianModel
from training.gaussian_splatting.utils.sh_utils import eval_sh
from ipdb import set_trace as st


def print_grad(name, grad):
    print(f"{name}:")
    if torch.all(grad==0):
        print("grad all 0s")
        return 
    # print(grad)
    print('\t',grad.max(), grad.min(), grad.mean())

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        # debug=pipe.debug
        debug=False
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    # if pipe.compute_cov3D_python:
    #     cov3D_precomp = pc.get_covariance(scaling_modifier)
    # else:
    scales = pc.get_scaling
    rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        # if pipe.convert_SHs_python:
        #     shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
        #     dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
        #     dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
        #     sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
        #     colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        # else:
        shs = pc.get_features # should map to ([0,1]=0.5)/C0
    else:
        st()
        colors_precomp = override_color # override_color -> [Npts, 3], range in [0,1]
    

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii, rendered_depth, rendered_alpha = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "depth": rendered_depth,
            "alpha": rendered_alpha,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}

def batch_render(viewpoint_camera_list, pc, pipe, bg_color : torch.Tensor,
           scaling_modifier = 1.0, override_color = None, deformation_kwargs = None,
           device_='cuda'
           ):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """
    from batch_diff_gaussian_rasterization import BatchGaussianRasterizationSettings, BatchGaussianRasterizer

    B = len(viewpoint_camera_list)
    # B = viewpoint_camera_list.world_view_transform.shape[0]
    deformation_kwargs = {} if deformation_kwargs is None else deformation_kwargs
    # means3D_list = []
    # means2D_list = []

    # for i in range(B):
    # if not isinstance(pc.get_xyz, torch.Tensor):
    #         means3D = pc.get_xyz(**deformation_kwargs)
    # else:
    means3D = pc["_xyz"]

    screenspace_points = torch.zeros_like(means3D, dtype=means3D.dtype, requires_grad=means3D.requires_grad, device=device_) + 0
        # screenspace_points = screenspace_points[None, ...].repeat(B, 1, 1)
    if screenspace_points.requires_grad:
            try:
                screenspace_points.retain_grad()
            except:
                pass
        # means3D_list.append(means3D)
        # means2D_list.append(screenspace_points)
    # means3D = torch.stack(means3D_list)
    # means2D = torch.stack(means2D_list)
    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera_list[0].FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera_list[0].FoVy * 0.5)
    # print(viewpoint_camera_list.FoVx )
    # print(viewpoint_camera_list.FoVy )
    # tanfovx = math.tan(viewpoint_camera_list.FoVx * 0.5)
    # tanfovy = math.tan(viewpoint_camera_list.FoVy * 0.5)

    # world_view_transform_list = viewpoint_camera_list.world_view_transform# torch.stack([c.world_view_transform for c in viewpoint_camera_list], axis=0)
    # full_proj_transform_list = viewpoint_camera_list.full_proj_transform #torch.stack([c.full_proj_transform for c in viewpoint_camera_list], axis=0)
    # camera_center_list =viewpoint_camera_list.camera_center #torch.stack([c.camera_center for c in viewpoint_camera_list], axis=0)
    world_view_transform_list = torch.stack([c.world_view_transform for c in viewpoint_camera_list], axis=0)
    full_proj_transform_list = torch.stack([c.full_proj_transform for c in viewpoint_camera_list], axis=0)
    camera_center_list = torch.stack([c.camera_center for c in viewpoint_camera_list], axis=0)
    raster_settings = BatchGaussianRasterizationSettings(
        image_height=int(viewpoint_camera_list[0].image_height),
        image_width=int(viewpoint_camera_list[0].image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=world_view_transform_list,
        projmatrix=full_proj_transform_list,
        sh_degree=pc["active_sh_degree"],
        campos=camera_center_list,
        prefiltered=False,
        gaussian_batched=True,
        debug=False
    )

    batch_rasterizer = BatchGaussianRasterizer(raster_settings=raster_settings)
    # means3D = pc.get_xyz[None, ...].repeat(B, 1, 1)
    means2D = screenspace_points
    opacity = pc["_opacity"]

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    # if pipe.compute_cov3D_python:
    #     cov3D_precomp = pc.get_covariance(scaling_modifier)
    # else:
    scales = pc["_scaling"]
    rotations = pc["_rotation"]

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        # if pipe.convert_SHs_python:
        #     # shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
        #     # dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
        #     # dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
        #     # sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
        #     # colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        #     # # colors_precomp = pc.get_my_features
        #     #colors_precomp = torch.clamp_min(pc.get_features[:, :3], 0.0)  # TODO: better filter
        #     # colors_precomp = torch.clamp_min(pc.get_features, 0.0) # TODO: better filter
        #     colors_precomp = torch.sigmoid(pc.get_features)
        #     # colors_precomp = colors_precomp[None,...].repeat(B, 1, 1)  # TODO: better filter
        # else:
        shs = pc["_features"]
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen).

    # print("(means2D)", means2D.shape, means2D.device)
    # print("(means3D)", means3D.shape, means3D.device)
    # # print("(shs)", shs.shape, shs.device)
    # print("(colors_precomp)", colors_precomp.shape, colors_precomp.device)
    # print("(opacity)", opacity.shape, opacity.device)
    # print("(scales)", scales.shape, scales.device)
    # print("(rotations)", rotations.shape, rotations.device)
    # # print("(cov3D_precomp)", cov3D_precomp.shape, cov3D_precomp.device)

    rendered_image, radii = batch_rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    # print(screenspace_points.shape, screenspace_points.device)
    # print(means3D.shape, means3D.device)
    # # print(shs.shape, shs.device)
    # print(colors_precomp.shape, colors_precomp.device)
    # print(opacity.shape, opacity.device)
    # print(scales.shape, scales.device)
    # print(rotations.shape, rotations.device)
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}
