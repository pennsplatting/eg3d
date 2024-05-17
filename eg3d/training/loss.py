# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Loss functions."""

import numpy as np
import torch
from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import upfirdn2d
from training.dual_discriminator import filtered_resizing

import dnnlib
import legacy
from pdb import set_trace as st
from torch_utils.extract_edge import EdgeExtractor
from PIL import Image

from training.face_parsing.model import BiSeNet
import torch.nn.functional as F
import torchvision.transforms as transforms
#----------------------------------------------------------------------------

class Loss:
    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, gain, cur_nimg): # to be overridden by subclass
        raise NotImplementedError()

#----------------------------------------------------------------------------

class StyleGAN2Loss(Loss):
    def __init__(self, device, G, D, edge_discriminate=False, use_segmentation=False, depth_distill=False, augment_pipe=None, r1_gamma=10, style_mixing_prob=0, pl_weight=0, pl_batch_shrink=2, pl_decay=0.01, pl_no_weight_grad=False, blur_init_sigma=0, blur_fade_kimg=0, r1_gamma_init=0, r1_gamma_fade_kimg=0, neural_rendering_resolution_initial=64, neural_rendering_resolution_final=None, neural_rendering_resolution_fade_kimg=0, gpc_reg_fade_kimg=1000, gpc_reg_prob=None, dual_discrimination=False, filter_mode='antialiased'):
        super().__init__()
        self.device             = device
        self.G                  = G
        self.D                  = D
        self.depth_distill      = depth_distill
        self.edge_discriminate  = edge_discriminate
        self.use_segmentation   = use_segmentation
        self.augment_pipe       = augment_pipe
        self.r1_gamma           = r1_gamma
        self.style_mixing_prob  = style_mixing_prob
        self.pl_weight          = pl_weight
        self.pl_batch_shrink    = pl_batch_shrink
        self.pl_decay           = pl_decay
        self.pl_no_weight_grad  = pl_no_weight_grad
        self.pl_mean            = torch.zeros([], device=device)
        self.blur_init_sigma    = blur_init_sigma
        self.blur_fade_kimg     = blur_fade_kimg
        self.r1_gamma_init      = r1_gamma_init
        self.r1_gamma_fade_kimg = r1_gamma_fade_kimg
        self.neural_rendering_resolution_initial = neural_rendering_resolution_initial
        self.neural_rendering_resolution_final = neural_rendering_resolution_final
        self.neural_rendering_resolution_fade_kimg = neural_rendering_resolution_fade_kimg
        self.gpc_reg_fade_kimg = gpc_reg_fade_kimg
        self.gpc_reg_prob = gpc_reg_prob
        self.dual_discrimination = dual_discrimination
        self.filter_mode = filter_mode
        self.resample_filter = upfirdn2d.setup_filter([1,3,3,1], device=device)
        self.blur_raw_target = True
        assert self.gpc_reg_prob is None or (0 <= self.gpc_reg_prob <= 1)

        if self.depth_distill:
            network_pkl = "/home/zxy/eg3d/eg3d/data/eg3d_1/ffhq512-128.pkl"
            # network_pkl = "/root/zxy/data/ffhq512-128.pkl"
            with dnnlib.util.open_url(network_pkl) as f:
                self.guide_G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore
        if self.edge_discriminate:
            self.edge_extractor = EdgeExtractor().cuda()
        if self.use_segmentation:
            pth = '/home/1TB/79999_iter.pth'
            n_classes = 19
            self.net = BiSeNet(n_classes=n_classes)
            self.net.cuda()
            self.net.load_state_dict(torch.load(pth))
            self.net.eval()

            # self.model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
            # self.model.eval()  # Set the model to evaluation mode

            # # If you're using a GPU
            # self.model.cuda()
        #     self.segmentation_model = load_segmentation_model()
        #     self.decode_fn = VOCSegmentation.decode_target
            
    def segmentation(self, img):
        # n_classes = 19
        # self.net = BiSeNet(n_classes=n_classes)
        # self.net.cuda()
        # self.net.load_state_dict(torch.load(pth))
        # self.net.eval()
        N, C, H, W = img.shape
        to_tensor = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        img = F.interpolate(img, size=(512, 512), mode='bilinear', align_corners=False)
        with torch.no_grad():
            # img = img.resize((512, 512), Image.BILINEAR)
            img = (img + 1.0) / 2
            img = to_tensor(img)
            # print(img.max(), img.min())
            out = self.net(img)[0].argmax(1).squeeze()
            # print(out.max(), out.min())
            # exit(0)
            # parsing = out.squeeze(0).cpu().numpy().argmax(0)
            
            return out # [8, 1, 128, 128]
            # print(np.unique(parsing))

    def run_G(self, z, c, swapping_prob, neural_rendering_resolution, update_emas=False):
        if swapping_prob is not None:
            c_swapped = torch.roll(c.clone(), 1, 0)
            c_gen_conditioning = torch.where(torch.rand((c.shape[0], 1), device=c.device) < swapping_prob, c_swapped, c)
        else:
            c_gen_conditioning = torch.zeros_like(c)

        ws = self.G.mapping(z, c_gen_conditioning, update_emas=update_emas)
        if self.style_mixing_prob > 0:
            with torch.autograd.profiler.record_function('style_mixing'):
                cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                ws[:, cutoff:] = self.G.mapping(torch.randn_like(z), c, update_emas=False)[:, cutoff:]
        gen_output = self.G.synthesis(ws, c, neural_rendering_resolution=neural_rendering_resolution, update_emas=update_emas)
        return gen_output, ws, c_gen_conditioning

    def run_D(self, img, c, blur_sigma=0, blur_sigma_raw=0, update_emas=False):
        blur_size = np.floor(blur_sigma * 3)
        if blur_size > 0:
            with torch.autograd.profiler.record_function('blur'):
                f = torch.arange(-blur_size, blur_size + 1, device=img['image'].device).div(blur_sigma).square().neg().exp2()
                img['image'] = upfirdn2d.filter2d(img['image'], f / f.sum())

        if self.augment_pipe is not None:
            augmented_pair = self.augment_pipe(torch.cat([img['image'],
                                                    torch.nn.functional.interpolate(img['image_raw'], size=img['image'].shape[2:], mode='bilinear', antialias=True)],
                                                    dim=1))
            img['image'] = augmented_pair[:, :img['image'].shape[1]]
            img['image_raw'] = torch.nn.functional.interpolate(augmented_pair[:, img['image'].shape[1]:], size=img['image_raw'].shape[2:], mode='bilinear', antialias=True)

        logits = self.D(img, c, update_emas=update_emas)
        return logits

    # TODO: L2 loss for debug
    def accumulate_gradients_debug(self, phase, real_img, real_c, gen_z, gen_c, gain, cur_nimg):
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        if self.G.rendering_kwargs.get('density_reg', 0) == 0:
            phase = {'Greg': 'none', 'Gboth': 'Gmain'}.get(phase, phase)
        if self.r1_gamma == 0:
            phase = {'Dreg': 'none', 'Dboth': 'Dmain'}.get(phase, phase)
        blur_sigma = max(1 - cur_nimg / (self.blur_fade_kimg * 1e3), 0) * self.blur_init_sigma if self.blur_fade_kimg > 0 else 0
        r1_gamma = self.r1_gamma

        alpha = min(cur_nimg / (self.gpc_reg_fade_kimg * 1e3), 1) if self.gpc_reg_fade_kimg > 0 else 1
        swapping_prob = (1 - alpha) * 1 + alpha * self.gpc_reg_prob if self.gpc_reg_prob is not None else None       
        
        if self.neural_rendering_resolution_final is not None:
            alpha = min(cur_nimg / (self.neural_rendering_resolution_fade_kimg * 1e3), 1)
            neural_rendering_resolution = int(np.rint(self.neural_rendering_resolution_initial * (1 - alpha) + self.neural_rendering_resolution_final * alpha))
        else:
            neural_rendering_resolution = self.neural_rendering_resolution_initial
            
        with torch.autograd.profiler.record_function('Gmain_forward'):
            gen_img, _gen_ws, _ = self.run_G(gen_z, gen_c, swapping_prob=swapping_prob, neural_rendering_resolution=neural_rendering_resolution)
            loss = torch.nn.functional.mse_loss(real_img, gen_img["image"])
        with torch.autograd.profiler.record_function('Gmain_backward'):
            loss.backward()
            
        ## FIXME: debug grad
        # print(f"Gradients for self.G -----begin---")
        # for name, param in self.G.named_parameters():
        #     # print(f" {name}") # not including gaussian and minicam
        #     if param.grad is not None:
        #         print(f"Gradients for {name} have been computed.")
        #     else:
        #         pass
        #         # print(f"Gradients for {name} have NOT been computed!!")
        # print(f"Gradients for self.G -----end---")
        
    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, gain, cur_nimg, cur_tick):
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        if self.G.rendering_kwargs.get('density_reg', 0) == 0:
            phase = {'Greg': 'none', 'Gboth': 'Gmain'}.get(phase, phase)
        if self.r1_gamma == 0:
            phase = {'Dreg': 'none', 'Dboth': 'Dmain'}.get(phase, phase)
        blur_sigma = max(1 - cur_nimg / (self.blur_fade_kimg * 1e3), 0) * self.blur_init_sigma if self.blur_fade_kimg > 0 else 0
        r1_gamma = self.r1_gamma

        alpha = min(cur_nimg / (self.gpc_reg_fade_kimg * 1e3), 1) if self.gpc_reg_fade_kimg > 0 else 1
        swapping_prob = (1 - alpha) * 1 + alpha * self.gpc_reg_prob if self.gpc_reg_prob is not None else None

        if self.neural_rendering_resolution_final is not None:
            alpha = min(cur_nimg / (self.neural_rendering_resolution_fade_kimg * 1e3), 1)
            neural_rendering_resolution = int(np.rint(self.neural_rendering_resolution_initial * (1 - alpha) + self.neural_rendering_resolution_final * alpha))
        else:
            neural_rendering_resolution = self.neural_rendering_resolution_initial

        if self.edge_discriminate:
            with torch.no_grad():
                real_img_edge = self.edge_extractor(real_img)
        real_img_raw = filtered_resizing(real_img, size=neural_rendering_resolution, f=self.resample_filter, filter_mode=self.filter_mode)

        if self.use_segmentation:
            img_mask = self.segmentation(real_img)
            N, H, W = img_mask.shape
            img_mask = torch.tensor(img_mask, dtype=torch.uint8)
            # img_mask = img_mask[0]
            # print(img_mask.shape)
            # print(torch.unique(img_mask))
            num_of_class = torch.max(img_mask)
            # print(img_mask.max(), img_mask.min(), img_mask.shape) # [8, 128, 128]

            # part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
            #        [255, 0, 85], [255, 0, 170],
            #        [0, 255, 0], [85, 255, 0], [170, 255, 0],
            #        [0, 255, 85], [0, 255, 170],
            #        [0, 0, 255], [85, 0, 255], [170, 0, 255],
            #        [0, 85, 255], [0, 170, 255],
            #        [255, 255, 0], [255, 255, 85], [255, 255, 170],
            #        [255, 0, 255], [255, 85, 255], [255, 170, 255],
            #        [0, 255, 255], [85, 255, 255], [170, 255, 255]]
            real_img_mask = torch.zeros((N, 1, H, W), dtype=torch.float) + 255
            for pi in range(1, num_of_class + 1):
                index = torch.where(img_mask == pi)
                # color[index[0], index[1], :] = torch.tensor(part_colors[pi], dtype=torch.uint8)
                real_img_mask[index[0], 0, index[1], index[2]] = torch.tensor([0], dtype=torch.float)

            real_img_mask = F.interpolate(real_img_mask, size=(128, 128), mode='bilinear', align_corners=False).to("cuda") # face should be 0
            # print(real_img_mask[0])
            # for i in range(8):
            #     color = real_img_mask[i].squeeze().detach().cpu().numpy().astype(np.uint8)
            #     color = Image.fromarray(color,'L').save(f'/home/zxy/eg3d/test_mask/prediction_{i}.png')
            # exit(0)
            real_img_mask = -(real_img_mask / 255 - 0.5) * 2 # [8,1,128,128]
            # print(real_img_mask.shape)
            # exit(0)

            # print(real_img_mask.shape)
            # print(color.shape)
            # print(img_mask)
            # color = color.detach().cpu().numpy()
            # color = Image.fromarray(color).save('/home/zxy/eg3d/eg3d/prediction.png')
            # exit(0)
            
            # with torch.no_grad():
            #     # Get the predictions
            #     predictions = self.model(real_img)

            # masks = []
            # for i, prediction in enumerate(predictions):
            #     # Sort the predictions by score, and take the top one. Assuming one face per image.
            #     scores = prediction['scores']
            #     best_index = torch.argmax(scores).item() if len(scores) > 0 else None
                
            #     # Create an empty mask for each image
            #     mask = torch.zeros_like(real_img[i, 0])  # Use the first channel to match image height and width
                
            #     if best_index is not None and scores[best_index] > 0.5:  # Applying a score threshold
            #         box = prediction['boxes'][best_index]
            #         x1, y1, x2, y2 = map(int, box.tolist())
            #         mask[y1:y2, x1:x2] = 1  # Fill the mask within the bounding box
                
            #     masks.append(mask.unsqueeze(0))  # Add channel dimension to be consistent with input [B,1,H,W]
            # real_img_mask = torch.stack(masks)
            # print(real_img_mask.shape)
            # save_mask = mask.squeeze().cpu().detach().numpy().astype(np.uint8)
            # print(save_mask.shape)
            # Image.fromarray(save_mask,'L').save('/home/zxy/eg3d/eg3d/prediction.png')
            # exit(0)
            
        if self.blur_raw_target:
            blur_size = np.floor(blur_sigma * 3)
            if blur_size > 0:
                f = torch.arange(-blur_size, blur_size + 1, device=real_img_raw.device).div(blur_sigma).square().neg().exp2()
                real_img_raw = upfirdn2d.filter2d(real_img_raw, f / f.sum())

        if self.use_segmentation:
            real_img = {'image': real_img, 'image_raw': real_img_raw, 'image_edge': real_img_edge, 'image_mask': real_img_mask} # no loss is calculated based on depth 
        else:
            real_img = {'image': real_img, 'image_raw': real_img_raw, 'image_edge': real_img_edge}
            
        # Gmain: Maximize logits for generated images.
        if phase in ['Gmain', 'Gboth']:
            with torch.autograd.profiler.record_function('Gmain_forward'):
                gen_img, _gen_ws, _gen_conditioning_params = self.run_G(gen_z, gen_c, swapping_prob=swapping_prob, neural_rendering_resolution=neural_rendering_resolution)
                gen_logits = self.run_D(gen_img, gen_c, blur_sigma=blur_sigma)
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Gmain = torch.nn.functional.softplus(-gen_logits)
                training_stats.report('Loss/G/loss', loss_Gmain)

            with torch.autograd.profiler.record_function('Gmain_backward'):
                loss_Gmain.mean().mul(gain).backward()

            if self.depth_distill:
                gen_img, _gen_ws, c_gen_conditioning = self.run_G(gen_z, gen_c, swapping_prob=swapping_prob, neural_rendering_resolution=neural_rendering_resolution)
                ws = self.guide_G.mapping(gen_z, c_gen_conditioning, update_emas=False)
                guide_img = self.guide_G.synthesis(ws, gen_c)
                # print(guide_img["image_depth"].max(), guide_img["image_depth"].min(), gen_img["image_depth"].max(), gen_img["image_depth"].min())
                # print(gen_img["image_depth"].shape, guide_img["image_depth"].shape,gen_img["image"].shape, guide_img["image"][:,:,::4, ::4].shape)
                # exit(0)
                loss_guide = torch.nn.functional.l1_loss(gen_img["image_depth"], guide_img["image_depth"]) # + torch.nn.functional.l1_loss(gen_img["image"], guide_img["image"][:,:,::4, ::4])
                training_stats.report('Loss/G/loss_guide', loss_guide)
                loss_guide.mul(gain).backward()
                
            #     ## FIXME: debug grad
            #     print(f"Gradients for self.G -----begin---")
            #     for name, param in self.G.named_parameters():
            #         # print(f" {name}") # not including gaussian and minicam
            #         if param.grad is not None:
            #             print(f"Gradients for {name} have been computed.")
            #         else:
            #             pass
            #             # print(f"Gradients for {name} have NOT been computed!!")
            #     print(f"Gradients for self.G -----end---")
            # st()
                
        # if phase in ['Greg', 'Gboth'] and self.G.rendering_kwargs['reg_type'] == 'l1': # opacity reg
        #     gen_img, _, _ = self.run_G(gen_z, gen_c, swapping_prob=swapping_prob, neural_rendering_resolution=neural_rendering_resolution)
        #     alpha_img = gen_img['image_mask']
        #     alpha_loss = torch.nn.functional.l1_loss(alpha_img, torch.ones_like(alpha_img, device=alpha_img.device)) * self.G.rendering_kwargs['opacity_reg']
        #     training_stats.report('Loss/G/loss_alpha', alpha_loss)
        #     alpha_loss.mul(gain).backward()
            
        # # Density Regularization
        # if phase in ['Greg', 'Gboth'] and self.G.rendering_kwargs.get('density_reg', 0) > 0 and self.G.rendering_kwargs['reg_type'] == 'l1':
        #     if swapping_prob is not None:
        #         c_swapped = torch.roll(gen_c.clone(), 1, 0)
        #         c_gen_conditioning = torch.where(torch.rand([], device=gen_c.device) < swapping_prob, c_swapped, gen_c)
        #     else:
        #         c_gen_conditioning = torch.zeros_like(gen_c)

        #     ws = self.G.mapping(gen_z, c_gen_conditioning, update_emas=False)
        #     if self.style_mixing_prob > 0:
        #         with torch.autograd.profiler.record_function('style_mixing'):
        #             cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
        #             cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
        #             ws[:, cutoff:] = self.G.mapping(torch.randn_like(z), c, update_emas=False)[:, cutoff:]
        #     initial_coordinates = torch.rand((ws.shape[0], 1000, 3), device=ws.device) * 2 - 1
        #     perturbed_coordinates = initial_coordinates + torch.randn_like(initial_coordinates) * self.G.rendering_kwargs['density_reg_p_dist']
        #     all_coordinates = torch.cat([initial_coordinates, perturbed_coordinates], dim=1)
        #     sigma = self.G.sample_mixed(all_coordinates, torch.randn_like(all_coordinates), ws, update_emas=False)['sigma']
        #     sigma_initial = sigma[:, :sigma.shape[1]//2]
        #     sigma_perturbed = sigma[:, sigma.shape[1]//2:]

        #     TVloss = torch.nn.functional.l1_loss(sigma_initial, sigma_perturbed) * self.G.rendering_kwargs['density_reg']
        #     TVloss.mul(gain).backward()

        # # Alternative density regularization
        # if phase in ['Greg', 'Gboth'] and self.G.rendering_kwargs.get('density_reg', 0) > 0 and self.G.rendering_kwargs['reg_type'] == 'monotonic-detach':
        #     if swapping_prob is not None:
        #         c_swapped = torch.roll(gen_c.clone(), 1, 0)
        #         c_gen_conditioning = torch.where(torch.rand([], device=gen_c.device) < swapping_prob, c_swapped, gen_c)
        #     else:
        #         c_gen_conditioning = torch.zeros_like(gen_c)

        #     ws = self.G.mapping(gen_z, c_gen_conditioning, update_emas=False)

        #     initial_coordinates = torch.rand((ws.shape[0], 2000, 3), device=ws.device) * 2 - 1 # Front

        #     perturbed_coordinates = initial_coordinates + torch.tensor([0, 0, -1], device=ws.device) * (1/256) * self.G.rendering_kwargs['box_warp'] # Behind
        #     all_coordinates = torch.cat([initial_coordinates, perturbed_coordinates], dim=1)
        #     sigma = self.G.sample_mixed(all_coordinates, torch.randn_like(all_coordinates), ws, update_emas=False)['sigma']
        #     sigma_initial = sigma[:, :sigma.shape[1]//2]
        #     sigma_perturbed = sigma[:, sigma.shape[1]//2:]

        #     monotonic_loss = torch.relu(sigma_initial.detach() - sigma_perturbed).mean() * 10
        #     monotonic_loss.mul(gain).backward()


        #     if swapping_prob is not None:
        #         c_swapped = torch.roll(gen_c.clone(), 1, 0)
        #         c_gen_conditioning = torch.where(torch.rand([], device=gen_c.device) < swapping_prob, c_swapped, gen_c)
        #     else:
        #         c_gen_conditioning = torch.zeros_like(gen_c)

        #     ws = self.G.mapping(gen_z, c_gen_conditioning, update_emas=False)
        #     if self.style_mixing_prob > 0:
        #         with torch.autograd.profiler.record_function('style_mixing'):
        #             cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
        #             cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
        #             ws[:, cutoff:] = self.G.mapping(torch.randn_like(z), c, update_emas=False)[:, cutoff:]
        #     initial_coordinates = torch.rand((ws.shape[0], 1000, 3), device=ws.device) * 2 - 1
        #     perturbed_coordinates = initial_coordinates + torch.randn_like(initial_coordinates) * (1/256) * self.G.rendering_kwargs['box_warp']
        #     all_coordinates = torch.cat([initial_coordinates, perturbed_coordinates], dim=1)
        #     sigma = self.G.sample_mixed(all_coordinates, torch.randn_like(all_coordinates), ws, update_emas=False)['sigma']
        #     sigma_initial = sigma[:, :sigma.shape[1]//2]
        #     sigma_perturbed = sigma[:, sigma.shape[1]//2:]

        #     TVloss = torch.nn.functional.l1_loss(sigma_initial, sigma_perturbed) * self.G.rendering_kwargs['density_reg']
        #     TVloss.mul(gain).backward()

        # # Alternative density regularization
        # if phase in ['Greg', 'Gboth'] and self.G.rendering_kwargs.get('density_reg', 0) > 0 and self.G.rendering_kwargs['reg_type'] == 'monotonic-fixed':
        #     if swapping_prob is not None:
        #         c_swapped = torch.roll(gen_c.clone(), 1, 0)
        #         c_gen_conditioning = torch.where(torch.rand([], device=gen_c.device) < swapping_prob, c_swapped, gen_c)
        #     else:
        #         c_gen_conditioning = torch.zeros_like(gen_c)

        #     ws = self.G.mapping(gen_z, c_gen_conditioning, update_emas=False)

        #     initial_coordinates = torch.rand((ws.shape[0], 2000, 3), device=ws.device) * 2 - 1 # Front

        #     perturbed_coordinates = initial_coordinates + torch.tensor([0, 0, -1], device=ws.device) * (1/256) * self.G.rendering_kwargs['box_warp'] # Behind
        #     all_coordinates = torch.cat([initial_coordinates, perturbed_coordinates], dim=1)
        #     sigma = self.G.sample_mixed(all_coordinates, torch.randn_like(all_coordinates), ws, update_emas=False)['sigma']
        #     sigma_initial = sigma[:, :sigma.shape[1]//2]
        #     sigma_perturbed = sigma[:, sigma.shape[1]//2:]

        #     monotonic_loss = torch.relu(sigma_initial - sigma_perturbed).mean() * 10
        #     monotonic_loss.mul(gain).backward()


        #     if swapping_prob is not None:
        #         c_swapped = torch.roll(gen_c.clone(), 1, 0)
        #         c_gen_conditioning = torch.where(torch.rand([], device=gen_c.device) < swapping_prob, c_swapped, gen_c)
        #     else:
        #         c_gen_conditioning = torch.zeros_like(gen_c)

        #     ws = self.G.mapping(gen_z, c_gen_conditioning, update_emas=False)
        #     if self.style_mixing_prob > 0:
        #         with torch.autograd.profiler.record_function('style_mixing'):
        #             cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
        #             cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
        #             ws[:, cutoff:] = self.G.mapping(torch.randn_like(z), c, update_emas=False)[:, cutoff:]
        #     initial_coordinates = torch.rand((ws.shape[0], 1000, 3), device=ws.device) * 2 - 1
        #     perturbed_coordinates = initial_coordinates + torch.randn_like(initial_coordinates) * (1/256) * self.G.rendering_kwargs['box_warp']
        #     all_coordinates = torch.cat([initial_coordinates, perturbed_coordinates], dim=1)
        #     sigma = self.G.sample_mixed(all_coordinates, torch.randn_like(all_coordinates), ws, update_emas=False)['sigma']
        #     sigma_initial = sigma[:, :sigma.shape[1]//2]
        #     sigma_perturbed = sigma[:, sigma.shape[1]//2:]

        #     TVloss = torch.nn.functional.l1_loss(sigma_initial, sigma_perturbed) * self.G.rendering_kwargs['density_reg']
        #     TVloss.mul(gain).backward()

        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        if phase in ['Dmain', 'Dboth']:
            with torch.autograd.profiler.record_function('Dgen_forward'):
                gen_img, _gen_ws, _ = self.run_G(gen_z, gen_c, swapping_prob=swapping_prob, neural_rendering_resolution=neural_rendering_resolution, update_emas=True)
                gen_logits = self.run_D(gen_img, gen_c, blur_sigma=blur_sigma, update_emas=True)
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Dgen = torch.nn.functional.softplus(gen_logits)
            with torch.autograd.profiler.record_function('Dgen_backward'):
                loss_Dgen.mean().mul(gain).backward()

        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if phase in ['Dmain', 'Dreg', 'Dboth']:
            name = 'Dreal' if phase == 'Dmain' else 'Dr1' if phase == 'Dreg' else 'Dreal_Dr1'
            with torch.autograd.profiler.record_function(name + '_forward'):
                real_img_tmp_image = real_img['image'].detach().requires_grad_(phase in ['Dreg', 'Dboth'])
                real_img_tmp_image_raw = real_img['image_raw'].detach().requires_grad_(phase in ['Dreg', 'Dboth'])
                real_img_tmp_image_edge = real_img['image_edge'].detach().requires_grad_(phase in ['Dreg', 'Dboth'])
                if self.use_segmentation:
                    real_img_tmp_image_mask = real_img['image_mask'].detach().requires_grad_(phase in ['Dreg', 'Dboth']) # [N, C, H, W]
                # # print(real_img['image_mask'].max(1))
                # for i in range(8):
                #     pred = real_img['image_mask'].max(1)[1].detach().cpu().numpy()[i]
                #     # print(pred.shape)
                #     # print(pred.max())
                #     colorized_preds = self.decode_fn(pred).astype('uint8')
                #     print(colorized_preds.max())
                #     # print(pred.shape)
                #     colorized_preds = Image.fromarray(pred, 'L')
                #     colorized_preds.save(f'/home/zxy/eg3d/eg3d/prediction_{i}.png')

                # exit(0)
                if self.use_segmentation:
                    real_img_tmp = {'image': real_img_tmp_image, 'image_raw': real_img_tmp_image_raw, 'image_edge': real_img_tmp_image_edge, 'image_mask': real_img_tmp_image_mask}
                else:
                    real_img_tmp = {'image': real_img_tmp_image, 'image_raw': real_img_tmp_image_raw, 'image_edge': real_img_tmp_image_edge}
                    
                real_logits = self.run_D(real_img_tmp, real_c, blur_sigma=blur_sigma)
                training_stats.report('Loss/scores/real', real_logits)
                training_stats.report('Loss/signs/real', real_logits.sign())

                loss_Dreal = 0
                if phase in ['Dmain', 'Dboth']:
                    loss_Dreal = torch.nn.functional.softplus(-real_logits)
                    training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)

                loss_Dr1 = 0
                if phase in ['Dreg', 'Dboth']:
                    if self.dual_discrimination:
                        with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                            r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp['image'], real_img_tmp['image_edge']], create_graph=True, only_inputs=True)
                            r1_grads_image = r1_grads[0]
                            r1_grads_image_raw = r1_grads[1]
                        r1_penalty = r1_grads_image.square().sum([1,2,3]) + r1_grads_image_raw.square().sum([1,2,3])
                    else: # single discrimination
                        with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                            r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp['image']], create_graph=True, only_inputs=True)
                            r1_grads_image = r1_grads[0]
                        r1_penalty = r1_grads_image.square().sum([1,2,3])
                    loss_Dr1 = r1_penalty * (r1_gamma / 2)
                    training_stats.report('Loss/r1_penalty', r1_penalty)
                    training_stats.report('Loss/D/reg', loss_Dr1)

            with torch.autograd.profiler.record_function(name + '_backward'):
                (loss_Dreal + loss_Dr1).mean().mul(gain).backward()

#----------------------------------------------------------------------------
