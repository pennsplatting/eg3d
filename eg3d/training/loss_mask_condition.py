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

from pdb import set_trace as st

#----------------------------------------------------------------------------

class Loss:
    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, gain, cur_nimg): # to be overridden by subclass
        raise NotImplementedError()

#----------------------------------------------------------------------------

class StyleGAN2Loss(Loss):
    def __init__(self, device, G, D, augment_pipe=None, r1_gamma=10, style_mixing_prob=0, pl_weight=0, pl_batch_shrink=2, pl_decay=0.01, pl_no_weight_grad=False, blur_init_sigma=0, 
                 blur_fade_kimg=0, r1_gamma_init=0, r1_gamma_fade_kimg=0, neural_rendering_resolution_initial=64, neural_rendering_resolution_final=None, neural_rendering_resolution_fade_kimg=0, gpc_reg_fade_kimg=1000, gpc_reg_prob=None, dual_discrimination=False, filter_mode='antialiased',
                 decode_first='all', reg_weight=0.1, opacity_reg= 1, l1_loss_reg = True, 
                 ref_scale=None, clamp=True, use_mask=False):
        super().__init__()
        self.device             = device
        self.G                  = G
        self.D                  = D
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
        # ---- new for mhg ------
        self.decode_first = decode_first
        self.reg_weight = reg_weight
        self.opacity_reg = opacity_reg
        self.l1_loss_reg = l1_loss_reg
        self.mask_image = None
        self.clamp = clamp
        self.use_mask = use_mask # the silhouettes mask.
        
        with torch.no_grad():
            if decode_first == 'all' or decode_first == 'wo_color':
                self.ref_scale = ref_scale or -5.0 # -4.5  #4.5 # -5.2902 #-5.2902 # -3.5616 #
            else:
                self.ref_scale = ref_scale or 0.0001
        # -----------------------

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
        return gen_output, ws

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
            if self.use_mask:
                img['image_mask'] = augmented_pair[:, img['image'].shape[1]:]
                
            img['image_raw'] = torch.nn.functional.interpolate(augmented_pair[:, img['image'].shape[1]:], size=img['image_raw'].shape[2:], mode='bilinear', antialias=True)

        logits = self.D(img, c, update_emas=update_emas)
        return logits
    
    
    # --------------- new for MHG ---------------
    def loss_opacity(self, opacity_map, texture_mask):
        """Inside the masked region of the texture, opacity should sum to 1.
        Args:
            texture_mask: (bs, 1, h, w) opacity after activation
            opacity_map: (bs, sh, h, w)
        returns:
            loss: float
        """
        texture_mask = texture_mask.to(opacity_map.device)
        texture_mask = filtered_resizing(texture_mask.unsqueeze(0), size=opacity_map.shape[-2], f=self.resample_filter, filter_mode=self.filter_mode).squeeze(0).repeat(opacity_map.shape[0], 1, 1)

        loss_map = torch.abs(torch.sum(opacity_map.squeeze(-1), dim=1, keepdim=False) - 1) * texture_mask
        return torch.sum(loss_map) / torch.sum(texture_mask)
    
    def loss_clamp_l2(self, source, target, mask=None, clamp=True):
        """
        Args:
            source: (bs, sh, h, w, c)
            target: float value
            mask: (bs, 1, h, w)
        Returns:
            float
        """
        if clamp:
            loss_map = torch.clamp((source - target), min=0)**2
        else:
            loss_map = (source - target)**2
        if mask is not None:
            mask = mask.to(source.device)
            texture_mask = filtered_resizing(mask.unsqueeze(0), size=source.shape[-2], f=self.resample_filter, filter_mode=self.filter_mode).repeat(source.shape[0], source.shape[1], 1, 1)
            return torch.sum(loss_map * texture_mask[..., None])/ torch.sum(texture_mask)
        else:
            return torch.mean(loss_map)

    def loss_clamp_l1(self, source, target_value, mask=None, clamp=False):
        """
        Args:
            source: (bs, sh, h, w, c)
            target_value: float value
            mask: (1, h, w)
        Returns:
            loss: float
        """
        if clamp:
            loss_map = torch.abs(torch.clamp(source - target_value, min=0))
        else:
            loss_map = torch.abs(source - target_value)
        if mask is not None:
            mask = mask.to(source.device)
            texture_mask = filtered_resizing(mask.unsqueeze(0), size=source.shape[-2], f=self.resample_filter, filter_mode=self.filter_mode).repeat(source.shape[0], source.shape[1], 1, 1)
            return torch.sum(loss_map * texture_mask[...,None]) / torch.sum(texture_mask)
        else:
            return torch.mean(loss_map)
    
    def get_real_img_mask(self, real_img):
        real_img_mean = real_img.mean(dim=1, keepdim=True)
        real_img_mask = (real_img_mean==real_img_mean.max()).to(real_img.device, real_img.dtype) # {0-fg,1-bg}
        real_img_mask = (real_img_mask - 0.5) * 2 # 0~1 map to -1~1
        return real_img_mask
        
    # --------------------------------------------
    
    
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
            gen_img, _gen_ws = self.run_G(gen_z, gen_c, swapping_prob=swapping_prob, neural_rendering_resolution=neural_rendering_resolution)
            loss = torch.nn.functional.mse_loss(gen_img["image_real"], gen_img["image"])
            
        with torch.autograd.profiler.record_function('Gmain_backward'):
            loss.backward()

        
    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, gain, cur_nimg):
        
        # if self.progressive_scale_reg_kimg > 0:
        #     reg_weight_cur = self.reg_weight - min(cur_nimg / (self.progressive_scale_reg_kimg * 1e3), 1) * (self.reg_weight-self.progressive_scale_reg_end)
        # else:
        #     reg_weight_cur = self.reg_weight
        reg_weight_cur = self.reg_weight #FIXME: not using progressive training for now
            
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
            st()
            alpha = min(cur_nimg / (self.neural_rendering_resolution_fade_kimg * 1e3), 1)
            neural_rendering_resolution = int(np.rint(self.neural_rendering_resolution_initial * (1 - alpha) + self.neural_rendering_resolution_final * alpha))
        else:
            neural_rendering_resolution = self.neural_rendering_resolution_initial

        real_img_raw = filtered_resizing(real_img, size=neural_rendering_resolution, f=self.resample_filter, filter_mode=self.filter_mode) # torch.Size([4, 3, 512, 512])
        
        if self.blur_raw_target:
            blur_size = np.floor(blur_sigma * 3)
            if blur_size > 0:
                f = torch.arange(-blur_size, blur_size + 1, device=real_img_raw.device).div(blur_sigma).square().neg().exp2()
                real_img_raw = upfirdn2d.filter2d(real_img_raw, f / f.sum())

                
        if self.use_mask:
            real_img_mask = self.get_real_img_mask(real_img)
            real_img = {'image': real_img, 'image_raw': real_img_raw, 'image_mask': real_img_mask}
        
        else:        
            real_img = {'image': real_img, 'image_raw': real_img_raw} # no loss is calculated based on depth 
        
        

        # Gmain: Maximize logits for generated images.
        if phase in ['Gmain', 'Gboth']:
            with torch.autograd.profiler.record_function('Gmain_forward'):
                gen_img, _gen_ws = self.run_G(gen_z, gen_c, swapping_prob=swapping_prob, neural_rendering_resolution=neural_rendering_resolution)
                
                gen_logits = self.run_D(gen_img, gen_c, blur_sigma=blur_sigma)
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Gmain = torch.nn.functional.softplus(-gen_logits)
                training_stats.report('Loss/G/loss', loss_Gmain)
                
                # -------------------- new for MHG: opacity and scale reg ---------------------
                if self.reg_weight > 0:
                    if self.decode_first == 'all' or self.decode_first == 'wo_color':
                        if self.l1_loss_reg:
                            scaling_loss = self.loss_clamp_l1(gen_img['scaling'], self.ref_scale, mask=self.mask_image, clamp=self.clamp)
                        else:
                            scaling_loss = self.loss_clamp_l2(gen_img['scaling'], self.ref_scale, mask=self.mask_image, clamp=self.clamp)

                        training_stats.report('Loss/G/loss_reg_scale', scaling_loss)
                        training_stats.report('Loss/G/scaling_max', torch.max(gen_img['scaling']))
                    else:
                        if self.l1_loss_reg:
                            scaling_loss = self.loss_clamp_l1(gen_img['scaling'], self.ref_scale, clamp=self.clamp)
                        else:
                            scaling_loss = self.loss_clamp_l2(gen_img['scaling'], self.ref_scale, clamp=self.clamp)

                        training_stats.report('Loss/G/loss_reg_scale', scaling_loss)
                        training_stats.report('Loss/G/scaling_max', torch.max(gen_img['scaling']))


                if self.opacity_reg > 0 and self.mask_image is not None:
                    opacity_loss = self.loss_opacity(gen_img['opacity'], self.mask_image).to(_gen_ws.device)
                    training_stats.report('Loss/G/opacity_loss', opacity_loss)
                    
                # --------------- -------------- -------------- --------------
            
            with torch.autograd.profiler.record_function('Gmain_backward'):

                if self.reg_weight > 0:
                    loss_Gmain = loss_Gmain + reg_weight_cur*scaling_loss
                if self.opacity_reg > 0 and self.mask_image is not None:
                    loss_Gmain = loss_Gmain + self.opacity_reg*opacity_loss

                loss_Gmain.mean().mul(gain).backward()
            
            # TODO: add density and scale regularization
            # with torch.autograd.profiler.record_function('Gmain_backward'):
            #     if self.use_face_dist:
            #         loss_Gmain = loss_Gmain + loss_face_Gmain * self.face_weight
            #     if self.use_hand_dist:
            #         loss_Gmain = loss_Gmain + loss_hand_Gmain.mean() * self.hand_foot_weight
            #     if self.use_foot_dist:
            #         loss_Gmain = loss_Gmain + loss_foot_Gmain.mean() * self.hand_foot_weight
            #     if self.use_patch_dist:
            #         loss_Gmain = loss_Gmain + loss_patch_Gmain
            #     if self.reg_weight > 0:
            #         loss_Gmain = loss_Gmain + reg_weight_cur*scaling_loss
            #     if self.opacity_reg > 0 and self.mask_image is not None:
            #         loss_Gmain = loss_Gmain + self.opacity_reg*opacity_loss

            #     loss_Gmain.mean().mul(gain).backward()
                
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
                

        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        if phase in ['Dmain', 'Dboth']:
            with torch.autograd.profiler.record_function('Dgen_forward'):
                gen_img, _gen_ws = self.run_G(gen_z, gen_c, swapping_prob=swapping_prob, neural_rendering_resolution=neural_rendering_resolution, update_emas=True)
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
                if self.use_mask:
                    real_img_tmp_image_mask = real_img['image_mask'].detach().requires_grad_(phase in ['Dreg', 'Dboth'])
                    real_img_tmp = {'image': real_img_tmp_image, 'image_raw': real_img_tmp_image_raw, 'image_mask': real_img_tmp_image_mask}
                else:   
                    real_img_tmp = {'image': real_img_tmp_image, 'image_raw': real_img_tmp_image_raw}

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
                            # r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp['image'], real_img_tmp['image_raw']], create_graph=True, only_inputs=True)
                            r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp['image'], real_img_tmp['image_mask']], create_graph=True, only_inputs=True)
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

    def vis_parsing_maps(im, parsing_anno, stride, save_im=False, save_path='vis_results/parsing_map_on_im.jpg'):
        # Colors for all 20 parts
        atts = ['skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g', 'l_ear', 'r_ear', 'ear_r',
                    'nose', 'mouth', 'u_lip', 'l_lip', 'neck', 'neck_l', 'cloth', 'hair', 'hat',
                    'background']
        
        # atts = [1 'skin', 2 'l_brow', 3 'r_brow', 4 'l_eye', 5 'r_eye', 6 'eye_g', 7 'l_ear', 8 'r_ear', 9 'ear_r',
        #                 10 'nose', 11 'mouth', 12 'u_lip', 13 'l_lip', 14 'neck', 15 'neck_l', 16 'cloth', 17 'hair', 18 'hat']
        use_part_colors_dicts = True
        if not use_part_colors_dicts:
            part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
                        [255, 0, 85], [255, 0, 170],
                        [0, 255, 0], [85, 255, 0], [170, 255, 0],
                        [0, 255, 85], [0, 255, 170],
                        [0, 0, 255], [85, 0, 255], [170, 0, 255],
                        [0, 85, 255], [0, 170, 255],
                        [255, 255, 0], [255, 255, 85], [255, 255, 170],
                        [255, 0, 255], [255, 85, 255], [255, 170, 255],
                        [0, 255, 255], [85, 255, 255], [170, 255, 255]]
            part_colors_dict = {atts[i] if i < (len(atts)) else f'others_{i}':part_colors[i] for i in range(len(part_colors))}
        
        else:
            ## the part_colors_dict is derived from the above lines of codes
            part_colors_dict = {'skin': [255, 0, 0], 'l_brow': [255, 85, 0], 'r_brow': [255, 170, 0], 'l_eye': [255, 0, 85], 'r_eye': [255, 0, 170], 'eye_g': [0, 255, 0], 'l_ear': [85, 255, 0], 'r_ear': [170, 255, 0], 'ear_r': [0, 255, 85], 'nose': [0, 255, 170], 'mouth': [0, 0, 255], 'u_lip': [85, 0, 255], 'l_lip': [170, 0, 255], 'neck': [0, 85, 255], 'neck_l': [0, 170, 255], 'cloth': [255, 255, 0], 'hair': [255, 255, 85], 'hat': [255, 255, 170], 'background': [255, 0, 255], 'others_19': [255, 85, 255], 'others_20': [255, 170, 255], 'others_21': [0, 255, 255], 'others_22': [85, 255, 255], 'others_23': [170, 255, 255]}
            ## group the color of parts
            face_parts = ['skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g', 'l_ear', 'r_ear', 'ear_r',
                    'nose', 'mouth', 'u_lip', 'l_lip','neck'] # this neck is actually lip!! the neck_l is the real neck
            face_parts_color = part_colors_dict['skin']
            for fp in face_parts:
                part_colors_dict[fp]=face_parts_color
            

        im = np.array(im)
        vis_im = im.copy().astype(np.uint8)
        vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
        vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
        vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255

        num_of_class = np.max(vis_parsing_anno)

        for pi in range(1, num_of_class + 1):
            index = np.where(vis_parsing_anno == pi)
            # assert part_colors[pi] == part_colors_dict[atts[pi]]
            if not use_part_colors_dicts:
                vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]
            else:
                vis_parsing_anno_color[index[0], index[1], :] = part_colors_dict[atts[pi]]

        vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
        # print(vis_parsing_anno_color.shape, vis_im.shape)
        vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.4, vis_parsing_anno_color, 0.6, 0)

        # Save result or not
        if save_im:
            cv2.imwrite(save_path[:-4] +'.png', vis_parsing_anno)
            print(f"Save path:{save_path}")
            cv2.imwrite(save_path, vis_im, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

        # return vis_im

    def accumulate_gradients_mask_real_face(self, phase, real_img, real_c, gen_z, gen_c, gain, cur_nimg):
            # the real_img is already masked, keeping only the facial area
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

           
            real_img_raw = filtered_resizing(real_img, size=neural_rendering_resolution, f=self.resample_filter, filter_mode=self.filter_mode)

            if self.blur_raw_target:
                blur_size = np.floor(blur_sigma * 3)
                if blur_size > 0:
                    f = torch.arange(-blur_size, blur_size + 1, device=real_img_raw.device).div(blur_sigma).square().neg().exp2()
                    real_img_raw = upfirdn2d.filter2d(real_img_raw, f / f.sum())

            real_img = {'image': real_img, 'image_raw': real_img_raw} # no loss is calculated based on depth 

            # Gmain: Maximize logits for generated images.
            if phase in ['Gmain', 'Gboth']:
                with torch.autograd.profiler.record_function('Gmain_forward'):
                    gen_img, _gen_ws = self.run_G(gen_z, gen_c, swapping_prob=swapping_prob, neural_rendering_resolution=neural_rendering_resolution)
                    gen_logits = self.run_D(gen_img, gen_c, blur_sigma=blur_sigma)
                    training_stats.report('Loss/scores/fake', gen_logits)
                    training_stats.report('Loss/signs/fake', gen_logits.sign())
                    loss_Gmain = torch.nn.functional.softplus(-gen_logits)
                    training_stats.report('Loss/G/loss', loss_Gmain)
                with torch.autograd.profiler.record_function('Gmain_backward'):
                    loss_Gmain.mean().mul(gain).backward()
                    
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
                    
            # Dmain: Minimize logits for generated images.
            loss_Dgen = 0
            if phase in ['Dmain', 'Dboth']:
                with torch.autograd.profiler.record_function('Dgen_forward'):
                    gen_img, _gen_ws = self.run_G(gen_z, gen_c, swapping_prob=swapping_prob, neural_rendering_resolution=neural_rendering_resolution, update_emas=True)
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
                    real_img_tmp = {'image': real_img_tmp_image, 'image_raw': real_img_tmp_image_raw}

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
                                r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp['image'], real_img_tmp['image_raw']], create_graph=True, only_inputs=True)
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