# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Main training loop."""

import os
import time
import copy
import json
import pickle
import psutil
import PIL.Image
import numpy as np
import torch
import dnnlib
from torch_utils import misc
from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import grid_sample_gradfix

import legacy
from metrics import metric_main
from camera_utils import LookAtPoseSampler
from training.crosssection_utils import sample_cross_section

from ipdb import set_trace as st

from training.face_parsing_model import BiSeNet
import cv2
import os.path as osp
from PIL import Image
import torchvision.transforms as transforms
from training.other_utils import get_optimizer_parameters

#----------------------------------------------------------------------------

def setup_snapshot_image_grid(training_set, random_seed=0):
    rnd = np.random.RandomState(random_seed)
    gw = np.clip(7680 // training_set.image_shape[2], 7, 32)
    gh = np.clip(4320 // training_set.image_shape[1], 4, 32)

    # No labels => show random subset of training samples.
    if not training_set.has_labels:
        all_indices = list(range(len(training_set)))
        rnd.shuffle(all_indices)
        grid_indices = [all_indices[i % len(all_indices)] for i in range(gw * gh)]

    else:
        # Group training samples by label.
        label_groups = dict() # label => [idx, ...]
        for idx in range(len(training_set)):
            label = tuple(training_set.get_details(idx).raw_label.flat[::-1])
            if label not in label_groups:
                label_groups[label] = []
            label_groups[label].append(idx)

        # Reorder.
        label_order = list(label_groups.keys())
        rnd.shuffle(label_order)
        for label in label_order:
            rnd.shuffle(label_groups[label])

        # Organize into grid.
        grid_indices = []
        for y in range(gh):
            label = label_order[y % len(label_order)]
            indices = label_groups[label]
            grid_indices += [indices[x % len(indices)] for x in range(gw)]
            label_groups[label] = [indices[(i + gw) % len(indices)] for i in range(len(indices))]

    # Load data.
    images, labels = zip(*[training_set[i] for i in grid_indices])
    return (gw, gh), np.stack(images), np.stack(labels)

#----------------------------------------------------------------------------

def save_image_grid(img, fname, drange, grid_size):
    lo, hi = drange
    img = np.asarray(img, dtype=np.float32)
    img = (img - lo) * (255 / (hi - lo))
    img = np.rint(img).clip(0, 255).astype(np.uint8)

    gw, gh = grid_size
    _N, C, H, W = img.shape
    img = img.reshape([gh, gw, C, H, W])
    img = img.transpose(0, 3, 1, 4, 2)
    img = img.reshape([gh * H, gw * W, C])

    assert C in [1, 3]
    if C == 1:
        PIL.Image.fromarray(img[:, :, 0], 'L').save(fname)
    if C == 3:
        PIL.Image.fromarray(img, 'RGB').save(fname)
        
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
                'nose', 'mouth', 'u_lip', 'l_lip','neck', 'neck_l'] # this neck is actually lip!! the neck_l is the real neck
        face_parts_color = part_colors_dict['skin']
        for fp in face_parts:
            part_colors_dict[fp]=face_parts_color
        

    # ... (existing code)
    im = np.array(im) # im: PIL image
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

    # mask = (vis_parsing_anno_color==face_parts_color)
    mask = np.all(vis_parsing_anno_color == np.array(face_parts_color), axis=-1)

    return torch.from_numpy(mask)
    
    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    # print(vis_parsing_anno_color.shape, vis_im.shape)
    vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.4, vis_parsing_anno_color, 0.6, 0)

    # Save result or not
    if save_im:
        cv2.imwrite(save_path[:-4] +'.png', vis_parsing_anno)
        print(f"Save path:{save_path}")
        cv2.imwrite(save_path, vis_im, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    
    vis_im_tensor = torch.from_numpy(vis_im.astype(im.dtype)) #.to(im_tensor.device)

    return torch.from_numpy(mask)

#----------------------------------------------------------------------------
def get_face_mask(real_img, type='torch'):
    
    if isinstance(real_img, np.ndarray):
        type='numpy'
        original_dtype = real_img.dtype
        real_img=torch.tensor(real_img).cuda().float()
    
    bg_color = 255 if real_img.max().item() > 1 else 1 # white bg as GS 
    ## TODO: unify the bg parameters with GS
    
  
    # Define the normalization parameters, applies to image of range in both [-1,1] and [0,255]
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()
    
    # Normalize the entire batch in a vectorized manner
    normalized_batch = (real_img/real_img.max() - mean) / std
  
    
    ##### use segmentation mask to keep only the facial part of the real image
    with torch.no_grad():
        real_img_masked_batch = []
        for _img in normalized_batch:
            
            img = torch.unsqueeze(_img, 0) # img: [1, 3, 512, 512]
            
            out = net(img)[0] # out: [1, 19, 512, 512]
            parsing = out.squeeze(0).cpu().numpy().argmax(0) # parsing: (512, 512)
            # print(np.unique(parsing))
          
            tensor_to_pil = transforms.ToPILImage()
            _img = tensor_to_pil(_img)          

            # img_save_pth = os.path.join(run_dir, f'reals_masked{cur_nimg//1000:06d}.png')
            # img_save_pth = f'reals_masked{1000:06d}.png'
            img_save_pth = None
            real_img_masked = vis_parsing_maps(_img, parsing, stride=1, save_im=True, save_path=img_save_pth)
            
            real_img_masked_batch.append(real_img_masked.unsqueeze(0))
    real_img_mask_batch = torch.cat(real_img_masked_batch).to(real_img.device).to(torch.float)[:,None]
    real_img_masked_batch = real_img * real_img_mask_batch + bg_color * (1-real_img_mask_batch)
    assert real_img_masked_batch.shape == real_img.shape
    
    if type=='numpy':
        # real_img=real_img.cpu().numpy().astype(original_dtype)
        real_img_masked_batch = real_img_masked_batch.cpu().numpy().astype(original_dtype)
        
    return real_img_masked_batch
    
    

#----------------------------------------------------------------------------

def training_loop(
    run_dir                 = '.',      # Output directory.
    training_set_kwargs     = {},       # Options for training set.
    data_loader_kwargs      = {},       # Options for torch.utils.data.DataLoader.
    G_kwargs                = {},       # Options for generator network.
    D_kwargs                = {},       # Options for discriminator network.
    G_opt_kwargs            = {},       # Options for generator optimizer.
    D_opt_kwargs            = {},       # Options for discriminator optimizer.
    augment_kwargs          = None,     # Options for augmentation pipeline. None = disable.
    loss_kwargs             = {},       # Options for loss function.
    metrics                 = [],       # Metrics to evaluate during training.
    random_seed             = 0,        # Global random seed.
    num_gpus                = 1,        # Number of GPUs participating in the training.
    rank                    = 0,        # Rank of the current process in [0, num_gpus[.
    batch_size              = 4,        # Total batch size for one training iteration. Can be larger than batch_gpu * num_gpus.
    batch_gpu               = 4,        # Number of samples processed at a time by one GPU.
    ema_kimg                = 10,       # Half-life of the exponential moving average (EMA) of generator weights.
    ema_rampup              = 0.05,     # EMA ramp-up coefficient. None = no rampup.
    G_reg_interval          = None,     # How often to perform regularization for G? None = disable lazy regularization.
    D_reg_interval          = 16,       # How often to perform regularization for D? None = disable lazy regularization.
    augment_p               = 0,        # Initial value of augmentation probability.
    ada_target              = None,     # ADA target value. None = fixed p.
    ada_interval            = 4,        # How often to perform ADA adjustment?
    ada_kimg                = 500,      # ADA adjustment speed, measured in how many kimg it takes for p to increase/decrease by one unit.
    total_kimg              = 25000,    # Total length of the training, measured in thousands of real images.
    kimg_per_tick           = 4,        # Progress snapshot interval.
    image_snapshot_ticks    = 50,       # How often to save image snapshots? None = disable.
    network_snapshot_ticks  = 50,       # How often to save network snapshots? None = disable.
    resume_pkl              = None,     # Network pickle to resume training from.
    resume_kimg             = 0,        # First kimg to report when resuming training.
    cudnn_benchmark         = True,     # Enable torch.backends.cudnn.benchmark?
    abort_fn                = None,     # Callback function for determining whether to abort training. Must return consistent results across ranks.
    progress_fn             = None,     # Callback function for updating training progress. Called for all ranks.
):
    # Initialize.
    start_time = time.time()
    device = torch.device('cuda', rank)
    np.random.seed(random_seed * num_gpus + rank)
    torch.manual_seed(random_seed * num_gpus + rank)
    torch.backends.cudnn.benchmark = cudnn_benchmark    # Improves training speed.
    torch.backends.cuda.matmul.allow_tf32 = False       # Improves numerical accuracy.
    torch.backends.cudnn.allow_tf32 = False             # Improves numerical accuracy.
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False  # Improves numerical accuracy.
    conv2d_gradfix.enabled = True                       # Improves training speed. # TODO: ENABLE
    grid_sample_gradfix.enabled = False                  # Avoids errors with the augmentation pipe.

    # Load training set.
    if rank == 0:
        print('Loading training set...')
    training_set = dnnlib.util.construct_class_by_name(**training_set_kwargs) # subclass of training.dataset.Dataset
    training_set_sampler = misc.InfiniteSampler(dataset=training_set, rank=rank, num_replicas=num_gpus, seed=random_seed)
    training_set_iterator = iter(torch.utils.data.DataLoader(dataset=training_set, sampler=training_set_sampler, batch_size=batch_size//num_gpus, **data_loader_kwargs))
    if rank == 0:
        print()
        print('Num images: ', len(training_set))
        print('Image shape:', training_set.image_shape)
        print('Label shape:', training_set.label_shape)
        print()

    # Construct networks.
    if rank == 0:
        print('Constructing networks...')
    common_kwargs = dict(c_dim=training_set.label_dim, img_resolution=training_set.resolution, img_channels=training_set.num_channels)
    G = dnnlib.util.construct_class_by_name(**G_kwargs, **common_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
    G.register_buffer('dataset_label_std', torch.tensor(training_set.get_label_std()).to(device))
    D = dnnlib.util.construct_class_by_name(**D_kwargs, **common_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
    G_ema = copy.deepcopy(G).eval() # running average of the weights of generator

    # Resume from existing pickle.
    if (resume_pkl is not None) and (rank == 0):
        print(f'Resuming from "{resume_pkl}"')
        with dnnlib.util.open_url(resume_pkl) as f:
            resume_data = legacy.load_network_pkl(f)
        for name, module in [('G', G), ('D', D), ('G_ema', G_ema)]:
            misc.copy_params_and_buffers(resume_data[name], module, require_all=False)

    # Print network summary tables.
    if rank == 0:
        z = torch.empty([batch_gpu, G.z_dim], device=device)
        c = torch.empty([batch_gpu, G.c_dim], device=device)
        img = misc.print_module_summary(G, [z, c])
        misc.print_module_summary(D, [img, c])

    # Setup augmentation.
    if rank == 0:
        print('Setting up augmentation...')
    augment_pipe = None
    ada_stats = None
    if (augment_kwargs is not None) and (augment_p > 0 or ada_target is not None):
        augment_pipe = dnnlib.util.construct_class_by_name(**augment_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
        augment_pipe.p.copy_(torch.as_tensor(augment_p))
        if ada_target is not None:
            ada_stats = training_stats.Collector(regex='Loss/signs/real')

    # Distribute across GPUs.
    if rank == 0:
        print(f'Distributing across {num_gpus} GPUs...')
    for module in [G, D, G_ema, augment_pipe]:
        if module is not None:
            for param in misc.params_and_buffers(module):
                if param.numel() > 0 and num_gpus > 1:
                    torch.distributed.broadcast(param, src=0)
    
    ## -------------------------- choose what loss to use --------------------------
    loss_modes = ['original', 'overfit', 'conditional', 'mask_real_face', 'overfit_by_GAN']
    loss_choice = loss_modes[0]
    print(f"Your choice of loss function is: {loss_choice}")
    ## -------------------------- plug in the BiSeNet for face segmentation -------------------------- 

    # evaluate():
    global net
    ## TODO: move this to dataloader to accelerate training
    if loss_choice == 'mask_real_face':
        n_classes = 19
        net = BiSeNet(n_classes=n_classes)
        net.cuda()
        # save_pth = osp.join('res/cp', cp)
        save_pth = '/home/xuyimeng/Repo/face-parsing.PyTorch/res/cp/79999_iter.pth'
        net.load_state_dict(torch.load(save_pth))
        net.eval()
    
    ##### --------------------------
    
    ## -------------- record some attributes of G -----------------------
    # get original contents
    # with open( 'wt') as f:
    #     json.dump(c, f, indent=2)
    with open(os.path.join(run_dir, 'training_options.json'), 'r') as json_file:
        existing_data = json.load(json_file)
    # append G's attributes
    # 
    existing_data.update(
        {'G':G.record_attributes_to_json()}
        )
    
    # append gaussian's attributes
    existing_data.update(
        {'gaussian optimizer':get_optimizer_parameters(G.g1.optimizer)}
        )
    
    # write back
    # Write the updated data back to the file
    with open(os.path.join(run_dir, 'training_options.json'), 'w') as json_file:
        json.dump(existing_data, json_file, indent=2)

    ## -------------- --------------------------- -----------------------


    # Setup training phases.
    if rank == 0:
        print('Setting up training phases...')
    loss = dnnlib.util.construct_class_by_name(device=device, G=G, D=D, augment_pipe=augment_pipe, **loss_kwargs) # subclass of training.loss.Loss
    phases = []
    for name, module, opt_kwargs, reg_interval in [('G', G, G_opt_kwargs, G_reg_interval), ('D', D, D_opt_kwargs, D_reg_interval)]:
        if reg_interval is None:
            opt = dnnlib.util.construct_class_by_name(params=module.parameters(), **opt_kwargs) # subclass of torch.optim.Optimizer
            phases += [dnnlib.EasyDict(name=name+'both', module=module, opt=opt, interval=1)]
        else: # Lazy regularization.
            mb_ratio = reg_interval / (reg_interval + 1)
            opt_kwargs = dnnlib.EasyDict(opt_kwargs)
            opt_kwargs.lr = opt_kwargs.lr * mb_ratio
            # if name=='D':
            #     print(f"{name}_lr:{opt_kwargs.lr}")
            #     st()
            opt_kwargs.betas = [beta ** mb_ratio for beta in opt_kwargs.betas]
            opt = dnnlib.util.construct_class_by_name(module.parameters(), **opt_kwargs) # subclass of torch.optim.Optimizer
            phases += [dnnlib.EasyDict(name=name+'main', module=module, opt=opt, interval=1)]
            phases += [dnnlib.EasyDict(name=name+'reg', module=module, opt=opt, interval=reg_interval)]
    for phase in phases:
        phase.start_event = None
        phase.end_event = None
        if rank == 0:
            phase.start_event = torch.cuda.Event(enable_timing=True)
            phase.end_event = torch.cuda.Event(enable_timing=True)

    # Export sample images.
    grid_size = None
    grid_z = None
    grid_c = None
    if rank == 0:
        print('Exporting sample images...')
        grid_size, images, labels = setup_snapshot_image_grid(training_set=training_set)
        save_image_grid(images, os.path.join(run_dir, 'reals.png'), drange=[0,255], grid_size=grid_size)
        
        if loss_choice == 'mask_real_face':
            images_masked = get_face_mask(images, type='numpy')
            save_image_grid(images_masked, os.path.join(run_dir, 'reals_masked.png'), drange=[0,255], grid_size=grid_size)

        grid_z = torch.randn([labels.shape[0], G.z_dim], device=device).split(batch_gpu)
        grid_c = torch.from_numpy(labels).to(device).split(batch_gpu)

    # Initialize logs.
    if rank == 0:
        print('Initializing logs...')
    stats_collector = training_stats.Collector(regex='.*')
    stats_metrics = dict()
    stats_jsonl = None
    stats_tfevents = None
    if rank == 0:
        stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'wt')
        try:
            import torch.utils.tensorboard as tensorboard
            stats_tfevents = tensorboard.SummaryWriter(run_dir)
        except ImportError as err:
            print('Skipping tfevents export:', err)

    # Train.
    if rank == 0:
        print(f'Training for {total_kimg} kimg...')
        print()
    cur_nimg = resume_kimg * 1000
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    batch_idx = 0
    if progress_fn is not None:
        progress_fn(0, total_kimg)
        
    
    while True:

        # Fetch training data.
        with torch.autograd.profiler.record_function('data_fetch'):
            phase_real_img, phase_real_c = next(training_set_iterator) # [4, 3, 512, 512]  [4, 25]
            phase_real_img = (phase_real_img.to(device).to(torch.float32) / 127.5 - 1).split(batch_gpu) # 0~255 -> -1~1
            phase_real_c = phase_real_c.to(device).split(batch_gpu)
            all_gen_z = torch.randn([len(phases) * batch_size, G.z_dim], device=device)
            all_gen_z = [phase_gen_z.split(batch_gpu) for phase_gen_z in all_gen_z.split(batch_size)]
            all_gen_c = [training_set.get_label(np.random.randint(len(training_set))) for _ in range(len(phases) * batch_size)]
            all_gen_c = torch.from_numpy(np.stack(all_gen_c)).pin_memory().to(device)
            all_gen_c = [phase_gen_c.split(batch_gpu) for phase_gen_c in all_gen_c.split(batch_size)]
        

        # Execute training phases.
        phases_updated_gaussians = set()

        
               
        for phase, phase_gen_z, phase_gen_c in zip(phases, all_gen_z, all_gen_c):
            # print(phase.name)
            if batch_idx % phase.interval != 0:
                continue
            if phase.start_event is not None:
                phase.start_event.record(torch.cuda.current_stream(device))

            # Accumulate gradients.
             
            phase.opt.zero_grad(set_to_none=True)
            phase.module.requires_grad_(True)
            ## Also set grad for gaussians
            # print("A")   
            if 'G' in phase.name:
                for _gi in range(1, phase.module.num_gaussians+1):
                    _gs = getattr(phase.module, f"g{_gi}")
                    _gs.optimizer.zero_grad(set_to_none = True)
                    _gs.requires_grad(True)
                    # if _gs._xyz.requires_grad:
                    #     print(f"_gs{_gi} _xyz.requires_grad")    
                    # print(f"set gs{_gi} to requires grad True ///") # except: self._features_dc, self._features_rest
                    
            # print("B")
            # for _gi in range(1, G.num_gaussians+1):
            #         _gs = getattr(G, f"g{_gi}")
            #         if _gs._features_dc.requires_grad:
            #             print(f"_gs{_gi} _features_dc.requires_grad")
                    
           
            
            for real_img, real_c, gen_z, gen_c in zip(phase_real_img, phase_real_c, phase_gen_z, phase_gen_c):
                if loss_choice == 'original':
                    loss.accumulate_gradients(phase=phase.name, real_img=real_img, real_c=real_c, gen_z=gen_z, gen_c=gen_c, gain=phase.interval, cur_nimg=cur_nimg)
                elif loss_choice == 'overfit':
                    loss.accumulate_gradients_debug(phase=phase.name, real_c=real_c, real_img=real_img, gen_z=gen_z, gen_c=gen_c, gain=phase.interval, cur_nimg=cur_nimg) 
                # elif loss_choice == 'conditional': # TODO: not develop this part yet
                #     loss.accumulate_gradients_conditionalD(phase=phase.name, real_c=real_c, real_img=real_img, gen_z=gen_z, gen_c=gen_c, gain=phase.interval, cur_nimg=cur_nimg) 
                elif loss_choice == 'mask_real_face':
                    ##### use segmentation mask to keep only the facial part of the real image
                    real_img_masked_batch = get_face_mask(real_img) 
                    # save_image_grid(real_img_masked_batch.cpu().numpy(), os.path.join(run_dir, f'debug{cur_nimg//1000:06d}.png'), drange=[-1,1], grid_size=(2,2))
                    loss.accumulate_gradients(phase=phase.name, real_c=real_c, real_img=real_img_masked_batch, gen_z=gen_z, gen_c=gen_c, gain=phase.interval, cur_nimg=cur_nimg) 
                elif loss_choice == 'overfit_by_GAN':   
                    # replace real image with 3DMM data rendered
                    # real_3dmm_image = loss.run_G(gen_z, real_c, )
                    _gen_img, _gen_ws = loss.run_G(gen_z, real_c, swapping_prob=0, neural_rendering_resolution=loss.neural_rendering_resolution_initial)
                    real_3dmm_image = _gen_img["image_real"]
                    # st()
                    save_image_grid(real_3dmm_image.detach().cpu().numpy(), os.path.join(run_dir, f'debug{cur_nimg//1000:06d}.png'), drange=[0,1], grid_size=(2,2))
                    loss.accumulate_gradients(phase=phase.name, real_c=real_c, real_img=real_3dmm_image, gen_z=gen_z, gen_c=gen_c, gain=phase.interval, cur_nimg=cur_nimg) 
                
                else:
                    print(f"Invalid loss choice, please choose from [{loss_modes}]")
                    
            # print("C")
            # for _gi in range(1, G.num_gaussians+1):
            #         _gs = getattr(G, f"g{_gi}")
            #         if _gs._features_dc.requires_grad:
            #             print(f"_gs{_gi} _features_dc.requires_grad")

                
            phase.module.requires_grad_(False)
            ## Also set grad for gaussians
            if 'G' in phase.name:
                for _gi in range(1, phase.module.num_gaussians+1):
                    _gs = getattr(phase.module, f"g{_gi}")
                    _gs.requires_grad(False)
                    # print(f"set gs{_gi} to requires grad False xxx")
            
            # print("D")
            # for _gi in range(1, G.num_gaussians+1):
            #         _gs = getattr(G, f"g{_gi}")
            #         if _gs._features_dc.requires_grad:
            #             print(f"_gs{_gi} _features_dc.requires_grad")
            

            # Update weights.
            with torch.autograd.profiler.record_function(phase.name + '_opt'):
                params = [param for param in phase.module.parameters() if param.numel() > 0 and param.grad is not None]
                if len(params) > 0:
                    flat = torch.cat([param.grad.flatten() for param in params])
                    if num_gpus > 1:
                        torch.distributed.all_reduce(flat)
                        flat /= num_gpus
                    misc.nan_to_num(flat, nan=0, posinf=1e5, neginf=-1e5, out=flat)
                    grads = flat.split([param.numel() for param in params])
                    for param, grad in zip(params, grads):
                        param.grad = grad.reshape(param.shape)
                phase.opt.step()
                
                # print(f"Gradients for {phase.name} -----begin---")
                # for name, param in phase.module.named_parameters():
                #     # print(f" {name}") # not including gaussian and minicam
                #     if param.grad is not None:
                #         print(f"Gradients for {name} have been computed.")
                #     else:
                #         pass
                #         # print(f"Gradients for {name} have NOT been computed!!")
                # print(f"Gradients for {phase.name} -----end---")
                # st()
                
                # optimizer step for gaussian
                # if iteration < opt.iterations: 
                
                if 'G' in phase.name:
                    for _gi in range(1, phase.module.num_gaussians+1):
                        _gs = getattr(phase.module, f"g{_gi}")
                        if _gs._scaling.grad is not None:
                            phases_updated_gaussians.add(_gi)
                            # st() # bu t here features_dc/rest having requires_grad=True???
                            _gs.optimizer.step()
                            _gs.optimizer.zero_grad(set_to_none = True)
                            # print(f"gs{_gi}._xyz has changed: {torch.any(G.gaussian_debug._xyz != _gs._xyz )}")
                            # print(f'---the gs {_gi} xyz has changed: {torch.any(getattr(G_ema, f"g{_gi}")._xyz != _gs._xyz )}')
            
                            
                elif 'D' in phase.name:
                    for _gi in range(1, G.num_gaussians+1):
                        _gs = getattr(G, f"g{_gi}")
                        if _gs._scaling.grad is not None:
                            print(f'there should be no grad for gaussians {_gi} in D phase')
                            st()
                else:
                    print(f"what is this phase? -> {phase.name}")
                            
            
            # past_phases.append({phase.name:_updated_gaussians})
            # Phase done.
            if phase.end_event is not None:
                phase.end_event.record(torch.cuda.current_stream(device))

        # Update G_ema.
        # FIXME: the gaussian_banks are not updated for G_ema!!!
        with torch.autograd.profiler.record_function('Gema'):
            ema_nimg = ema_kimg * 1000
            if ema_rampup is not None:
                ema_nimg = min(ema_nimg, cur_nimg * ema_rampup)
            ema_beta = 0.5 ** (batch_size / max(ema_nimg, 1e-8))
            for p_ema, p in zip(G_ema.parameters(), G.parameters()):
                p_ema.copy_(p.lerp(p_ema, ema_beta))
            for b_ema, b in zip(G_ema.buffers(), G.buffers()):
                b_ema.copy_(b)
            G_ema.neural_rendering_resolution = G.neural_rendering_resolution
            G_ema.rendering_kwargs = G.rendering_kwargs.copy()
            # TODO: update gaussian bank
            for _gi in phases_updated_gaussians:
                # print(f'copying G.g{_gi} to G_ema.g{_gi}')
                _gs_copy = getattr(G, f"g{_gi}").get_copy()
                setattr(G_ema, f"g{_gi}", _gs_copy)
            
            

        # Update state.
        cur_nimg += batch_size
        batch_idx += 1

        # Execute ADA heuristic.
        if (ada_stats is not None) and (batch_idx % ada_interval == 0):
            ada_stats.update()
            adjust = np.sign(ada_stats['Loss/signs/real'] - ada_target) * (batch_size * ada_interval) / (ada_kimg * 1000)
            augment_pipe.p.copy_((augment_pipe.p + adjust).max(misc.constant(0, device=device)))

        # Perform maintenance tasks once per tick.
        done = (cur_nimg >= total_kimg * 1000)
        if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
            continue

        # Print status line, accumulating the same information in training_stats.
        tick_end_time = time.time()
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<8.1f}"]
        fields += [f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
        fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
        fields += [f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"]
        fields += [f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"]
        fields += [f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"]
        fields += [f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}"]
        fields += [f"reserved {training_stats.report0('Resources/peak_gpu_mem_reserved_gb', torch.cuda.max_memory_reserved(device) / 2**30):<6.2f}"]
        torch.cuda.reset_peak_memory_stats()
        fields += [f"augment {training_stats.report0('Progress/augment', float(augment_pipe.p.cpu()) if augment_pipe is not None else 0):.3f}"]
        training_stats.report0('Timing/total_hours', (tick_end_time - start_time) / (60 * 60))
        training_stats.report0('Timing/total_days', (tick_end_time - start_time) / (24 * 60 * 60))
        if rank == 0:
            print(' '.join(fields))

        # Check for abort.
        if (not done) and (abort_fn is not None) and abort_fn():
            done = True
            if rank == 0:
                print()
                print('Aborting...')

        # Save image snapshot.
        print(f"image_snapshot_ticks is {image_snapshot_ticks}")
        if (rank == 0) and (image_snapshot_ticks is not None) and (done or cur_tick % image_snapshot_ticks == 0):
            out = [G_ema(z=z, c=c, noise_mode='const') for z, c in zip(grid_z, grid_c)]
            images = torch.cat([o['image'].cpu() for o in out]).detach().numpy()
            images_raw = torch.cat([o['image_raw'].cpu() for o in out]).detach().numpy()
            images_mask = -torch.cat([o['image_mask'].cpu() for o in out]).detach().numpy()
            images_real = torch.cat([o['image_real'].cpu() for o in out]).detach().numpy() # FIXME: init with gt texture for debug
            save_image_grid(images, os.path.join(run_dir, f'fakes{cur_nimg//1000:06d}.png'), drange=[-1,1], grid_size=grid_size)
            save_image_grid(images_raw, os.path.join(run_dir, f'fakes{cur_nimg//1000:06d}_raw.png'), drange=[-1,1], grid_size=grid_size)
            save_image_grid(images_mask, os.path.join(run_dir, f'fakes{cur_nimg//1000:06d}_mask.png'), drange=[images_mask.min(), images_mask.max()], grid_size=grid_size)
            save_image_grid(images_real, os.path.join(run_dir, f'reals{cur_nimg//1000:06d}.png'), drange=[0,1], grid_size=grid_size)
            
            
            save_override_color = getattr(G_ema, 'use_colors_precomp')
            
            # save ply and see if the texture is well optimized
            # if not os.path.exists(os.path.join(run_dir, "./gt_3dmm.ply")):
            G_ema.gaussian_debug.save_ply(os.path.join(run_dir, "./gt_3dmm.ply"), save_override_color)
            
            print(f"total updated gaussians:{phases_updated_gaussians}")
            
            for gs_i in phases_updated_gaussians:
                # getattr(G_ema, f'g{gs_i}').save_ply(os.path.join(run_dir, f"./fake_3dmm_{gs_i}.ply"))
                _gs = getattr(G_ema, f'g{gs_i}')
                # print(f"gs in updated gaussians: {gs_i in past_phases}")
                try:
                    _gs.save_ply(os.path.join(run_dir, f"./fake_3dmm_{gs_i}.ply"),save_override_color=save_override_color)
                    # print(f"Saved sucessfully the {gs_i}th gaussian")
                except:
                    print(f"The {gs_i}th gaussian not updated yet")
                    pass
               
            print(f"Saved ply for {phases_updated_gaussians} gaussians")
        
            #--------------------
            # # Log forward-conditioned images

            # forward_cam2world_pose = LookAtPoseSampler.sample(3.14/2, 3.14/2, torch.tensor([0, 0, 0.2], device=device), radius=2.7, device=device)
            # intrinsics = torch.tensor([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]], device=device)
            # forward_label = torch.cat([forward_cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)

            # grid_ws = [G_ema.mapping(z, forward_label.expand(z.shape[0], -1)) for z, c in zip(grid_z, grid_c)]
            # out = [G_ema.synthesis(ws, c=c, noise_mode='const') for ws, c in zip(grid_ws, grid_c)]

            # images = torch.cat([o['image'].cpu() for o in out]).numpy()
            # images_raw = torch.cat([o['image_raw'].cpu() for o in out]).numpy()
            # images_depth = -torch.cat([o['image_depth'].cpu() for o in out]).numpy()
            # save_image_grid(images, os.path.join(run_dir, f'fakes{cur_nimg//1000:06d}_f.png'), drange=[-1,1], grid_size=grid_size)
            # save_image_grid(images_raw, os.path.join(run_dir, f'fakes{cur_nimg//1000:06d}_raw_f.png'), drange=[-1,1], grid_size=grid_size)
            # save_image_grid(images_depth, os.path.join(run_dir, f'fakes{cur_nimg//1000:06d}_depth_f.png'), drange=[images_depth.min(), images_depth.max()], grid_size=grid_size)

            #--------------------
            # # Log Cross sections

            # grid_ws = [G_ema.mapping(z, c.expand(z.shape[0], -1)) for z, c in zip(grid_z, grid_c)]
            # out = [sample_cross_section(G_ema, ws, w=G.rendering_kwargs['box_warp']) for ws, c in zip(grid_ws, grid_c)]
            # crossections = torch.cat([o.cpu() for o in out]).numpy()
            # save_image_grid(crossections, os.path.join(run_dir, f'fakes{cur_nimg//1000:06d}_crossection.png'), drange=[-50,100], grid_size=grid_size)

        # Save network snapshot.
        snapshot_pkl = None
        snapshot_data = None

        # ## TODO: recover this save snapshot
        # if (network_snapshot_ticks is not None) and (done or cur_tick % network_snapshot_ticks == 0):
        #     snapshot_data = dict(training_set_kwargs=dict(training_set_kwargs))
        #     for name, module in [('G', G), ('D', D), ('G_ema', G_ema), ('augment_pipe', augment_pipe)]:
        #         if module is not None:
        #             if num_gpus > 1:
        #                 misc.check_ddp_consistency(module, ignore_regex=r'.*\.[^.]+_(avg|ema)')
        #             module = copy.deepcopy(module).eval().requires_grad_(False).cpu()
        #             # module = module.detach().clone().eval().requires_grad_(False).cpu()
        #             # print(f"copying module {name}")
        #             # st()
        #             # module = copy.deepcopy(module).eval().requires_grad_(False).cpu()
        #         snapshot_data[name] = module
        #         del module # conserve memory
        #     snapshot_pkl = os.path.join(run_dir, f'network-snapshot-{cur_nimg//1000:06d}.pkl')
        #     if rank == 0:
        #         with open(snapshot_pkl, 'wb') as f:
        #             pickle.dump(snapshot_data, f)

        # # Evaluate metrics.
        # if (snapshot_data is not None) and (len(metrics) > 0):
        #     if rank == 0:
        #         print(run_dir)
        #         print('Evaluating metrics...')
        #     for metric in metrics:
        #         result_dict = metric_main.calc_metric(metric=metric, G=snapshot_data['G_ema'],
        #             dataset_kwargs=training_set_kwargs, num_gpus=num_gpus, rank=rank, device=device)
        #         if rank == 0:
        #             metric_main.report_metric(result_dict, run_dir=run_dir, snapshot_pkl=snapshot_pkl)
        #         stats_metrics.update(result_dict.results)
        # del snapshot_data # conserve memory
        # # TODO: RECOVER THE ABOVE MODULE TO CALC METRICS
        # # if rank == 0:
        # #     print('SKIP Evaluating metrics!!!')
        

        # Collect statistics.
        for phase in phases:
            value = []
            if (phase.start_event is not None) and (phase.end_event is not None):
                phase.end_event.synchronize()
                value = phase.start_event.elapsed_time(phase.end_event)
            training_stats.report0('Timing/' + phase.name, value)
        stats_collector.update()
        stats_dict = stats_collector.as_dict()

        # Update logs.
        timestamp = time.time()
        if stats_jsonl is not None:
            fields = dict(stats_dict, timestamp=timestamp)
            stats_jsonl.write(json.dumps(fields) + '\n')
            stats_jsonl.flush()
        if stats_tfevents is not None:
            global_step = int(cur_nimg / 1e3)
            walltime = timestamp - start_time
            for name, value in stats_dict.items():
                stats_tfevents.add_scalar(name, value.mean, global_step=global_step, walltime=walltime)
            for name, value in stats_metrics.items():
                stats_tfevents.add_scalar(f'Metrics/{name}', value, global_step=global_step, walltime=walltime)
            stats_tfevents.flush()
        if progress_fn is not None:
            progress_fn(cur_nimg // 1000, total_kimg)

        # Update state.
        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        if done:
            break

    # Done.
    if rank == 0:
        print()
        print('Exiting...')

#----------------------------------------------------------------------------
