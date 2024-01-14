# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--indir', type=str, required=True)
parser.add_argument("--subset_size", type=int, default=1000, help="Size of the subset to create.")
args = parser.parse_args()

## helper func: remove unempty folders
import shutil
def remove_contents_if_not_empty(folder_path):
    # Check if the folder exists
    if os.path.exists(folder_path):
        # Get the list of files and subdirectories in the folder
        folder_contents = os.listdir(folder_path)

        # Check if the folder is not empty
        if folder_contents:
            print(f"Removing contents of the folder: {folder_path}")
            
            # Iterate through each file and subdirectory and remove them
            for item in folder_contents:
                item_path = os.path.join(folder_path, item)
                
                # Check if it is a file or a directory
                if os.path.isfile(item_path):
                    os.remove(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                    
            print("Contents removed successfully.")
        else:
            print(f"The out folder is empty: {folder_path}")
    else:
        pass
# -----------------------------------------------------------------------------------------------

try:
    # save subset 
    subset_zip_name = f'FFHQ_subset_{args.subset_size}.zip'
    command = f"python save_subset_ffhq.py --original_zip {args.indir} --output_zip_name {subset_zip_name} --subset_size {args.subset_size}" 
    print(command)
    os.system(command)
except:
    exit()


base_name, extension = os.path.splitext(subset_zip_name)
try:
    remove_contents_if_not_empty(base_name)
    command = f"unzip {base_name}.zip -d {base_name}"
    print(command)
    os.system(command)
except:
    exit()

args.indir = os.path.join(base_name,'00000')
print(f"Now the args.indir is {args.indir}")
out_folder = args.indir.split("/")[-2] if args.indir.endswith("/") else args.indir.split("/")[-1]
print(f"out_folder: {out_folder}")

remove_contents_if_not_empty(out_folder)

intermd_content_path = os.path.join(os.getcwd(),'Deep3DFaceRecon_pytorch/checkpoints/pretrained/results',out_folder)
remove_contents_if_not_empty(intermd_content_path)


try:
    # run mtcnn needed for Deep3DFaceRecon
    command = "python batch_mtcnn.py --in_root " + args.indir
    print(command)
    os.system(command)
except:
    exit()


try:
    # run Deep3DFaceRecon
    os.chdir('Deep3DFaceRecon_pytorch')
    command = "python test.py --img_folder=../" + args.indir + " --gpu_ids=0 --name=pretrained --epoch=20"
    print(command)
    os.system(command)
    os.chdir('..')
except:
    exit()


try:
    # crop out the input image
    command = "python resize_image_of_3DMM.py --indir=" + args.indir
    print(command)
    os.system(command)
except:
    exit()


try:
    # convert the pose to our format
    command = f"python 3dface2idr_mat.py --in_root Deep3DFaceRecon_pytorch/checkpoints/pretrained/results/{out_folder}/epoch_20_000000 --out_path {os.path.join(args.indir, 'crop', 'cameras.json')}"
    print(command)
    os.system(command)
except:
    exit()


try:
    # additional correction to match the submission version
    command = f"python preprocess_face_cameras.py --source {os.path.join(args.indir, 'crop')} --dest {out_folder} --mode orig"
    print(command)
    os.system(command)
except:
    exit()

try:
    ## make a zip ready for training
    os.chdir(out_folder)
    # command = f"zip -r {args.indir.split('/')[0]}_finished.zip {out_folder}"
    command = f"zip -o {args.indir.split('/')[0]}_finished.zip *"
    print(command)
    os.system(command)
    os.chdir('..')
    
    command = f"mv {out_folder}/{args.indir.split('/')[0]}_finished.zip ~/Repo/eg3d/"
    print(command)
    os.system(command)
except:
    exit()


## Usage of this file:
## --indir FFHQ_subset_xx/00000 (not end with '/')
## ENV: conda activate eg3d