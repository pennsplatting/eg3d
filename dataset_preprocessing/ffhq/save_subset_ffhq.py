import os
import sys
import argparse
import zipfile
import numpy as np
import PIL.Image
import json

try:
    import pyspng
except ImportError:
    pyspng = None
    
class ImageLoader:
    def __init__(self, path, type,):
        self._path = path
        self._zipfile = None

        if os.path.isdir(self._path):
            self._type = 'dir'
            self._all_fnames = {os.path.relpath(os.path.join(root, fname), start=self._path) for root, _dirs, files in os.walk(self._path) for fname in files}
        elif self._file_ext(self._path) == '.zip':
            self._type = 'zip'
            self._all_fnames = set(self._get_zipfile().namelist())
        else:
            raise IOError('Path must point to a directory or zip')

        PIL.Image.init()
        self._image_fnames = sorted(fname for fname in self._all_fnames if self._file_ext(fname) in PIL.Image.EXTENSION)
        if len(self._image_fnames) == 0:
            raise IOError('No image files found in the specified path')

        # name = os.path.splitext(os.path.basename(self._path))[0]
        # raw_shape = [len(self._image_fnames)] + list(self._load_raw_image(0).shape)
        
    
    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _load_raw_labels(self):
        fname = 'dataset.json'
        if fname not in self._all_fnames:
            return None
        with self._open_file(fname) as f:
            labels = json.load(f)['labels']

            # ## print all the contents of the json file
            # data = json.load(f) # to use this line, must comment the above line to load the json
            # # Print all contents
            # for key, value in data.items():
            #     print(f"{key}:, {value[:3]}")
            # exit()
            # ## the 'dataset.json' only have one key: "label"

        if labels is None:
            return None
        labels = dict(labels)
        labels = [labels[fname.replace('\\', '/')] for fname in self._image_fnames]
        labels = np.array(labels)
        labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])
        return labels
    
    def _load_raw_image(self, raw_idx):
        fname = self._image_fnames[raw_idx]
        with self._open_file(fname) as f:
            if pyspng is not None and self._file_ext(fname) == '.png':
                image = pyspng.load(f.read())
            else:
                image = np.array(PIL.Image.open(f))
        if image.ndim == 2:
            image = image[:, :, np.newaxis] # HW => HWC
        image = image.transpose(2, 0, 1) # HWC => CHW
        return image

    def _open_file(self, fname):
        if self._type == 'dir':
            return open(os.path.join(self._path, fname), 'rb')
        if self._type == 'zip':
            return self._get_zipfile().open(fname, 'r')
        return None

    def _get_zipfile(self):
        if self._type == 'zip':
            return zipfile.ZipFile(self._path, 'r')
        return None

    def _get_image_fnames(self):
        if self._type == 'dir':
            return [f for f in os.listdir(self._path) if self._file_ext(f) == '.png']
        if self._type == 'zip':
            with self._get_zipfile() as zip_file:
                return [f for f in zip_file.namelist() if self._file_ext(f) == '.png']
        return []

    def _file_ext(self, fname):
        return os.path.splitext(fname)[-1].lower()

    ## to create a subset
    def save_subset_to_zip(self, subset_size, output_zip_path):
        subset_fnames = self._image_fnames[:subset_size]
        fname = 'dataset.json'
        if fname not in self._all_fnames:
            return None
        with self._open_file(fname) as f:
            original_labels = json.load(f)['labels']
            # data = json.load(f)
            # for key, value in data.items():
            #     print(f"{key}:, {type(value)}")
            # exit()
        original_labels = dict(original_labels)
        # subset_labels = self._load_raw_labels()[:subset_size]
        # original_labels = self._load_raw_labels()
        # Create a subset of labels with the same structure as the original dataset.json
        subset_labels = {fname.replace('\\', '/'): original_labels[fname.replace('\\', '/')] for fname in subset_fnames}

        with zipfile.ZipFile(output_zip_path, 'w') as output_zip:
            with self._get_zipfile() as input_zip:
                for fname, label in zip(subset_fnames, subset_labels):
                    data = input_zip.read(fname)
                    output_zip.writestr(fname, data)         

                # Create and add a new dataset.json for the subset
                print("subset_fnames\n", subset_fnames)
                print("subset_labels\n",subset_labels.keys())
                subset_labels_list = [[key, value] for key, value in subset_labels.items()]
                dataset_json = {'labels': subset_labels_list}
                output_zip.writestr('dataset.json', json.dumps(dataset_json))
                
# # Example usage:
# ffhq_loader = ImageLoader(path='FFHQ.zip', type='zip')
# ffhq_loader.save_subset_to_zip(subset_size=1000, output_zip_path='FFHQ_subset.zip')

def main():
    parser = argparse.ArgumentParser(description="Create a subset of images from a given input directory or zip file.")
    parser.add_argument("--indir", required=True, help="Input directory or zip file containing images.")
    parser.add_argument("--output_zip_name", required=True, help="Name of the output zip file.")
    parser.add_argument("--subset_size", type=int, default=1000, help="Size of the subset to create.")

    args = parser.parse_args()

    # Example usage:
    loader = ImageLoader(path=args.indir, type='zip' if args.indir.endswith('.zip') else 'dir')
    loader.save_subset_to_zip(subset_size=args.subset_size, output_zip_path=args.output_zip_name)

if __name__ == "__main__":
    main()