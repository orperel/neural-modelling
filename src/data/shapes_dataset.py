import os
import zipfile
import json
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import torch
import torch.utils.data as data
from torchvision.datasets.utils import download_url, makedir_exist_ok


# http://ubee.enseeiht.fr/ShapesDataset/
class ShapesDataset(data.Dataset):

    urls = dict(
        classes_metadata='http://ubee.enseeiht.fr/ShapesDataset/data/NamesJSON.zip',
        shapes_information='http://ubee.enseeiht.fr/ShapesDataset/data/ShapesJSON.zip',
        user_annotations='http://ubee.enseeiht.fr/ShapesDataset/data/annotation.csv',
        majority_vote='http://ubee.enseeiht.fr/ShapesDataset/data/MajorityJSON.zip',
        spectral_clustering='http://ubee.enseeiht.fr/ShapesDataset/data/SpectralJSON.zip'
    )

    def __init__(self, root, train=True, transform=None, target_transform=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        self.download()
        if not self._check_raw_exists():
            raise RuntimeError('Dataset not found.' +
                               ' Did download or extraction fail?')

        if not self._check_processed_exists():
            entries = self.parse_shapes_dir()

            processed_entries = []
            for entry in tqdm(entries):
                # mesh, modifiers = simplify(entry)
                from src.data.visvalingam import simplify_mesh
                mesh, modifiers = simplify_mesh(entry, self.processed_folder)
                processed_entries.append((mesh, modifiers))

        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file
        self.data, self.targets = torch.load(os.path.join(self.processed_folder, data_file))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    @property
    def raw_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'raw')

    @property
    def processed_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'processed')

    @property
    def class_to_idx(self):
        return {_class: i for i, _class in enumerate(self.classes)}

    def _check_processed_exists(self):
        return os.path.exists(os.path.join(self.processed_folder, 'Shapes'))

    def _check_raw_exists(self):
        return os.path.exists(os.path.join(self.raw_folder, 'Shapes'))

    @staticmethod
    def extract_zip(zip_path, remove_finished=False):
        print('Extracting {}'.format(zip_path))
        with zipfile.ZipFile(zip_path, 'r') as zip_file:
            out_path = Path(zip_path).parent
            zip_file.extractall(out_path)
            # extracted_folder_name = zip_file.filelist[0].filename
        if remove_finished:
            os.unlink(zip_path)

    def download(self):
        """Download 2DShapesStructure data if it doesn't exist in processed_folder already."""

        if self._check_raw_exists():
            return

        makedir_exist_ok(self.raw_folder)
        makedir_exist_ok(self.processed_folder)

        # download files
        for url_title, url in self.urls.items():
            filename = url.rpartition('/')[2]
            file_path = os.path.join(self.raw_folder, filename)
            download_url(url, root=self.raw_folder)
            if filename.endswith('.zip'):
                self.extract_zip(zip_path=file_path, remove_finished=True)
            print(f'Fetched {filename}.')

        # training_set = (
        #     self.read_image_file(os.path.join(self.raw_folder, 'train-images-idx3-ubyte')),
        #     self.read_label_file(os.path.join(self.raw_folder, 'train-labels-idx1-ubyte'))
        # )
        # test_set = (
        #     self.read_image_file(os.path.join(self.raw_folder, 't10k-images-idx3-ubyte')),
        #     self.read_label_file(os.path.join(self.raw_folder, 't10k-labels-idx1-ubyte'))
        # )
        # with open(os.path.join(self.processed_folder, self.training_file), 'wb') as f:
        #     torch.save(training_set, f)
        # with open(os.path.join(self.processed_folder, self.test_file), 'wb') as f:
        #     torch.save(test_set, f)

        print('Done!')

    def parse_shapes_dir(self):
        # process and save as torch files
        print('Processing Shapes folder...')
        entries = []
        shapes_path = os.path.join(self.raw_folder, 'Shapes')
        for single_entry_filename in tqdm(os.listdir(shapes_path)):
            shape_json_path = os.path.join(shapes_path, single_entry_filename)
            with open(shape_json_path, 'rb') as shape_json:
                json_data = json.load(shape_json)
                object_name = single_entry_filename.split('.json')[0]
                points = np.array([(pt['x'], pt['y']) for pt in json_data['points']])
                triangles = np.array([(tri['p1'], tri['p2'], tri['p3']) for tri in json_data['triangles']])
                entries.append(dict(object_name=object_name, points=points, triangles=triangles))
        return entries

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

    @staticmethod
    def read_label_file(path):
        pass
        # with open(path, 'rb') as f:
        #     data = f.read()
        #     assert get_int(data[:4]) == 2049
        #     length = get_int(data[4:8])
        #     parsed = np.frombuffer(data, dtype=np.uint8, offset=8)
        #     return torch.from_numpy(parsed).view(length).long()


ShapesDataset('data')