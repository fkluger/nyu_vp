import glob
import os
import imageio
import numpy as np


class NYUVP:

    def __init__(self, image_dir_path, csv_dir_path, split=None, keep_in_mem=True, crop=((7, 6), (625, 468))):
        self.image_dir = image_dir_path
        self.csv_dir = csv_dir_path
        self.keep_in_mem = keep_in_mem
        self.crop = crop

        self.image_files = glob.glob(os.path.join(self.image_dir, "*.png"))
        self.image_files.sort()

        assert len(self.image_files) == 1449, "images are incomplete, or wrong directory"
        if split is not None:
            if split == "train":
                self.image_files = self.image_files[0:1000]
            elif split == "val":
                self.image_files = self.image_files[1000:1224]
            elif split == "test":
                self.image_files = self.image_files[1224:1450]
            else:
                assert False, "invalid split"

        self.csv1_files = [self._csv1_from_path(path) for path in self.image_files]
        self.csv2_files = [self._csv2_from_path(path) for path in self.image_files]
        self.dataset = [None for _ in self.image_files]

    @staticmethod
    def _id_string_from_path(path):
        basename = os.path.basename(path)
        id_string = os.path.splitext(basename)[0]
        return id_string

    def _csv1_from_path(self, path):
        id_string = self._id_string_from_path(path)
        csv1 = os.path.join(self.csv_dir, "csv1_%s.csv" % id_string)
        return csv1

    def _csv2_from_path(self, path):
        id_string = self._id_string_from_path(path)
        csv2 = os.path.join(self.csv_dir, "csv2_%s.csv" % id_string)
        return csv2

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, key):
        datum = self.dataset[key]

        if datum is None:
            image = imageio.imread(self.image_files[key])
            csv1 = np.genfromtxt(self.csv1_files[key], delimiter=" ", skip_header=1)
            if len(csv1.shape) > 1:
                vp = csv1[:, 1:]
            else:
                vp = csv1[1:]
                vp = np.expand_dims(vp, 0)

            if self.crop is not None:
                vp[:, 0] -= self.crop[0][0]
                vp[:, 1] -= self.crop[0][1]

                image = image[self.crop[0][1]:self.crop[0][1]+self.crop[1][1],
                              self.crop[0][0]:self.crop[0][0]+self.crop[1][0], :]

            # csv2 = np.genfromtxt(self.csv1_files[key], delimiter=" ", skip_header=1)
            datum = {'image': image, 'vp': vp}
            if self.keep_in_mem:
                self.dataset[key] = datum

        return datum

if __name__ == '__main__':
    import matplotlib.pyplot as plt


    dataset = NYUVP("/tnt/data/scene_understanding/NYU/images", "/tnt/data/kluger/datasets/NYU")

    for idx in range(len(dataset)):
        print(idx)
        test_datum = dataset[idx]

        plt.figure()
        plt.imshow(test_datum['image'])
        vp = test_datum['vp']
        # for vi in range(vp.shape[0]):
        #     p = vp[vi,:]
        #     plt.plot(p[0], p[1], 'rx')

        plt.show()

    print(len(dataset))