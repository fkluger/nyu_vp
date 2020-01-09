import glob
import os
import csv
import numpy as np
import scipy.io
from .lsd import lsd


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


class NYUVP:

    def __init__(self, data_dir_path="./data", split='all', keep_data_in_memory=True, mat_file_path=None,
                 normalise_coordinates=False, remove_borders=False, extract_lines=False):

        self.keep_in_mem = keep_data_in_memory
        self.normalise_coords = normalise_coordinates
        self.remove_borders = remove_borders
        self.extract_lines = extract_lines

        self.vps_files = glob.glob(os.path.join(data_dir_path, "vps*"))
        self.lsd_line_files = glob.glob(os.path.join(data_dir_path, "lsd_lines*"))
        self.labelled_line_files = glob.glob(os.path.join(data_dir_path, "labelled_lines*"))
        self.vps_files.sort()
        self.lsd_line_files.sort()
        self.labelled_line_files.sort()

        if split == "train":
            self.set_ids = list(range(0, 1000))
        elif split == "val":
            self.set_ids = list(range(1000, 1224))
        elif split == "trainval":
            self.set_ids = list(range(0, 1224))
        elif split == "test":
            self.set_ids = list(range(1224, 1449))
        elif split == "all":
            self.set_ids = list(range(0, 1449))
        else:
            assert False, "invalid split: %s " % split

        self.dataset = [None for _ in self.set_ids]

        self.data_mat = None
        if mat_file_path is not None:
            self.data_mat = scipy.io.loadmat(mat_file_path, variable_names=["images"])

        fx_rgb = 5.1885790117450188e+02
        fy_rgb = 5.1946961112127485e+02
        cx_rgb = 3.2558244941119034e+02
        cy_rgb = 2.5373616633400465e+02

        K = np.matrix([[fx_rgb, 0, cx_rgb], [0, fy_rgb, cy_rgb], [0, 0, 1]])

        if normalise_coordinates:
            S = np.matrix([[1. / 320., 0, -1.], [0, 1. / 320., -.75], [0, 0, 1]])
            K = S * K

        self.Kinv = K.I

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, key):

        id = self.set_ids[key]

        datum = self.dataset[key]

        if datum is None:

            lsd_line_segments = None

            if self.data_mat is not None:
                image_rgb = self.data_mat['images'][:,:,:,id]
                image = rgb2gray(image_rgb)

                if self.remove_borders:
                    image_ = image[6:473,7:631].copy()
                else:
                    image_ = image

                if self.extract_lines:
                    lsd_line_segments = lsd.detect_line_segments(image_)

                    if self.remove_borders:
                        lsd_line_segments[:,0] += 7
                        lsd_line_segments[:,2] += 7
                        lsd_line_segments[:,1] += 6
                        lsd_line_segments[:,3] += 6
            else:
                image_rgb = None

            if lsd_line_segments is None:
                lsd_line_segments = []
                with open(self.lsd_line_files[id], 'r') as csv_file:
                    reader = csv.DictReader(csv_file, delimiter=' ')
                    for line in reader:
                        p1x = float(line['point1_x'])
                        p1y = float(line['point1_y'])
                        p2x = float(line['point2_x'])
                        p2y = float(line['point2_y'])
                        lsd_line_segments += [np.array([p1x, p1y, p2x, p2y])]
                lsd_line_segments = np.vstack(lsd_line_segments)

            labelled_line_segments = []
            with open(self.labelled_line_files[id], 'r') as csv_file:
                reader = csv.DictReader(csv_file, delimiter=' ')
                for line in reader:
                    lines_per_vp = []
                    for i in range(1,5):
                        key_x1 = 'line%d_x1' % i
                        key_y1 = 'line%d_y1' % i
                        key_x2 = 'line%d_x2' % i
                        key_y2 = 'line%d_y2' % i

                        if line[key_x1] == '':
                            break

                        p1x = float(line[key_x1])
                        p1y = float(line[key_y1])
                        p2x = float(line[key_x2])
                        p2y = float(line[key_y2])

                        ls = np.array([p1x, p1y, p2x, p2y])
                        lines_per_vp += []
                        if self.normalise_coords:
                            ls[0] -= 320
                            ls[2] -= 320
                            ls[1] -= 240
                            ls[3] -= 240
                            ls[0:4] /= 320.
                        lines_per_vp += [ls]
                    lines_per_vp = np.vstack(lines_per_vp)
                    labelled_line_segments += [lines_per_vp]

            if self.normalise_coords:
                lsd_line_segments[:,0] -= 320
                lsd_line_segments[:,2] -= 320
                lsd_line_segments[:,1] -= 240
                lsd_line_segments[:,3] -= 240
                lsd_line_segments[:,0:4] /= 320.

            line_segments = np.zeros((lsd_line_segments.shape[0], 7+2+3+3))
            for li in range(line_segments.shape[0]):
                p1 = np.array([lsd_line_segments[li,0], lsd_line_segments[li,1], 1])
                p2 = np.array([lsd_line_segments[li,2], lsd_line_segments[li,3], 1])
                centroid = 0.5*(p1+p2)
                line = np.cross(p1, p2)
                line /= np.linalg.norm(line[0:2])
                line_segments[li, 0:3] = p1
                line_segments[li, 3:6] = p2
                line_segments[li, 6:9] = line
                line_segments[li, 9:12] = centroid

            vp_list = []
            vd_list = []
            with open(self.vps_files[id]) as csv_file:
                reader = csv.reader(csv_file, delimiter=' ')
                for ri, row in enumerate(reader):
                    if ri == 0: continue
                    vp = np.array([float(row[1]), float(row[2]), 1])
                    if self.normalise_coords:
                        vp[0] -= 320
                        vp[1] -= 240
                        vp[0:2] /= 320.
                    vp_list += [vp]

                    vd = np.array(self.Kinv * np.matrix(vp).T)
                    vd /= np.linalg.norm(vd)
                    vd_list += [vd]
            vps = np.vstack(vp_list)
            vds = np.vstack(vd_list)

            datum = {'line_segments': line_segments, 'VPs': vps, 'id': id, 'VDs': vds, 'image': image_rgb,
                     'labelled_lines': labelled_line_segments}

            for vi in range(datum['VPs'].shape[0]):
                datum['VPs'][vi,:] /= np.linalg.norm(datum['VPs'][vi,:])

            if self.keep_in_mem:
                self.dataset[key] = datum

        return datum


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    mat_file_path = "/home/kluger/tmp/nyu_depth_v2_labeled.matv7.mat"

    assert (mat_file_path is not None), "Replace mat_file_path with the path where your 'nyu_depth_v2_labeled.mat' " + \
                                        "is stored in order to load the original RGB images, or comment this line"

    dataset = NYUVP("./data", mat_file_path="/home/kluger/tmp/nyu_depth_v2_labeled.matv7.mat",
                    split='all', normalise_coordinates=False, remove_borders=True)

    show_plots = True

    max_num_vp = 0
    all_num_vps = []

    for idx in range(len(dataset)):
        vps = dataset[idx]['VPs']
        num_vps = vps.shape[0]
        print("%04d -- vps: %d" % (idx, num_vps))
        all_num_vps += [num_vps]
        if num_vps > max_num_vp: max_num_vp = num_vps

        ls = dataset[idx]['line_segments']
        vp = dataset[idx]['VPs']

        if show_plots:
            image = dataset[idx]['image']
            ls_per_vp = dataset[idx]['labelled_lines']

            colours = ['#e6194b', '#4363d8', '#aaffc3', '#911eb4', '#46f0f0', '#f58231', '#3cb44b', '#f032e6',
                       '#008080', '#bcf60c', '#fabebe', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3',
                       '#808000', '#ffd8b1', '#000075', '#808080', '#ffffff', '#000000']

            fig = plt.figure(figsize=(16,5))
            ax1 = plt.subplot2grid((1,3), (0,0))
            ax2 = plt.subplot2grid((1,3), (0,1))
            ax3 = plt.subplot2grid((1,3), (0,2))
            ax1.set_aspect('equal', 'box')
            ax2.set_aspect('equal', 'box')
            ax3.set_aspect('equal', 'box')
            ax1.axis('off')
            ax2.axis('off')
            ax3.axis('off')
            ax1.set_title('original image')
            ax2.set_title('labelled line segments per VP')
            ax3.set_title('extracted line segments')

            if image is not None:
                ax1.imshow(image)
                ax2.imshow(rgb2gray(image), cmap='Greys_r')
            else:
                ax1.text(0.5, 0.5, 'not loaded', horizontalalignment='center', verticalalignment='center',
                         transform=ax1.transAxes, fontsize=12, fontweight='bold')

            for vpidx, lss in enumerate(ls_per_vp):
                c = colours[vpidx]
                for l in lss:
                    if image is None:
                        l[1] *= -1
                        l[3] *= -1
                    ax2.plot([l[0], l[2]], [l[1], l[3]], '-', c=c, lw=5)
            for li in range(ls.shape[0]):
                ax3.plot([ls[li,0], ls[li,3]], [-ls[li,1], -ls[li,4]], 'k-', lw=2)

            fig.tight_layout()
            plt.show()

    print("num VPs: ", np.sum(all_num_vps), np.sum(all_num_vps)*1./len(dataset), np.max(all_num_vps))

    plt.rcParams.update({'font.size': 18})
    plt.figure(figsize=(9, 3))
    values, bins, patches = plt.hist(all_num_vps, bins=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5])
    print(values)
    print(bins)
    plt.show()
