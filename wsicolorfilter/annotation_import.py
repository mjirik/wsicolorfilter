import numpy as np


def import_pixel_train_data():
    # load annotations
    img = np.load('image.npy')
    seeds = np.load('seeds.npy')

    # remove not labeled pixels
    img = img[seeds > 0.0]
    seeds = seeds[seeds > 0.0]

    with open('train_data.csv', 'a') as fd:
        for i in range(img.shape[0]):
            row = ','.join(map(str, img[i, :])) + ',' + str(seeds[i]) + '\n'
            fd.write(row)


if __name__ == '__main__':
    import_pixel_train_data()
