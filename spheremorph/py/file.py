import numpy as np


def read_freesurfer_label(filename):
    with open(filename, 'r') as f:
        lines = f.read().splitlines()[2:]
    idx_vert = np.array([int(l.split()[0]) for l in lines], dtype=np.int32)
    coords = np.array([list(map(float, l.split()[1:4])) for l in lines], dtype=np.float32)
    return idx_vert, coords


def write_freesurfer_label(filename, idx_vert, coords, comment='# spheremorph fs label writer'):
    if len(idx_vert.shape) != 1:
        raise ValueError("idx_vert must be a 1D array")

    if len(coords.shape) != 2 or coords.shape[1] != 3:
        raise ValueError("coords must be a 3D array")

    if idx_vert.shape[0] != coords.shape[0]:
        raise ValueError("idx_vert and coords must have the same length in the first dimension")

    with open(filename, 'w') as f:
        f.write(comment + '\n')
        f.write(f'{len(idx_vert)}\n')
        for i, c in zip(idx_vert, coords):
            f.write(f'{i} {c[0]} {c[1]} {c[2]} 0\n')
