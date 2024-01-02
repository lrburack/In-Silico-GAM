import pickle
import numpy as np
import gsd.hoomd
from prody import parseDCD


def to_pkl(structures, outpath, fnames=None):
    """Converts an array of structures to pkl files (the preferred filetype for In-Silico GAM)
    :param structures: The structure array may contain:
    1. nx3 numpy arrays containing xyz coordinates for beads
    2. paths to gsd files
    3. paths to dcd files
    All structures must have the same number of beads.
    :param outpath: Location for pkl files
    :param fnames: Optional, filenames for the resulting pickle files
    """

    if fnames is None:
        fnames = np.char.join(np.arange(len(structures), dtype=int).astype("str"), ".pkl")

    fnames = np.char.join(outpath, fnames)

    for i in range(len(structures)):
        s = structures[i]
        save = None
        if isinstance(s, np.ndarray):
            save = s
        else:
            if s.endswith(".gsd"):
                frames = gsd.hoomd.open(s, 'rb')
                last_frame = frames[-1]
                save = last_frame.particles.position
            elif s.endswith(".dcd"):
                structure = parseDCD(s)
                save = structure.getCoordsets(0)

        if save is not None:
            with open(fnames, 'wb') as f:
                pickle.dump(save, fnames[i])


def random_unit_vector(n, dim=3):
    v = np.random.normal(0, 1, (n, dim))
    return np.divide(v, np.linalg.norm(v, axis=1)[:, None])


def random_walk(length=100, delta=1, origin=np.zeros(3)):
    """ Generates a random walk polymer

    :param length: Number of beads
    :param delta: Distance between consecutive beads
    :param origin: Starting point
    :return: A random walk polymer
    """
    structure = origin + np.concatenate((np.zeros([1, 3]), np.cumsum(random_unit_vector(length - 1), axis=0)))

    return structure


def center_structure(structure):
    """ Puts the center of mass of the structure at the origin"""
    com = np.sum(structure, axis=0) / len(structure)
    return structure - com


def crop_structure(structure, radius):
    """ Removes points outside the radius"""
    return structure[np.linalg.norm(structure, axis=1) < radius, :]