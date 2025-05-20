import numpy as np
from scipy.spatial.transform import Rotation as r
import scipy.spatial.distance
from utilities import fast_dists
import pickle
from Model import Model
import time
from OpenMiChroM.CndbTools import cndbTools


class GAM:
    def __init__(self, slice_width=1, multiplexing=1, detection_probability=1, pick_slice=None, nuclear_radius=None,
                 homolog_map=None):
        """ Instantiate GAM object
        :param slice_width: The width of the slices in the same units used by structures
        :param multiplexing: The number of NPs per sequenced 'tube'
        :param detection_probability: The probability that a sectioned bead is successfully detected (or 'sequenced')
        :param nuclear_radius: Optional, used by some pick_slice functions. The radius of the nucleus in the same
        units used by structures
        :param pick_slice: The function called to get the position of the bottom of each slice.
        Defaults to uniform_edges
        :param homolog_map: A numpy integer array of length beads. Loci in the structure with the same homolog_map value
        will be treated as indistinguishable. Every integer value between zero and the maximum value should appear at
        least once
        """
        self.slice_width = slice_width
        self.pick_slice = pick_slice if pick_slice is not None else GAM.uniform_edges
        self.nuclear_radius = None
        if pick_slice is GAM.uniform_radius:
            if nuclear_radius is None:
                raise ValueError('Must provide nuclear_radius when using pick_slice function GAM.uniform_radius')
            else:
                self.nuclear_radius = nuclear_radius

        self.multiplexing = multiplexing
        if detection_probability > 1 or detection_probability < 0:
            raise ValueError("detection_probability must be between zero and one")
        self.detection_probability = detection_probability
        self.homolog_map = homolog_map

    def run(self, structures, beads=None, frames=None, NPs=1):
        """ Run GAM over an ensemble of structures
        
        :param structures: Array of structures. Can currently be:
            - paths to pkl objects each containing a beadsx3 numpy array of xyz coordinates
            - paths to cndb files
            - a (frames x beads x 3) numpy array
        :param NPs: The number of nuclear profiles to take per structure.
        :param beads: The indexes of beads to include in the experiment. If None, all beads will be used
        :param frames: The indexes of frames to include in the experiment. If None, all frames will be used
        :return: A dictionary with the following fields:
            {
                'raw': boolean array,
                'processed': boolean array,
            }
        """
        structures_object, Nbeads, Nframes, load_structure_fxn = GAM.prep_structures(structures)
        
        # If not specified, use them all
        if beads is None:
            beads = np.arange(Nbeads)
        if frames is None:
            frames = np.arange(Nframes)
        
        # Preallocate memory for the structure and the sectioning data
        structure = np.empty((Nbeads, 3), dtype=np.float64)
        sec = np.empty([NPs * len(frames), len(beads)], dtype=bool)

        for i, frame in enumerate(frames):
            # This will load the frame'th structure and put it in the preallocated structure variable
            load_structure_fxn(structures_object, frame, structure)

            for j in range(NPs):
                sec[i * NPs + j, :] = self.NP(structure[beads])

        processed = GAM.detect(sec, self.detection_probability)
        processed = GAM.collapse_homologs(processed, self.homolog_map)
        processed = GAM.multiplex(processed, self.multiplexing)

        return {'raw': sec, 'processed': processed}
    
    def theoretical_cosegregation(self, structures, frames=None, beads=None, NPs=1):
        """ Predict the cosegregation probability (the converged limit) and its variance
        
        :param structures: Array of structures. Can currently be:
            - paths to pkl objects each containing a beadsx3 numpy array of xyz coordinates
            - paths to cndb files
            - a (frames x beads x 3) numpy array
        :param NPs: The number of nuclear profiles to take per structure.
        :param beads: The indexes of beads to include in the experiment. If None, all beads will be used
        :param frames: The indexes of frames to include in the experiment. If None, all frames will be used
        :return: A dictionary with the following fields:
            {
                'p_co': float array,
                'p_co_variance': float array,
            }
        """
        
        structures_object, Nbeads, Nframes, load_structure_fxn = GAM.prep_structures(structures)
        
        # If not specified, use them all
        if beads is None:
            beads = np.arange(Nbeads)
        if frames is None:
            frames = np.arange(Nframes)
        
        # Preallocate memory for the structure
        structure = np.empty((Nbeads, 3), dtype=np.float64)

        def p_2(D, l, h):
            m = np.minimum(D, h)
            result = (2 * h * m - m ** 2) / (2 * D * l)
            return result

        upper_triu_length = int((len(beads) ** 2 - len(beads)) / 2)
        p_co = np.zeros(upper_triu_length)
        p_co_variance = np.zeros(upper_triu_length)

        for frame in range(len(frames)):
            load_structure_fxn(structures_object, frame, structure)

            cosegregation_probabilities = p_2(scipy.spatial.distance.pdist(structure[beads]), 2 * self.nuclear_radius + self.slice_width, self.slice_width)
            p_co += NPs * cosegregation_probabilities
            p_co_variance += NPs * (cosegregation_probabilities * (1 - cosegregation_probabilities))
        
        normalize = NPs * len(frames)
        p_co = scipy.spatial.distance.squareform(p_co / normalize)
        p_co_variance = scipy.spatial.distance.squareform(p_co_variance / (normalize**2))
        # Fill the diagonal
        p = self.slice_width / (2 * self.nuclear_radius + self.slice_width)
        p_co[np.arange(len(p_co)), np.arange(len(p_co))]          = p
        p_co_variance[np.arange(len(p_co)), np.arange(len(p_co))] = (p * (1-p)) / (normalize)
        
        return {'p_co': p_co, 
                'p_co_variance': p_co_variance}
    
    def NP(self, structure, slice_axis=2):
        """ Takes a nuclear profile of the structure
        :param structure: A beadsx3 numpy array containing the xyz positions of the beads
        :param slice_axis: x=0 y=1 z=2 The axis perpendicular to the slice plane
        :return: A boolean array containing True where beads appeared in the slice
        """
        rotation = r.random().apply(structure)
        slice_pos = self.pick_slice(self.slice_width, self.nuclear_radius, rotation, slice_axis)
        sec = np.logical_and(rotation[:, slice_axis] > slice_pos,
                             rotation[:, slice_axis] < slice_pos + self.slice_width)

        return sec

    def illustrate_NP(self, ax, structure, slice_pos=None, slice_axis=2):
        """ Illustrates a nuclear profile on the given axes"""

        if slice_pos is None:
            slice_pos = GAM.uniform_edges(self.slice_width, self.nuclear_radius or None, structure, slice_axis)
        sectioned = np.logical_and(structure[:, slice_axis] > slice_pos,
                                   structure[:, slice_axis] < slice_pos + self.slice_width)

        ax.scatter(*structure[sectioned].T, zorder=9, color='red')

        x_bounds = [np.min(structure[:, 0]), np.max(structure[:, 0]), np.min(structure[:, 0]) - 1e-9,
                    1e-9 + np.max(structure[:, 0])]
        y_bounds = [np.min(structure[:, 1]), np.max(structure[:, 1]), np.max(structure[:, 1]) + 1e-9,
                    np.min(structure[:, 1]) - 1e-9]

        # Plot profile
        ax.plot_trisurf(x_bounds, y_bounds, [slice_pos] * 4, zorder=10, color='yellow', alpha=.4)
        ax.plot_trisurf(x_bounds, y_bounds, [slice_pos + self.slice_width] * 4, zorder=10, color='yellow', alpha=.4)

        return sectioned

    def make_model(self, model_class=None):
        return Model.make_model(self) if model_class is None else model_class.make_model(self)

    def __str__(self):
        return ', '.join("%s: %s" % item for item in vars(self).items())

    
    @staticmethod
    def results(sec):
        """ Computes results of a GAM experiment from sectioning data

        :param sec: samples x beads numpy array of sectioning data
        :return: sectioning_counts, cosectioning_counts, sectioning_frequency, cosectioning_frequency,
        normalized_cosectioning
        """
        print("Starting")
        a = time.time()
        beads = np.shape(sec)[1]
        slices = np.shape(sec)[0]
        sectioning_counts = np.sum(sec, axis=0, dtype=np.ushort)

        # Calculate m_is. This is done using a tiled array to quickly compare all pairs of loci
        print(f"computing m_is: {time.time() - a}")
        m_i = np.zeros((beads, beads, 3), dtype=np.float32)
        for i in range(len(sec)):
            m_i[:, :, 0] += np.outer(~sec[i], ~sec[i])
            # m_i[:, :, 1] += np.outer(sec[i], ~sec[i]) | np.outer(~sec[i], sec[i])
            m_i[:, :, 2] += np.outer(sec[i], sec[i])
        
        m_i[:, :, 1] = slices - (m_i[:, :, 0] + m_i[:, :, 2])    
        
        print(f"copying m_2 as int: {time.time() - a}")
        cosectioning_counts = np.copy(m_i[:, :, 2]).astype(np.ushort)
        m_i /= slices
        
        print(f"computing normalized cosectioning: {time.time() - a}")
        tile_sec = np.tile(sectioning_counts, (beads, 1))
        either = tile_sec + tile_sec.T - cosectioning_counts
        normalized_cosectioning = np.nan_to_num(cosectioning_counts / either)
        
        print(f"done: {time.time() - a}")
        
        return {
            'sectioning_counts': sectioning_counts,
            'cosectioning_counts': cosectioning_counts,
            'sectioning_frequency': sectioning_counts / len(sec),
            'normalized_cosectioning': normalized_cosectioning,
            'm_i': m_i
        }

    @staticmethod
    def results2(sec):
        """ Computes results of a GAM experiment from sectioning data

        :param sec: samples x beads numpy array of sectioning data
        :return: sectioning_counts, cosectioning_counts, sectioning_frequency, cosectioning_frequency,
        normalized_cosectioning
        """
        print("Starting")
        a = time.time()
        sec = sec.astype(bool)  # Ensure boolean operations
        beads = sec.shape[1]
        slices = sec.shape[0]

        # Compute sectioning counts
        sectioning_counts = np.sum(sec, axis=0, dtype=np.ushort)
        
        print(f"computing m_is: {time.time() - a}")
        # Compute m_i using vectorized broadcasting instead of looping
        sec_expanded = sec[:, :, None]  # Expand sec to shape (samples, beads, 1)
        sec_T_expanded = sec[:, None, :]  # Expand sec to shape (samples, 1, beads)

        # Compute the m_i components using logical operations
        m_i_0 = np.sum(~sec_expanded & ~sec_T_expanded, axis=0, dtype=np.float32)
        m_i_2 = np.sum(sec_expanded & sec_T_expanded, axis=0, dtype=np.float32)
        m_i_1 = slices - (m_i_0 + m_i_2)  # XOR for different values

        # Stack m_i results
        m_i = np.stack((m_i_0, m_i_1, m_i_2), axis=-1)
        
        print(f"not copying m_2 as int: {time.time() - a}")
        cosectioning_counts = m_i[:, :, 2].astype(np.ushort)
        m_i /= slices
        
        print(f"computing normalized cosectioning: {time.time() - a}")
        # Compute normalized cosectioning frequency
        tile_sec = np.tile(sectioning_counts, (beads, 1))
        either = tile_sec + tile_sec.T - cosectioning_counts
        normalized_cosectioning = np.nan_to_num(cosectioning_counts / either)
        
        print(f"done: {time.time() - a}")
        
        return {
            'sectioning_counts': sectioning_counts,
            'cosectioning_counts': cosectioning_counts,
            'sectioning_frequency': sectioning_counts / sec.shape[0],
            'normalized_cosectioning': normalized_cosectioning,
            'm_i': m_i
        }
    
    @staticmethod
    def faster_results2(sec):
        """ Computes results of a GAM experiment from sectioning data

        :param sec: samples x beads numpy array of sectioning data
        :return: sectioning_counts, cosectioning_counts, sectioning_frequency, cosectioning_frequency,
        normalized_cosectioning
        """
        print("Starting")
        a = time.time()
        sec = sec.astype(np.uint8)  # Use uint8 for efficient memory usage
        N, beads = sec.shape

        # Compute sectioning counts
        sectioning_counts = np.sum(sec, axis=0, dtype=np.ushort)

        # Compute m_i components using np.tensordot
        sec_T = sec.T  # Transpose to avoid redundant computation

        m_i_2 = np.tensordot(sec_T, sec, axes=(1, 0))  # Equivalent to sum of outer products
        m_i_0 = np.tensordot(1 - sec_T, 1 - sec, axes=(1, 0))
        m_i_1 = N - (m_i_0 + m_i_2)  # Efficiently derive this instead of computing explicitly

        # Stack m_i results
        m_i = np.stack((m_i_0, m_i_1, m_i_2), axis=-1).astype(np.float32) / N

        cosectioning_counts = m_i_2.astype(np.ushort)

        # Compute normalized cosectioning frequency
        either = sectioning_counts[:, None] + sectioning_counts - cosectioning_counts
        normalized_cosectioning = np.nan_to_num(cosectioning_counts / either)
        
        print(f"done: {time.time() - a}")

        return {
            'sectioning_counts': sectioning_counts,
            'cosectioning_counts': cosectioning_counts,
            'sectioning_frequency': sectioning_counts / N,
            'normalized_cosectioning': normalized_cosectioning,
            'm_i': m_i
        }

    
    @staticmethod
    def triplets(sec):
        # Reshape the array for broadcasting
        arr_i = sec[:, np.newaxis, np.newaxis]  # Shape (N, 1, 1)
        arr_j = sec[np.newaxis, :, np.newaxis]  # Shape (1, N, 1)
        arr_k = sec[np.newaxis, np.newaxis, :]  # Shape (1, 1, N)

        # Compute the 3D correlation matrix
        return arr_i & arr_j & arr_k

    @staticmethod
    def rebin(sec, binsize, new_binsize):
        bincount = np.shape(sec)[1]
        slicecount = np.shape(sec)[0]

        bin_ratio = int(new_binsize // binsize)
        new_bincount = (bincount // bin_ratio) + 1
        rebinned = np.zeros((slicecount, new_bincount), dtype=bool)

        for i in range(0, new_bincount):
            rebinned[:, i] = np.any(sec[:, i * bin_ratio:(i + 1) * bin_ratio], axis=1)

        return rebinned

    @staticmethod
    def collapse_homologs(sec, homolog_map):
        """ Collapses loci into a sectioning array that does not distinguish between homologs

        :param sec: A boolean array in which each row contains sectioning data
        :param homolog_map: A numpy integer array of length beads. For each unique entry j in the array, all loci with
        that homolog_map value j will be collapsed (logical or) into position j in the collapsed sectioning array
        :return: A collapsed array of sectioning data that does not distinguish between homologs
        """
        if homolog_map is None:
            return sec

        collapsed = np.empty([len(sec), np.max(homolog_map) + 1], dtype=bool)
        for ind in np.arange(np.max(homolog_map) + 1):
            collapsed[:, ind] = np.any(sec[:, homolog_map == ind], axis=1)
        return collapsed

    @staticmethod
    def multiplex(sec, multiplexing):
        """ Aggregates sectioning from NPs as though it was pooled in a single tube
        
        :param sec: A boolean array in which each row contains sectioning data from a separate NP
        :param multiplexing: The number of NPs to be pooled together
        :return: A boolean array of multiplexed sectioning data
        """

        if multiplexing == 1:
            return sec

        # Randomly reorder sectioning data
        np.random.shuffle(sec)

        samples = len(sec) // multiplexing
        multiplexed = np.empty((samples, len(sec[0])), dtype=bool)
        for i in range(samples):
            multiplexed[i, :] = np.any(sec[i * multiplexing:i * multiplexing + multiplexing, :], axis=0)
        return multiplexed

    @staticmethod
    def detect(sec, detection_probability):
        """ Simulates a finite sequencing efficiency by discarding data

        :param sec: A boolean array in which each row contains sectioning data
        :param detection_probability: The probability that a bead is successfully detected
        :return: A boolean array of filtered sectioning data
        """

        if detection_probability == 1:
            return sec

        return np.logical_and(sec, np.random.uniform(size=np.shape(sec)) < detection_probability)

    # some sample pick_slice functions
    @staticmethod
    def uniform_radius(slice_width, nuclear_radius, structure, slice_axis):
        """ pick_slice function
        :return a position for the bottom of the slice uniformly between the bounds of the sphere defined by
        nuclear_radius
        """
        return np.random.uniform(-nuclear_radius - slice_width, nuclear_radius)

    @staticmethod
    def uniform_edges(slice_width, nuclear_radius, structure, slice_axis):
        """ pick_slice function
        :return a position for the bottom of the slice uniformly between the minimum and maximum positions of beads
        along the slice axis
        """
        return np.random.uniform(np.min(structure[:, slice_axis]) - slice_width,
                                 np.max(structure[:, slice_axis]))

    
    # All stuff for loading structures is below
    @staticmethod
    def prep_structures(raw_structures):
        if not isinstance(raw_structures, np.ndarray):
            try:
                raw_structures = np.array(raw_structures)
            except:
                raise ValueError("Structures must be a numpy array of paths or xyz coordinates")
        
        # For a particular file type we need to get structures_object, Nbeads, Nframes, load_structure_fxn
        if isinstance(raw_structures[0], str):
            # PKL
            if raw_structures[0].endswith(".pkl"):
                structures_object = raw_structures
                load_structure_fxn = GAM.load_structure_pkl
                Nframes = len(raw_structures)
                s = open(raw_structures[0], 'rb')
                Nbeads = len(pickle.load(s))
                s.close()
            # CNDB
            elif raw_structures[0].endswith(".cndb"):
                # Here, structures_object is a numpy array of length Nchromosomes containing cndbTools objects
                structures_object = np.empty(len(raw_structures), dtype=object)
                load_structure_fxn = GAM.load_structure_cndb
                Nbeads = 0
                for i in range(len(raw_structures)):
                    structures_object[i] = cndbTools()
                    structures_object[i].load(raw_structures[i])
                    Nbeads += int(structures_object[i].Nbeads)
                Nframes = structures_object[0].Nframes
            else:
                raise ValueError("Only configured to handle .pkl or .cndb")
        # NUMPY
        else:
            structures_object = raw_structures
            load_structure_fxn = GAM.load_structure_numpy
            Nframes, Nbeads, dim = np.shape(raw_structures)
            if dim != 3:
                raise ValueError("You've somehow passed structures that aren't 3D")
        
        return structures_object, Nbeads, Nframes, load_structure_fxn
    
    @staticmethod
    def load_structure_pkl(structures_object, ensemble_index, structure_out):
        # Here, structures_object is a numpy array of paths to pkl objects
        s = open(structures_object[ensemble_index], 'rb')
        structure_out[:] = pickle.load(s)
        s.close()
    
    @staticmethod
    def load_structure_cndb(structures_object, ensemble_index, structure_out):
        # Here, structures_object is a numpy array of length Nchromosomes containing cndbTools objects
        bead_offset = 0  # Keep track of where each chromosome’s data goes
        for chr_idx in range(len(structures_object)):
            chr_beads = int(structures_object[chr_idx].Nbeads)
            structure_out[bead_offset:bead_offset + chr_beads, :] = structures_object[chr_idx].cndb[str(ensemble_index)]
            bead_offset += chr_beads

    @staticmethod
    def load_structure_numpy(structures_object, ensemble_index, structure_out):
        structure_out[:] = structures_object[ensemble_index]