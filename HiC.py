from GAM import GAM
import numpy as np
import scipy.spatial.distance

class HiC:
    def __init__(self, ligation_probability_fxn, detection_probability):
        self.ligation_probability_fxn = ligation_probability_fxn
        # self.digested_fragments_per_bead = digested_fragments_per_bead
        self.detection_probability = detection_probability
    
    def theoretical_ligation(self, structures, frames=None, beads=None):
        """ Predict the ligation frequency (the converged limit) and its variance
        
        :param structures: Array of structures. Can currently be:
            - paths to pkl objects each containing a beadsx3 numpy array of xyz coordinates
            - paths to cndb files
            - a (frames x beads x 3) numpy array
        :param NPs: The number of nuclear profiles to take per structure.
        :param beads: The indexes of beads to include in the experiment. If None, all beads will be used
        :param frames: The indexes of frames to include in the experiment. If None, all frames will be used
        :return: A dictionary with the following fields:
            {
                'ligation_freq': float array,
                'ligation_freq_variance': float array,
                'ligation_freq_variance_one_cell': float array,
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

        upper_triu_length = int((len(beads) ** 2 - len(beads)) / 2)
        ligation_freq = np.zeros(upper_triu_length)
        ligation_freq_variance = np.zeros(upper_triu_length)

        for frame in range(len(frames)):
            load_structure_fxn(structures_object, frame, structure)

            ligation_probabilities = self.ligation_probability_fxn(scipy.spatial.distance.pdist(structure[beads]))
            # Account for finite detection
            ligation_probabilities *= self.detection_probability
            
            ligation_freq += ligation_probabilities
            ligation_freq_variance += ligation_probabilities * (1 - ligation_probabilities)
        
        normalize = len(frames)
        ligation_freq = scipy.spatial.distance.squareform(ligation_freq / normalize)
        ligation_freq_variance = scipy.spatial.distance.squareform(ligation_freq_variance / (normalize**2))
        
        return {'ligation_freq': ligation_freq, 
                'ligation_freq_variance': ligation_freq_variance,
                'ligation_freq_variance_one_cell': ligation_freq_variance * normalize
               }
    
#     @staticmethod
#     def step(distances):
#         threshold = (20 / 1000) / 0.14
#         return distances < threshold
    
    @staticmethod
    def dipierro2016(distances):
        mu = 3.22
        r_c = 1.78
        return 0.5 * (1 + np.tanh(mu * (r_c - distances)))
    
    # @staticmethod
    # def actual(distances):
    #     sigma = 0.1
    #     # r_c is 20nm
    #     # 1 unit in the simulation is about 0.14 microns
    #     r_c = (20 / 1000) / 0.14
    #     return np.tanh((distances - r_c) / sigma)