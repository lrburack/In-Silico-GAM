# The StructureEnsemble object is a general object containing coordinates of genomic loci
# In this framework, In-Silico GAM is compatible with many structure formats without adding unneeded dependencies
import numpy as np

class StructureEnsemble:
    def __init__(self, structures):
        self.Nframes = 0
        self.Nbeads = 0
    
    def load_structure(self, ensemble_index, structure_out):
        structure_out[:] = 0
        

class Ensemble_numpy(StructureEnsemble):
    def __init__(self, structures):        
        self.structures = structures
        self.Nframes, self.Nbeads, dim = np.shape(structures)
        if dim != 3: raise ValueError("You've passed structures that aren't 3D")
    
    def load_structure(self, ensemble_index, structure_out):
        structure_out[:] = self.structures[ensemble_index]
        
class Ensemble_cndb(StructureEnsemble):
    def __init__(self, structures):
        from OpenMiChroM.CndbTools import cndbTools
        
        self.chromosomes = np.empty(len(structures), dtype=object)
        self.Nbeads = 0
        for i in range(len(structures)):
            self.chromosomes[i] = cndbTools()
            self.chromosomes[i].load(structures[i])
            self.Nbeads += int(self.chromosomes[i].Nbeads)
        self.Nframes = self.chromosomes[0].Nframes
        
    def load_structure(self, ensemble_index, structure_out):
        bead_offset = 0  # Keep track of where each chromosome’s data goes
        for chr_idx in range(len(self.chromosomes)):
            chr_beads = int(self.chromosomes[chr_idx].Nbeads)
            structure_out[bead_offset:bead_offset + chr_beads, :] = self.chromosomes[chr_idx].cndb[str(ensemble_index)]
            bead_offset += chr_beads

class Ensemble_pkl(StructureEnsemble):
    def __init__(self, structures):
        import pickle
        self.pickle = pickle
        
        self.structure_paths = structures
        self.Nframes = len(structures)
        with open(self.structure_paths[0], 'rb') as s:
            self.Nbeads, dim = np.shape(pickle.load(s))
        if dim != 3: raise ValueError("You've passed structures that aren't 3D")
        
    def load_structure(self, ensemble_index, structure_out):
        with open(self.structure_paths[ensemble_index], 'rb') as s:
            structure_out[:] = self.pickle.load(s)
    
    def load_all(self):
        structures = np.zeros((self.Nframes, self.Nbeads, 3))
        for i in range(self.Nframes):
            with open(self.structure_paths[i], 'rb') as s:
                structures[i] = self.pickle.load(s)
        return structures
            

# This function is for convenience to create the proper StructureEnsemble object for a structure format    
def prep_structures(raw_structures):
    if isinstance(raw_structures, StructureEnsemble):
        return raw_structures
    
    if not isinstance(raw_structures, np.ndarray):
        raw_structures = np.array(raw_structures)
        
    if np.issubdtype(raw_structures.dtype, np.number):
        return Ensemble_numpy(raw_structures)
    
    if raw_structures[0].endswith(".pkl"):
        return Ensemble_pkl(raw_structures)
    
    if raw_structures[0].endswith(".cndb"):
        return Ensemble_cndb(raw_structures)
    
    raise ValueError("Unrecognized structure format.")