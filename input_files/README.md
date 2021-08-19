# Input File Format
The DataLoaders parse JSON input files to specify some hyperparameters, paths, and parsing options. The descriptions for the supported options are given below:

- **input_paths :** &nbsp; [ *list of strings* ] Contains paths to the .root files of the inputs (events data) {multiple input files implementation under progress}

- **output_paths :** &nbsp; [ *list of strings* ] Contains paths to the .root files of the outputs (weights) corresponding to the inputs above

- **tree :** &nbsp; [ *string* ] Specifies the tree/directory to parse

- **particles :** &nbsp; [ *list of strings* ] Contains the names of the particles that are being considered for input

- **4v_coords :** &nbsp; [ *list of strings* ] The co-ordinate names of the 4 vectors of each particle. Should have 1 timelike and 3 spacelike components. This is a project specific implementation for convenience.

- **dir_name_processing :** &nbsp; [ *boolean* ] Further processing on the particle name string (if required) to identify particle directory.

Put simply the above three options will identify the directory that needs to be parsed in the .root file.

For example if particle name is "lepton", co-ordinate  names are 4v_coords = ["E", "Px", "Py", "Pz"], dir_name_processing is true, then the dataloader will extract data from:

` "tree/" + dir_name_processing("lepton") + 4v_coords[i] ` for each i from the root file.

- **weights :** &nbsp; [ *list of strings* ] Specifies the directory with weights associated with each event

- **batch_size :** &nbsp; [ *integer* ] Batch size hyperparameter for training

- **chunk_entries :** &nbsp; [ *integer* ] This is for the hybrid dataloader implementation. Specifies the number of entries in one loaded chunk

- **shuffle :** &nbsp; [ *boolean* ] Whether the loaded data needs to be shuffled
