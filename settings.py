BASE_DIR = '..'

# directories
LOGS_DIR = '%s/logs' % BASE_DIR
RESULTS_DIR = '%s/results' % BASE_DIR
MODELS_DIR = '%s/models' % BASE_DIR
TEMP_DIR = '/tmp'

# sequence parameters
MAX_SEQ_LEN = 180
bases = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
SEQ_DIM = len(bases)
