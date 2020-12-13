import pickle, os

def save(data, outfile):
    outfile_dir = os.path.dirname(outfile)
    if not os.path.exists(outfile_dir):
        os.makedirs(outfile_dir)
    with open(outfile, 'wb') as f:
        pickle.dump(data, f)

def load(outfile):
    with open(outfile, 'rb') as f:
        return pickle.load(f)