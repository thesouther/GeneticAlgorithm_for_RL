import pickle

def dump_model(model, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(model, f)

def load_model(file_name):
    with open(file_name, 'rb') as f:
        model = pickle.load(f)
    return model