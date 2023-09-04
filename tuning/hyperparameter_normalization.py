# This file contains information pertinnent to the normalization of each optimizer's space of hyperparameters.

import numpy as np
import tensorflow as tf

import context
from training import hyperparameter_setup

# Make a dict of name : <number of hyperparameters> pairs
n_defaults = {key:len(defaults_) for key, defaults_ in hyperparameter_setup.autoregression_defaults.items()}
# Make a dict of name : <list of hyperparameter names> pairs
hyperparameter_names = {key:[name for name in fn.__code__.co_varnames] for key, fn in hyperparameter_setup.minimizer_setup_fns.items()}
# Make a dict of name : <list of how to unnormalize each hyperparameter> pairs
types = {key:["decay" if "momentum" in name or "beta" in name else "standard" for name in names_] for key, names_ in hyperparameter_names.items()}
# Make a dict of name : <list of how to unnormalize each hyperparameter> pairs
unnormalizing_fns = {key:[(lambda x: np.e**x) if type_=="standard" else (lambda x: np.maximum(0.0, 1-np.e**x)) for type_ in types_] for key, types_ in types.items()}
# Make a dict of name : <list of how to normalize each hyperparameter> pairs
normalizing_fns = {key:[(lambda x: np.log(x)) if type_=="standard" else (lambda x: np.log(np.maximum(1e-30, 1-x))) for type_ in types_] for key, types_ in types.items()}
# Make a dict of name : <list of normalized default hyperparameter> pairs
normalized_defaults = {key:[normalizing_fn(x) for normalizing_fn, x in zip(normalizing_fns[key], hyperparameter_setup.autoregression_defaults[key])] for key in hyperparameter_setup.names}
