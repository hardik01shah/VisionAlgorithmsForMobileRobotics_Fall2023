import numpy as np
from scipy.spatial.distance import cdist


def matchDescriptors(query_descriptors, database_descriptors, match_lambda):
    """
    Returns a 1xQ matrix where the i-th coefficient is the index of the database descriptor which matches to the
    i-th query descriptor. The descriptor vectors are MxQ and MxD where M is the descriptor dimension and Q and D the
    amount of query and database descriptors respectively. matches(i) will be -1 if there is no database descriptor
    with an SSD < lambda * min(SSD). No elements of matches will be equal except for the -1 elements.
    """
    pass
    # TODO: Your code here
    distances = cdist(query_descriptors.T, database_descriptors.T)
    thresh = np.min(distances)*match_lambda

    matches = np.argmin(distances, axis=1)
    nval_indx = np.where(np.min(distances, axis=1)>thresh)[0]
    matches[nval_indx] = -1

    # remove double matches
    unique_matches = np.ones_like(matches) * -1
    _, unique_match_idxs = np.unique(matches, return_index=True)
    unique_matches[unique_match_idxs] = matches[unique_match_idxs]
    
    return matches