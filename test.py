import numpy as np

test = np.array(((1, None, 1, None), (1, None, 1, None)))
test[test == None] = 0
truncated_ids = np.flatnonzero(test)

print(truncated_ids)
