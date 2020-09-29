import numpy as np
from utils import MinMaxScaler,NormalizationScaler
features=[[2, -1], [-1, 5], [0, 0]]
result=[[1, 0], [0, 1], [0.333333, 0.16667]]
features_to_zero = np.argwhere(np.max(features,axis=0)-np.min(features,axis=0) == 0)
print(MinMaxScaler()(features))

# for i in range(features.shape[1]):
#     print(features[:,i])
#     print(np.min(features[:,i]))
#     print(np.subtract(features[:,i],np.min(features[:,i]))/(np.max(features[:,i])-np.min(features[:,i])))


temp = [1,2,3124141,7,124,213,532]
print(temp[:-1])
print(np.argpartition(temp,3)[:10])