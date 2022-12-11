import torch
from torch.autograd import Variable
import numpy as np
import pandas as pd
import time
import gc

import os
import freerec
from freerec.data.tags import USER, ITEM, ID
from freerec.utils import mkdirs

root = '../../data'
dataset = 'Gowalla_m1'
# dataset = 'Yelp18_m1'
# dataset = 'AmazonBooks_m1'
# dataset = 'MovieLens1M_m2'
# dataset = 'AmazonCDs_m1'
# dataset = 'AmazonBeauty_m1'
basepipe = getattr(freerec.data.datasets, dataset)(root)

User, Item = basepipe.fields[USER, ID], basepipe.fields[ITEM, ID]
user, item = User.count, Item.count


for alpha in [0, 5, 10, 15, 20]:
	rate_matrix=torch.zeros(user, item).cpu()

	for row in basepipe.train().raw2data():
		x, y = row[User.name], row[Item.name]
		rate_matrix[x, y] = 1


	#save interaction matrix
	path = os.path.join(dataset, str(alpha))
	mkdirs(path)
	file_ = os.path.join(dataset, "rate_sparse.npy")
	np.save(file_,rate_matrix.cpu().numpy())


	D_u=rate_matrix.sum(1)+alpha
	D_i=rate_matrix.sum(0)+alpha




	for i in range(user):
		if D_u[i]!=0:
			D_u[i]=1/D_u[i].sqrt()

	for i in range(item):
		if D_i[i]!=0:
			D_i[i]=1/D_i[i].sqrt()


	#\tilde{R}
	rate_matrix=D_u.unsqueeze(1)*rate_matrix*D_i


	#free space
	del D_u,D_i
	gc.collect()
	torch.cuda.empty_cache()


	'''
	q:the number of singular vectors in descending order.
	to make the calculated singular value/vectors more accurate,
	q shuld be (slightly) larger than K.
	'''

	print('start!')
	start=time.time()
	U,value,V=torch.svd_lowrank(rate_matrix, q=1000, niter=30)
	#U,value,V=torch.svd(R)
	end=time.time()
	print('processing time is %f' % (end-start))
	print('singular value range %f ~ %f' % (value.min(),value.max()))


	np.save(path + r'/svd_u.npy',U.cpu().numpy())
	np.save(path + r'/svd_v.npy',V.cpu().numpy())
	np.save(path + r'/svd_value.npy',value.cpu().numpy())
