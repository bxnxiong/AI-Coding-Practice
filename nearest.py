#!/usr/bin/python
"""
K-Nearest Neighbor Implementation
"""


import time
import sys
import numpy as np
import math
import random
from heapq import *
from collections import Counter

def train(traind,fraction=1):
    starttime = time.time()
    print " Training...",
    #labels = [int(l[1]) for l in traind]
    #colors = [[int(i) for i in l[2:]] for l in traind]
    labels,colors = [],[]
    for l in traind:
    	choose = random.random()
    	if choose <= fraction:
    		labels += [int(l[1])]
    		colors += [[int(i) for i in l[2:]]]

    print "Done in", round(time.time() - starttime, 5), "seconds!"
    return labels,colors

def test(traind_labels, traind_colors, testd, K,fraction):
	starttime = time.time()
	print " Testing...",

	traind_colors = [np.array(l) for l in traind_colors]
	testd_ids = [l[0] for l in testd]
	testd_labels = [int(l[1]) for l in testd]
	testd_colors = [[int(i) for i in l[2:]] for l in testd]

	i = 0
	predict = []
	#print len(traind_colors),len(traind_labels)
	#print traind_labels[0],traind_colors[0]
	while i < len(testd_colors):
		to_test = testd_colors[i]
		vector1 = np.array(to_test)
		q = [] # heap queue to store top K nearest neighbors
		for j in range(len(traind_colors)):
			vector2 = traind_colors[j]
			diff = vector1-vector2
			dist = -math.sqrt(np.dot(diff.T,diff)) # take negative because python uses a min heap
			if len(q) < K:
				heappush(q,(j,dist))
			else:
				if dist > q[0][1]:
					heappushpop(q,(j,dist))
		nearest_labels = [traind_labels[j] for j,dist in q]
		maj_label = Counter(nearest_labels).most_common()[0][0] # most_common() returns [(item, freq),..]
		predict.append([testd_ids[i],maj_label])

		i += 1
		filename = 'nearest_output_{}_{}.txt'.format(K,fraction)
		
		try:
			output_file = open(filename,'w')
		except(IOError) as e:
			print 'Unable overwrite or create output file.\n'
			sys.exit()

		for p in predict:
			output_file.write(str(p[0]) + '\t' + str(p[1]) + '\n')
		output_file.close()


	print "Done in", round(time.time() - starttime, 5), "seconds!"
