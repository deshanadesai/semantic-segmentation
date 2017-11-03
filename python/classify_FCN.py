#!/usr/bin/env python
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
# caffe_root = os.environ['CAFFE_ROOT']
# sys.path.insert(0, os.path.join(caffe_root, 'python'))
import caffe

plt.rcParams['image.interpolation'] = 'nearest'  # don't interpolate

caffe.set_mode_cpu()
image_name = 'ski.jpg'
'''
if (os.environ.get('CAFFE_CPU_MODE')):
    caffe.set_mode_cpu()
else:
    caffe.set_mode_gpu()
'''

net_root = '../models'
model_def = net_root + '/deploy.prototxt'
model_weights = net_root + '/fcn-8s-pascalcontext.caffemodel'



# img = cv2.resize(cv2.imread(image_name),(500,500))
# cv2.imwrite('resized_'+image_name,img)

net = caffe.Net(model_def, model_weights, caffe.TEST)
mu = np.array([104.00698793, 116.66876762, 122.67891434])
# create transformer for the input called 'data'
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
# move image channels to outermost dimension
transformer.set_transpose('data', (2, 0, 1))
# subtract the dataset-mean value in each channel
transformer.set_mean('data', mu)
# rescale from [0, 1] to [0, 255]
transformer.set_raw_scale('data', 255)
# swap channels from RGB to BGR
transformer.set_channel_swap('data', (2, 1, 0))

image = caffe.io.load_image('resized_'+image_name)
transformed_image = transformer.preprocess('data', image)
# copy the image data into the memory allocated for the net
net.blobs['data'].data[...] = transformed_image

print net.blobs['data'].data[0]


print('Running image through net.')
output = net.forward()
print('Done.')

score = output['score'][0]
classed = np.argmax(score, axis=0)

w,h = classed.shape

import cv2
result = np.zeros([w,h])

for i in range(0,w):
	for j in range(0,h):
		if classed[i][j]==15 or classed[i][j]==28:
			result[i][j] = 255
		else:
			result[i][j] = 0

print result.shape
cv2.imwrite('grabcut_mask_'+image_name+'.jpg', result)

plt.imshow(transformer.deprocess('data', net.blobs['data'].data[0]),cmap='gray')
# plt.savefig('check.jpg')

# transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
# image = caffe.io.load_image(image_nsame)
# overlay_image = transformer.preprocess('data', image)

# print overlay_image.shape
# plt.figure()
# plt.imshow(image, cmap='gray')
plt.imshow(result, cmap='jet', alpha=0.5) # interpolation='none'
plt.savefig('Overlay_mask_'+image_name+'.jpg')
	
names = dict()
all_labels = ["0: Background"] + open(net_root + '/legend.txt').readlines()
scores = np.unique(classed)
labels = [all_labels[s] for s in scores]
num_scores = len(scores)


def rescore(c):
	""" rescore values from original score values (0-59) to values ranging from
	0 to num_scores-1 """
	return np.where(scores == c)[0][0]

rescore = np.vectorize(rescore)
painted = rescore(classed)

print " Printing Classed "
print classed
print " Printing Painted"
print painted


plt.figure(figsize=(10, 10))
plt.imshow(painted)
formatter = plt.FuncFormatter(lambda val, loc: labels[val])
plt.colorbar(ticks=range(0, num_scores), format=formatter)
plt.clim(-0.5, num_scores - 0.5)

# plt.savefig(sys.argv[2])
plt.savefig('result.jpg')
