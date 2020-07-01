## coding: UTF-8
import os.path
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms

import random

import numpy as np

import parameters

from PIL import Image


epochs = parameters.epochs
image_size = parameters.image_size
batch_size = parameters.batch_size
workers = parameters.workers
crop_padding = parameters.crop
My_lambda = parameters.My_lambda
data_max = parameters.data_max


Mytransform = transforms.Compose([
   transforms.ToTensor(),
   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
   ])

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def SHOW(tensor_image):
	# image_numpy = tensor_image.to("cpu").detach().numpy().copy()
	# image_numpy = tensor_image.detach().numpy().copy()
	# image_numpy = np.transpose(image_numpy, (2,1,0))
	# image = Image.fromarray(image_numpy.astype(np.uint8))
	tensor_image.show()

class Mydatasets(torch.utils.data.Dataset):
	def __init__(self, transform = None, train = True):
		self.transform = transform
		self.train = train

		self.trainAset = []
		self.trainBset = []

		self.testAset = []
		self.testBset = []

		# データ数の定義
		self.datanum = 400 if train else 106

		train_images = []
		test_images = []
		dir1 = "facades/train"
		dir2 = "facades/test"

		for root, _, fnames in sorted(os.walk(dir1)):
			for fname in fnames:
				if is_image_file(fname):
					path = os.path.join(root, fname)
					image = Image.open(path).convert('RGB')
					train_images.append(image)

		for root, _, fnames in sorted(os.walk(dir2)):
			for fname in fnames:
				if is_image_file(fname):
					path = os.path.join(root, fname)
					image = Image.open(path).convert('RGB')
					test_images.append(image)

		cnt = 0

		for image in train_images:
			# 画像を A と B に分割する
			w, h = image.size
			w2 = int(w / 2)

			A = image.crop((0, 0, w2, h))
			B = image.crop((w2, 0, w, h))

			self.trainAset.append(A)
			self.trainBset.append(B)

			if cnt == 0:
				cnt = 1
				# A.show()
				# SHOW(self.transform(A))

		for image in test_images:
			# 画像を A と B に分割する
			w, h = image.size
			w2 = int(w / 2)

			A = image.crop((0, 0, w2, h))
			B = image.crop((w2, 0, w, h))

			self.testAset.append(A)
			self.testBset.append(B)

	def __len__(self):
		return self.datanum

	def __getitem__(self, idx):

		x = random.randint(0, crop_padding)
		y = random.randint(0, crop_padding)

		A = self.trainAset[idx]
		B = self.trainBset[idx]

		A = transforms.functional.resize(A, image_size + crop_padding)
		B = transforms.functional.resize(B, image_size + crop_padding)

		if random.random() > 0.5:
			A = A.transpose(Image.FLIP_LEFT_RIGHT)
			B = B.transpose(Image.FLIP_LEFT_RIGHT)


		A = transforms.functional.crop(A, x, y, image_size, image_size)
		B = transforms.functional.crop(B, x, y, image_size, image_size)
		
		if self.train:
			return (self.transform(A), self.transform(B))
		else:
			return (self.transform(self.testAset[idx]), self.transform(self.testBset[idx]))

dataset = Mydatasets(Mytransform)

# Create the dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
 shuffle=True, num_workers=workers)

# for i in enumerate(dataloader, 0):


















