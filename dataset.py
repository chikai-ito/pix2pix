
## conding: UTF-8
import os.path
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms

import numpy as np

import parameters

from PIL import Image


epochs = parameters.epochs
image_size = parameters.image_size
batch_size = parameters.batch_size
workers = parameters.workers
My_lambda = parameters.My_lambda


Mytransform = transforms.Compose([
   transforms.Resize(image_size),
   transforms.CenterCrop(image_size),
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

		for image in train_images:
			# 画像を A と B に分割する
			w, h = image.size
			w2 = int(w / 2)

			A = image.crop((0, 0, w2, h))
			B = image.crop((w2, 0, w, h))

			self.trainAset.append(transform(A))
			self.trainBset.append(transform(B))

		for image in test_images:
			# 画像を A と B に分割する
			w, h = image.size
			w2 = int(w / 2)

			A = image.crop((0, 0, w2, h))
			B = image.crop((w2, 0, w, h))

			self.testAset.append(transform(A))
			self.testBset.append(transform(B))

	def __len__(self):
		return self.datanum

	def __getitem__(self, idx):
		
		if self.train:
			return (self.trainAset[idx], self.trainBset[idx])
		else:
			return (self.testAset[idx], self.testBset[idx])

dataset = Mydatasets(Mytransform)

# Create the dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
 shuffle=True, num_workers=workers)

# for i in enumerate(dataloader, 0):


















