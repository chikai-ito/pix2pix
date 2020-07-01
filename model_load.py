# coding: UTF-8

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
import os.path
import torchvision.transforms as transforms
import torchvision.utils as vutils
from PIL import Image

import dataset
import parameters

import time
import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.animation as animation

epochs = parameters.epochs
image_size = parameters.image_size
batch_size = parameters.batch_size
workers = parameters.workers
My_lambda = parameters.My_lambda
data_max = parameters.data_max

# setting hyper parameters
G_losses = []
D_losses = []

print(torch.cuda.is_available())

# 画像生成のネットワーク
class NetG(nn.Module):

	def __init__(self):
		# 各層の定義
		super(NetG, self).__init__()

		# Definition of Convolution
		# in_channel, out_channel, kernel_size, stride = 1, padding = 0, 
		# dilation = 1, groups = 1, bias = True, padding_mode = 'zeros'

		self.conv1 = nn.Conv2d(3, 64, 4, 2, 1)
		self.conv2 = nn.Conv2d(64, 128, 4, 2, 1)
		self.conv3 = nn.Conv2d(128, 256, 4, 2, 1)
		self.conv4 = nn.Conv2d(256, 512, 4, 2, 1)
		self.conv5 = nn.Conv2d(512, 512, 4, 2, 1)
		self.conv6 = nn.Conv2d(512, 512, 4, 2, 1)
		self.conv7 = nn.Conv2d(512, 512, 4, 2, 1)
		self.conv8 = nn.Conv2d(512, 512, 4, 2, 1)

		# デコーダーのネットワーク
		self.upconv1 = nn.ConvTranspose2d(512, 512, 4, 2, 1)
		self.upconv2 = nn.ConvTranspose2d(512, 512, 4, 2, 1)
		self.upconv3 = nn.ConvTranspose2d(1024, 512, 4, 2, 1)
		self.upconv4 = nn.ConvTranspose2d(1024, 512, 4, 2, 1)
		self.upconv5 = nn.ConvTranspose2d(1024, 256, 4, 2, 1)
		self.upconv6 = nn.ConvTranspose2d(512, 128, 4, 2, 1)
		self.upconv7 = nn.ConvTranspose2d(256, 64, 4, 2, 1)
		self.upconv8 = nn.ConvTranspose2d(128, 3, 4, 2, 1)

		self.batchnorm_64 = nn.BatchNorm2d(64)
		self.batchnorm_128 = nn.BatchNorm2d(128)
		self.batchnorm_256 = nn.BatchNorm2d(256)
		self.batchnorm_512 = nn.BatchNorm2d(512)

		self.dropout = nn.Dropout2d(p=0.5)
		self.ReluE = nn.LeakyReLU(0.2)
		self.ReluD = nn.ReLU()
		self.tanh = nn.Tanh()

	def forward(self, x):
		# 各種データ処理用の関数

		# ここからは、Encoderフェーズ
		x1 = self.conv1(x)

		x2 = self.batchnorm_128(self.conv2(self.ReluE(x1)))

		x3 = self.batchnorm_256(self.conv3(self.ReluE(x2)))

		x4 = self.batchnorm_512(self.conv4(self.ReluE(x3)))

		x5 = self.batchnorm_512(self.conv5(self.ReluE(x4)))

		x6 = self.batchnorm_512(self.conv6(self.ReluE(x5)))

		x = self.batchnorm_512(self.conv7(self.ReluE(x6)))
		x = self.conv8(self.ReluE(x))

		# ここからは、Decoderフェーズ
		x = self.batchnorm_512(self.upconv1(self.ReluD(x)))

		x = self.dropout(self.batchnorm_512(self.upconv2(self.ReluD(x))))

		x = torch.cat((x, x6), 1)
		x = self.dropout(self.batchnorm_512(self.upconv3(self.ReluD(x))))

		x = torch.cat((x, x5), 1)
		x = self.dropout(self.batchnorm_512(self.upconv4(self.ReluD(x))))

		x = torch.cat((x, x4), 1)
		x = self.batchnorm_256(self.upconv5(self.ReluD(x)))

		x = torch.cat((x, x3), 1)
		x = self.batchnorm_128(self.upconv6(self.ReluD(x)))

		x = torch.cat((x, x2), 1)
		x = self.batchnorm_64(self.upconv7(self.ReluD(x)))

		x = torch.cat((x, x1), 1)



		x = self.tanh(self.upconv8(self.ReluD(x)))

		return x

# 画像真偽判定のネットワーク

model = NetG()
model.load_state_dict(torch.load("model/model"))

Mytransform = transforms.Compose([
   transforms.ToTensor(),
   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
   ])

directory = "facades/val"

cnt = 0

for root, _, fnames in sorted(os.walk(directory)):
	for fname in fnames:
		path = os.path.join(root, fname)
		image = Image.open(path).convert('RGB')

		w, h = image.size 
		w2 = int(w / 2)

		# A: 実際の建物の画像
		# B: 抽象画像
		A = image.crop((0, 0, w2, h))
		B = image.crop((w2, 0, w, h))

		A = Mytransform(A)
		B = Mytransform(B)

		inputs = torch.zeros(1, 3, 256, 256)
		truth = torch.zeros(1, 3, 256, 256)

		"""
		for i in range(3):
			for j in range(256):
				for k in range(256):
					inputs[0][i][j][k] = B[i][j][k]
					truth[0][i][j][k] = A[i][j][k]
		"""
		inputs[0] = B
		truth[0] = A

		outputs = model(inputs)

		print(inputs.size())
		print(outputs.size())
		print(truth.size())

		output_image = torch.cat([inputs, outputs, truth], dim=3)
		cnt = cnt + 1
		vutils.save_image(output_image, '{}/{}.png'.format("images", cnt), normalize=True)
		if cnt >= 20:
			break