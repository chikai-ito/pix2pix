## conding: UTF-8
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
from PIL import Image

import dataset
import parameters

epochs = parameters.epochs
image_size = parameters.image_size
batch_size = parameters.batch_size
workers = parameters.workers
My_lambda = parameters.My_lambda

# setting hyper parameters
G_losses = []
D_losses = []

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

	def forward(self, x):
		# Encoderの時に得られた情報を格納するリスト
		skip_data = []

		# 各種データ処理用の関数
		batchnorm_64 = nn.BatchNorm2d(64)
		batchnorm_128 = nn.BatchNorm2d(128)
		batchnorm_256 = nn.BatchNorm2d(256)
		batchnorm_512 = nn.BatchNorm2d(512)
		dropout = nn.Dropout2d(p=0.5)
		ReluE = nn.LeakyReLU(0.2)
		ReluD = nn.ReLU()
		tanh = nn.Tanh()

		# ここからは、Encoderフェーズ
		x = self.conv1(x)
		skip_data.append(x)

		x = batchnorm_128(self.conv2(ReluE(x)))
		skip_data.append(x)

		x = batchnorm_256(self.conv3(ReluE(x)))
		skip_data.append(x)

		x = batchnorm_512(self.conv4(ReluE(x)))
		skip_data.append(x)

		x = batchnorm_512(self.conv5(ReluE(x)))
		skip_data.append(x)

		x = batchnorm_512(self.conv6(ReluE(x)))
		skip_data.append(x)

		x = batchnorm_512(self.conv7(ReluE(x)))
		x = self.conv8(ReluE(x))

		# ここからは、Decoderフェーズ
		x = batchnorm_512(self.upconv1(ReluD(x)))
		x = dropout(batchnorm_512(self.upconv2(ReluD(x))))

		x = torch.cat((x, skip_data[5]), 1)
		x = dropout(batchnorm_512(self.upconv3(ReluD(x))))

		x = torch.cat((x, skip_data[4]), 1)
		x = dropout(batchnorm_512(self.upconv4(ReluD(x))))

		x = torch.cat((x, skip_data[3]), 1)
		x = batchnorm_256(self.upconv5(ReluD(x)))

		x = torch.cat((x, skip_data[2]), 1)
		x = batchnorm_128(self.upconv6(ReluD(x)))

		x = torch.cat((x, skip_data[1]), 1)
		x = batchnorm_64(self.upconv7(ReluD(x)))

		x = torch.cat((x, skip_data[0]), 1)



		x = tanh(self.upconv8(ReluD(x)))

		return x

# 画像真偽判定のネットワーク

class NetD(nn.Module):

	def __init__(self):
		# 各層の定義
		super(NetD, self).__init__()

		self.conv1 = nn.Conv2d(3, 64, 4, 2, 1)
		self.conv2 = nn.Conv2d(64, 128, 4, 2, 1)
		self.conv3 = nn.Conv2d(128, 256, 4, 2, 1)
		self.conv4 = nn.Conv2d(256, 512, 4, 2, 1)
		self.conv5 = nn.Conv2d(512, 1, 1, 1, 0)

	def forward(self, x):

		# 各種データ処理用の関数
		Relu = nn.LeakyReLU(0.2)
		batchnorm_128 = nn.BatchNorm2d(128)
		batchnorm_256 = nn.BatchNorm2d(256)
		batchnorm_512 = nn.BatchNorm2d(512)
		sigmoid = nn.Sigmoid()

		x = Relu(self.conv1(x))
		x = Relu(batchnorm_128(self.conv2(x)))
		x = Relu(batchnorm_256(self.conv3(x)))
		x = Relu(batchnorm_512(self.conv4(x)))

		x = self.conv5(x)
		x = sigmoid(x)

		return x

netG = NetG()
netD = NetD()

# 損失関数と最適化の設定を行う
loss_function = nn.BCEWithLogitsLoss()
loss_function_l1 = nn.L1Loss()
optimizerG = torch.optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerD = torch.optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))

# これ以降は学習

print("Starting Training Loop ...")


for epoch in range(epochs):
	print("Epoch: " + str(epoch + 1))
	running_loss = 0.0
	for (i, data) in enumerate(dataset.dataloader, 0):
		print("Iterate: " + str(i + 1))
		# バッチごとに、データを取り出す
		# 32 * 3 * 512 * 512
		inputs = data[0]
		truth = data[1]

		# Gの生成画像
		outputs = netG(inputs)

		# Dの学習
		# グラディエントパラメーターを初期化する
		optimizerD.zero_grad()

		# 損失関数の計算と、逆伝播処理
		# 条件画像とGの生成画像を見て判断したときの誤差
		fake_judge = netD.forward(torch.cat((inputs, outputs), 0)).view(-1)
		fake_loss = loss_function(torch.zeros(batch_size * (256 // 16) * (256 // 16) * 2, dtype=torch.float), fake_judge)

		# 条件画像と実際の画像を見て判断したときの誤差
		real_judge = netD.forward(torch.cat((inputs, truth), 0)).view(-1)
		real_loss = loss_function(torch.ones(batch_size * (256 // 16) * (256 // 16) * 2, dtype=torch.float), real_judge)

		total_loss_D = (fake_loss + real_loss) * 0.5

		total_loss_D.backward(retain_graph = True)
		optimizerD.step()

		# Gの学習
		# グラディエントパラメーターを初期化
		optimizerG.zero_grad()

		# 損失関数の計算と、逆伝搬処理
		# Dをどのくらい騙せたか
		fake_judge_new = netD.forward(torch.cat((inputs, outputs), 0)).view(-1)
		GAN_loss = loss_function(torch.ones(batch_size * (256 // 16) * (256 // 16) * 2, dtype=torch.float), fake_judge_new)

		L1_loss = loss_function_l1(outputs, truth) * My_lambda

		total_loss_G = GAN_loss + L1_loss

		total_loss_G.backward()
		optimizerG.step()

		G_losses.append(total_loss_D.item())
		D_losses.append(total_loss_G.item())



plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()








