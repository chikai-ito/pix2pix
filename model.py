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
import scipy.ndimage
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
class NetD(nn.Module):

	def __init__(self):
		# 各層の定義
		super(NetD, self).__init__()

		self.conv1 = nn.Conv2d(6, 64, 4, 2, 1)
		self.conv2 = nn.Conv2d(64, 128, 4, 2, 1)
		self.conv3 = nn.Conv2d(128, 256, 4, 2, 1)
		self.conv4 = nn.Conv2d(256, 512, 4, 2, 1)
		self.conv5 = nn.Conv2d(512, 1, 4, 1, 1)


		self.batchnorm_128 = nn.BatchNorm2d(128)
		self.batchnorm_256 = nn.BatchNorm2d(256)
		self.batchnorm_512 = nn.BatchNorm2d(512)

		self.Relu = nn.LeakyReLU(0.2)
		
		self.sigmoid = nn.Sigmoid()

	def forward(self, x):

		# 各種データ処理用の関数

		x = self.Relu(self.conv1(x))
		x = self.Relu(self.batchnorm_128(self.conv2(x)))
		x = self.Relu(self.batchnorm_256(self.conv3(x)))
		x = self.Relu(self.batchnorm_512(self.conv4(x)))

		x = self.conv5(x)
		x = self.sigmoid(x)

		return x

# ガウスフィルター
class GaussianLayer(nn.Module):
    def __init__(self):
        super(GaussianLayer, self).__init__()
        self.seq = nn.Sequential(
            nn.ReflectionPad2d(10), 
            nn.Conv2d(3, 3, 21, stride=1, padding=0, bias=None, groups=3)
        )

        self.weights_init()
    def forward(self, x):
        return self.seq(x)

    def weights_init(self):
        n= np.zeros((21,21))
        n[10,10] = 1
        k = scipy.ndimage.gaussian_filter(n,sigma=3)
        for name, f in self.named_parameters():
            f.data.copy_(torch.from_numpy(k))

ngpu = 1
netG = NetG()
netG = netG.to("cuda:0")
netD = NetD().to("cuda:0")
netF = GaussianLayer().to("cuda:0")
# 損失関数と最適化の設定を行う
loss_function = nn.BCEWithLogitsLoss()
loss_function_l1 = nn.L1Loss()
optimizerG = torch.optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerD = torch.optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))

# これ以降は学習

def SHOW(tensor_image):
	image_numpy = tensor_image.to("cpu").detach().numpy().copy()
	image_numpy = np.transpose(image_numpy, (2,1,0))
	maxi = np.amax(image_numpy)
	mini = np.amin(image_numpy)
	image_numpy = (image_numpy - mini) * 255.0 / (maxi - mini)
	image = Image.fromarray(image_numpy.astype(np.uint8))
	image.show()

def weights_init(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		m.weight.data.normal_(0.0, 0.02)
	elif classname.find('BatchNorm') != -1:
		m.weight.data.normal_(1.0, 0.02)
		m.bias.data.fill_(0)

Mytransform = transforms.Compose([
   # transforms.Resize(image_size + crop_padding),
   # transforms.RandomCrop(image_size),
   transforms.ToTensor(),
   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
   ])

print("Starting Training Loop ...")

netG.apply(weights_init)
netD.apply(weights_init)


start = time.time()

ite = 0
max_ite = epochs * data_max
ratio = 0.1

for net in [netF]:
	for param in net.parameters():
		param.requires_grad = False

for epoch in range(epochs):
	print("Epoch: " + str(epoch + 1))
	running_loss = 0.0
	cnt = 0
	for (i, data) in enumerate(dataset.dataloader, 0):
		if i % 10 == 10 - 1:
			print("Iterate: " + str(i + 1) + " time: " + str(time.time() - start))
		if i == data_max:
			break
		# バッチごとに、データを取り出す
		# 32 * 3 * 512 * 512
		inputs = data[1].to("cuda:0")
		truth = data[0].to("cuda:0")

		if cnt == -1:
			SHOW(inputs[0])
			# SHOW(truth[0])
			cnt = 1

		# Gの生成画像
		outputs = netG(inputs)
		# blurs = netF(truth)

		# Dの学習
		

		# 損失関数の計算と、逆伝播処理
		# 条件画像とGの生成画像を見て判断したときの誤差
		fake_judge = netD.forward(torch.cat((inputs, outputs), 1)).view(-1)
		fake_loss = loss_function(fake_judge, torch.zeros(225, dtype=torch.float).to("cuda:0"))

		# 条件画像と実際の画像を見て判断したときの誤差
		real_judge = netD.forward(torch.cat((inputs, truth), 1)).view(-1)
		real_loss = loss_function(real_judge, torch.ones(225, dtype=torch.float).to("cuda:0"))

		# blur_judge = netD.forward(torch.cat((inputs, blurs), 1)).view(-1)
		# blur_loss = loss_function(blur_judge, torch.zeros(225, dtype=torch.float).to("cuda:0"))

		total_loss_D = (fake_loss + real_loss) * 0.5

		# これ大分怪しいけど大丈夫ですか？
		# グラディエントパラメーターを初期化する
		optimizerD.zero_grad()
		total_loss_D.backward(retain_graph = True)
		optimizerD.step()

		for net in [netD]:
			for param in net.parameters():
				param.requires_grad = False

		# Gの学習

		# 損失関数の計算と、逆伝搬処理
		# Dをどのくらい騙せたか
		fake_judge_new = netD.forward(torch.cat((inputs, outputs), 1)).view(-1)
		GAN_loss = loss_function(fake_judge_new, torch.ones(225, dtype=torch.float).to("cuda:0"))
		L1_loss = loss_function_l1(outputs, truth) * My_lambda

		total_loss_G = GAN_loss + L1_loss

		# グラディエントパラメーターを初期化
		optimizerG.zero_grad()
		total_loss_G.backward()
		optimizerG.step()

		G_losses.append(total_loss_G.item())
		D_losses.append(total_loss_D.item() * My_lambda)

		ite = ite + 1

		for net in [netD]:
			for param in net.parameters():
				param.requires_grad = True

		if ratio * max_ite < ite:
			print("Completed " + str(ratio * 100) + "% ...")
			ratio = ratio + 0.1
	output_image = torch.cat([inputs, outputs, truth], dim=3)
	vutils.save_image(output_image, '{}/pix2pix_epoch_{}.png'.format("images", epoch), normalize=True)
		




plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig("figure_model.png")

torch.save(netG.state_dict(), "model/model")
Complete_NetG = netG.to("cpu")

is_image_file = dataset.is_image_file

"""
dir3 = "facades/val"
for root, _, fnames in sorted(os.walk(dir3)):
	cnt = 0
	for fname in fnames:
		if cnt >= 3:
			break
		if is_image_file(fname):
			path = os.path.join(root, fname)
			image = Image.open(path).convert('RGB')

			# 画像を A と B に分割する
			w, h = image.size
			w2 = int(w / 2)

			A = image.crop((0, 0, w2, h))
			B = image.crop((w2, 0, w, h))

			# A.show()
			# B.show()

			A = Mytransform(A)
			B = Mytransform(B)

			inputs = [B]
			inptus = torch.tensor(inputs)
			truth = [A]
			truth = torch.tensor(truth)
			outputs = netG(inputs)

			output_image = torch.cat([inputs, outputs, truth], dim=3)
			vutils.save_image(output_image, '{}/pix2pix_' + str(cnt) + '.png'.format("images_l=1", epoch), normalize=True)
			# SHOW(output_image)

			cnt = cnt + 1

"""