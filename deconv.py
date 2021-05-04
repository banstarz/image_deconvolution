import numpy as np
from numpy.fft import fft2, ifft2
from scipy.signal import convolve2d as conv2
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

class richardson_lucy:
	def __init__(self, image, psf):
		self.image = image
		self.psf = psf
		self.latent_est = np.ones(image.shape) * 0.5
		self.psf_hat = self.psf[::-1, ::-1]
		self.iterations = None

	def recover_image(self, iterations = None):
		if not iterations and not self.iterations:
			print('Не указано количество итераций')
			return

		if not iterations:
			iterations = self.iterations

		for i in range(iterations):
			est_conv = conv2(self.latent_est, self.psf, 'same')
			relative_blur = self.image / est_conv
			error_est = conv2(relative_blur, self.psf_hat, 'same')
			self.latent_est = self.latent_est * error_est
			yield np.clip(self.latent_est, 0, 1)

	def optimize(self, original, metric, step_after = 3, **kwargs):
		rich = self.recover_image(10000)
		self.step = []
		self.metric_val = []
		flag = 0
		current_metric_val = 42
		for i, est in enumerate(rich):
			if flag >= step_after:
				return

			self.step.append(i)
			current_metric_val = metric(original, est, **kwargs)

			if len(self.metric_val):
				if current_metric_val > self.metric_val[-1]:
					flag += 1
				else:
					flag = 0

			self.metric_val.append(current_metric_val)
			if current_metric_val == min(self.metric_val):
				self.best_est = est
				self.best_metric_val = current_metric_val
				self.iterations = i

	def debug_graph(self):

		plt.plot(self.step, self.metric_val)
		print(np.argmin(self.metric_val))


class wiener:
	def __init__ (self, image, psf):
		self.image = np.copy(image)
		self.psf = psf
		self.psf1 = np.zeros(self.image.shape)
		self.psf1[:self.psf.shape[0], :self.psf.shape[1]] = self.psf
		self.psf1_ft = fft2(self.psf1)
		self.image_ft = fft2(self.image)

		self.p = p
		self.est = None

	def recover_image(self, p = None):
		if not p:
			p = self.p
		latent_est_ft = self.image_ft / self.psf1_ft * np.absolute(self.psf1_ft) ** 2 / (np.absolute(self.psf1_ft) ** 2 + p)
		latent_est = ifft2(latent_est_ft)
		return np.clip(np.real(latent_est), 0, 1)

	def optimize(self, original, metric, eps = 0.0001, iterations = 20, **kwargs):
		p_l = 0
		p_r = 1000
		for i in range(iterations):
			p_c = (p_l + p_r) / 2
			err_1 = metric(original, metric, self.recover_image(p_c), **kwargs)
			err_2 = metric(original, metric, self.recover_image(p_c+eps), **kwargs)
			if err_1 > err_2:
				p_l = p_c
			else:
				p_r = p_c
			#print('(', gamma_l,',', gamma_r, ')')
		self.p = (p_l + p_r) / 2

	def debug_graph(self, original, metric, irange=(0, 50), step=0.1, **kwargs):
		x = []
		y = []
		for i in range(irange[0], irange[1]):
			x.append(i * step)
			y.append(self.metric(original, metric, self.recover_image(i * step), **kwargs))

		plt.plot(x, y)
		print(np.argmin(y)*step)

class tykhonov:
	def __init__ (self, image, psf):
		self.image = np.copy(image)
		self.psf = psf
		self.psf1 = np.zeros(self.image.shape)
		self.psf1[:self.psf.shape[0], :self.psf.shape[1]] = self.psf
		self.psf1_ft = fft2(self.psf1)
		self.image_ft = fft2(self.image)

		self.lapl = np.asarray([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
		self.lapl1 = np.zeros(self.image.shape)
		self.lapl1[:self.lapl.shape[0], :self.lapl.shape[1]] = self.lapl
		self.lapl_ft = fft2(self.lapl1)
		self.gamma = gamma

		self.est = None

	def recover_image(self, gamma = None):
		if not gamma:
			gamma = self.gamma
		latent_est_ft = self.image_ft * (np.conj(self.psf1_ft) / (np.absolute(self.psf1_ft) ** 2 + gamma * np.absolute(self.lapl_ft) ** 2))
		latent_est = ifft2(latent_est_ft)
		return np.clip(np.real(latent_est), 0, 1)

	def optimize(self, original, metric, eps = 0.0001, iterations = 20, **kwargs):
		gamma_l = 0
		gamma_r = 1000
		for i in range(iterations):
			gamma_c = (gamma_l + gamma_r) / 2
			err_1 = self.metric(original, metric, self.recover_image(gamma_c), **kwargs)
			err_2 = self.metric(original, metric, self.recover_image(gamma_c+eps), **kwargs)
			if err_1 > err_2:
				gamma_l = gamma_c
			else:
				gamma_r = gamma_c
			#print('(', gamma_l,',', gamma_r, ')')
		self.gamma = (gamma_l + gamma_r) / 2

	def debug_graph(self, original, metric, irange=(0, 50), step=0.1, **kwargs):
		x = []
		y = []
		for i in range(irange[0], irange[1]):
			x.append(i * step)
			y.append(self.metric(original, metric, self.recover_image(i * step), **kwargs))

		plt.plot(x, y)
		print(np.argmin(y)*step)

