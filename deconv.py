import numpy as np
from numpy.fft import fft2, ifft2
from scipy.signal import convolve2d as conv2
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

class deconvolution:
	def __init__(self, original, corrupted, psf):
		self.original = original
		self.corrupted = corrupted
		self.psf = psf
		self.best_est = None
		self.best_metric_val = None
		self.best_parameter_val = None
		self.log = []

	def debug_graph(self):
		self.log.sort(key=lambda x: x[0])
		a, b = list(zip(*self.log))
		plt.plot(a, b, '--')
		values = min(self.log, key = lambda x: x[1])
		print('Наилучший параметр:', values[0])
		print('Наименьшая ошибка: ', values[1])

	def plot3imgs(self, suptitle='Images', figsize=(16, 10)):
		fig, ax = plt.subplots(nrows=1, ncols=3, figsize=figsize)
		plt.gray()

		fig.suptitle(suptitle, fontsize=24, y=0.83)

		for a in (ax[0], ax[1], ax[2]):
			a.axis('off')

		ax[0].imshow(self.corrupted)
		ax[0].set_title('Повреждённое изображение', fontsize=18)

		ax[1].imshow(self.best_est)
		ax[1].set_title('Восстановленное изображение', fontsize=18)

		ax[2].imshow(self.original)
		ax[2].set_title('Изначальное изображение', fontsize=18)

		fig.subplots_adjust(wspace=0.02, hspace=0.2,
							top=0.9, bottom=0.05, left=0, right=1)
		plt.show()

	def bruteforce(self, metric, irange = (0, 5), points = 25, depth = 4, **kwargs):
		self.best_metric_val = 42
		for _ in range(depth):
			metric_val_list = []
			step_size = (irange[1] - irange[0]) / points
			for i in range(points):
				parameter = step_size * i
				est = self.recover_image(parameter)
				current_metric_val = metric(self.original, est, **kwargs)
				self.log.append((parameter, current_metric_val))
				metric_val_list.append(current_metric_val)

				if current_metric_val < self.best_metric_val:
					self.best_est = est
					self.best_metric_val = current_metric_val
					self.best_parameter_val = parameter

			minima = np.argmin(metric_val_list)
			minima += 1 if minima == 0 else minima
			minima -= 1 if minima == len(metric_val_list) - 1 else minima
			irange = (metric_val_list[minima - 1], metric_val_list[minima + 1])


class richardson_lucy(deconvolution):
	def __init__(self, original, corrupted, psf):
		super().__init__(original, corrupted, psf)
		self.latent_est = np.ones(corrupted.shape) * 0.5
		self.psf_hat = self.psf[::-1, ::-1]

	def recover_image(self, iterations = None):
		if not iterations and not self.best_num_iter:
			print('Не указано количество итераций')
			return

		if not iterations:
			iterations = self.best_num_iter

		for i in range(iterations):
			est_conv = conv2(self.latent_est, self.psf, 'same')
			relative_blur = self.corrupted / est_conv
			error_est = conv2(relative_blur, self.psf_hat, 'same')
			self.latent_est = self.latent_est * error_est
			yield np.clip(self.latent_est, 0, 1)

	def fit(self, metric, steps = 1000, step_after = 3, **kwargs):
		rich_lucy = self.recover_image(steps)
		flag = 0
		current_metric_val = None
		self.best_metric_val = 42
		for i, est in enumerate(rich_lucy):
			if flag >= step_after:
				return

			current_metric_val = metric(self.original, est, **kwargs)

			if len(self.log):
				if current_metric_val > self.log[-1][1]:
					flag += 1
				else:
					flag = 0

			self.log.append((i, current_metric_val))

			if current_metric_val < self.best_metric_val:
				self.best_est = est
				self.best_metric_val = current_metric_val
				self.best_parameter_val = i


class wiener(deconvolution):
	def __init__ (self, original, corrupted, psf):
		super().__init__(original, corrupted, psf)
		self.psf1 = np.zeros(self.corrupted.shape)
		self.psf1[:self.psf.shape[0], :self.psf.shape[1]] = self.psf
		self.psf1_ft = fft2(self.psf1)
		self.corrupted_ft = fft2(self.corrupted)

	def recover_image(self, p = None):
		latent_est_ft = self.corrupted_ft / self.psf1_ft * np.absolute(self.psf1_ft) ** 2 / (np.absolute(self.psf1_ft) ** 2 + p)
		latent_est = ifft2(latent_est_ft)
		return np.clip(np.real(latent_est), 0, 1)

	def fit(self, metric, eps = 0.0001, iterations = 20, **kwargs):
		p_l = 0
		p_r = 100
		for i in range(iterations):
			p_c = (p_l + p_r) / 2
			est_1 = self.recover_image(p_c)
			est_2 = self.recover_image(p_c+eps)
			err_1 = metric(self.original, est_1, **kwargs)
			err_2 = metric(self.original, est_2, **kwargs)
			self.log.append((p_c, err_1))
			if err_1 > err_2:
				p_l = p_c
			else:
				p_r = p_c

		self.best_est = est_1
		self.best_metric_val = err_1
		self.best_parameter_val = p_c


class tykhonov(deconvolution):
	def __init__ (self, original, corrupted, psf):
		super().__init__(original, corrupted, psf)
		self.psf1 = np.zeros(self.corrupted.shape)
		self.psf1[:self.psf.shape[0], :self.psf.shape[1]] = self.psf
		self.psf1_ft = fft2(self.psf1)
		self.corrupted_ft = fft2(self.corrupted)

		self.lapl = np.asarray([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
		self.lapl1 = np.zeros(self.corrupted.shape)
		self.lapl1[:self.lapl.shape[0], :self.lapl.shape[1]] = self.lapl
		self.lapl_ft = fft2(self.lapl1)

	def recover_image(self, gamma = None):
		latent_est_ft = self.corrupted_ft * (np.conj(self.psf1_ft) / (np.absolute(self.psf1_ft) ** 2 + gamma * np.absolute(self.lapl_ft) ** 2))
		latent_est = ifft2(latent_est_ft)
		return np.clip(np.real(latent_est), 0, 1)

	def fit(self, metric, eps = 0.0001, iterations = 20, **kwargs):
		gamma_l = 0
		gamma_r = 100
		for i in range(iterations):
			gamma_c = (gamma_l + gamma_r) / 2
			est_1 = self.recover_image(gamma_c)
			est_2 = self.recover_image(gamma_c+eps)
			err_1 = self.metric(self.original, est_1, **kwargs)
			err_2 = self.metric(self.original, est_2, **kwargs)
			self.log.append((gamma_c, err_1))
			if err_1 > err_2:
				gamma_l = gamma_c
			else:
				gamma_r = gamma_c

		self.best_est = est_1
		self.best_metric_val = err_1
		self.best_parameter_val = gamma_c
