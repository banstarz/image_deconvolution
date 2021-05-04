import numpy as np
import time
from skimage.metrics import structural_similarity as struct_sim

def MSE(orig, degrad, n = None):
	if n:
		s = orig.shape
		cx = s[0] // 2
		cy = s[1] // 2
		return ((orig[cx - n:cx + n, cy - n:cy + n] - np.real(degrad[cx - n:cx + n, cy - n:cy + n])) ** 2).mean()
	else:
		return ((orig-np.real(degrad))**2).mean()

def MAE(orig, degrad, n = None):
	if n:
		s = orig.shape
		cx = s[0] // 2
		cy = s[1] // 2
		return (np.abs(orig[cx - n:cx + n, cy - n:cy + n] - np.real(degrad[cx - n:cx + n, cy - n:cy + n]))).mean()
	else:
		return (np.abs(orig-np.real(degrad))).mean()

def time_of_work(some_func, **kwargs):
	start_time = time.clock()
	some_func(**kwargs)
	return (time.clock() - start_time)

def ssim(orig, degrad, n = None):
	if n:
		s = orig.shape
		cx = s[0] // 2
		cy = s[1] // 2
		return (1 - struct_sim(orig[cx - n:cx + n, cy - n:cy + n], degrad[cx - n:cx + n, cy - n:cy + n], data_range=degrad.max() - degrad.min())) / 2
	else:
		return (1 - struct_sim(orig, degrad, data_range = degrad.max() - degrad.min()))/2

def histogram(orig, degrad, n = None):
	if n:
		s = orig.shape
		cx = s[0] // 2
		cy = s[1] // 2
		hist1, _ = np.histogram(orig[cx - n:cx + n, cy - n:cy + n], bins=255, range=(0, 1))
		hist2, _ = np.histogram(degrad[cx - n:cx + n, cy - n:cy + n], bins=255, range=(0, 1))
	else:
		hist1, _ = np.histogram(orig, bins=255, range=(0, 1))
		hist2, _ = np.histogram(degrad, bins=255, range=(0, 1))
	return ((np.asarray(hist1) - np.asarray(hist2))**2).mean()

def Grad(orig):
	pass

