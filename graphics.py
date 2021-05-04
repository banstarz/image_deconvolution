import matplotlib.pyplot as plt
import seaborn as sns
from numpy import real
import matplotlib

sns.set()

def plot3imgs(original_img, degrad_img, restored_img, suptitle = 'Images', figsize = (16, 10)):
	fig, ax = plt.subplots(nrows=1, ncols=3, figsize=figsize)
	plt.gray()

	fig.suptitle(suptitle, fontsize=24, y=0.83)

	for a in (ax[0], ax[1], ax[2]):
	       a.axis('off')

	ax[0].imshow(real(original_img))
	ax[0].set_title('Изначальное изображение', fontsize=18)

	ax[1].imshow(real(degrad_img))
	ax[1].set_title('Повреждённое изображение', fontsize=18)

	ax[2].imshow(real(restored_img))
	ax[2].set_title('Восстановленное изображение', fontsize=18)


	fig.subplots_adjust(wspace=0.02, hspace=0.2,
	                    top=0.9, bottom=0.05, left=0, right=1)
	plt.show()


def plot_curve(x, y, title=None, xlabel=None, ylabel=None, figsize=(8,4)):
	plt.figure(figsize=figsize)
	plt.plot(x, y)
	if title:
		plt.title(title)
	if xlabel:
		plt.xlabel(xlabel)
	if ylabel:
		plt.ylabel(ylabel)
	
	plt.show()
