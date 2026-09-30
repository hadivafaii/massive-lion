"""Shared plotting theme for the paper's experiment panels."""
from typing import List, Literal, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def set_style(
		context: str = 'notebook',
		style: str = 'ticks',
		palette: str = None,
		font: str = 'sans-serif', ):
	sns.set_theme(
		context=context,
		style=style,
		palette=palette,
		font=font,
	)
	matplotlib.rcParams['grid.linestyle'] = ':'
	matplotlib.rcParams['figure.figsize'] = (3.0, 2.0)
	matplotlib.rcParams['image.interpolation'] = 'none'
	matplotlib.rcParams['font.family'] = font
	return


def create_figure(
		nrows: int = 1,
		ncols: int = 1,
		figsize: Tuple[float, float] = None,
		sharex: Literal['none', 'all', 'row', 'col'] = 'none',
		sharey: Literal['none', 'all', 'row', 'col'] = 'none',
		layout: str = None,
		wspace: float = None,
		hspace: float = None,
		width_ratios: List[float] = None,
		height_ratios: List[float] = None,
		reshape: bool = False,
		style: str = 'ticks',
		dpi: float = None,
		cnst: bool = True,
		**kwargs, ):
	"""
	:param nrows:
	:param ncols:
	:param figsize:
	:param layout: {'constrained', 'compressed', 'tight', None}
	:param sharex: {'none', 'all', 'row', 'col'} or bool
	:param sharey: {'none', 'all', 'row', 'col'} or bool
	:param style: {'darkgrid', 'whitegrid', 'dark', 'white', 'ticks'}
	:param wspace:
	:param hspace:
	:param width_ratios:
	:param height_ratios:
	:param reshape:
	:param dpi:
	:param cnst: shortcut to turn on layout='constrained'
	:param kwargs:
	:return: fig, axes
	"""
	set_style(style=style)
	figsize = figsize or [
		mult * default_size for mult, default_size in
		zip((ncols, nrows), plt.rcParams.get('figure.figsize'))
	]
	dpi = dpi if dpi else plt.rcParams.get('figure.dpi')
	layout = 'constrained' if cnst else layout

	fig, axes = plt.subplots(
		nrows=nrows,
		ncols=ncols,
		sharex=sharex,
		sharey=sharey,
		layout=layout,
		figsize=figsize,
		gridspec_kw={
			'wspace': wspace,
			'hspace': hspace,
			'width_ratios': width_ratios,
			'height_ratios': height_ratios},
		dpi=dpi,
		**kwargs,
	)
	if reshape:
		axes = np.array(axes).reshape(
			(nrows, ncols))
	return fig, axes
