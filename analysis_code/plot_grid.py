#!/usr/bin/env python3

scan_start = 0.9
scan_end = 5.0
scan_step = 0.1
interpolation_steps = 200

recalculate = False

import sys
import os
import glob
import plotly.graph_objects as go
import numpy as np
import mout
from scipy.interpolate import griddata, CloughTocher2DInterpolator, LinearNDInterpolator, interp1d
from molparse import hijack
import molparse as mp
from ase.optimize import BFGS
from ase.optimize import MDMin, BFGS
from ase.neb import NEB, interpolate
import pickle

from clever import print_stat

import plotly.io as pio   
pio.kaleido.scope.mathjax = None

mout.hideDebug()

class Scan(object):

	def __init__(self,key,rst_key=None,wall_height=10,locut=0.9,hicut=2.0,path='.',cap=None):
		self.key = key
		self.rst_key = rst_key
		self.wall_height = wall_height
		self.hicut = hicut
		self.locut = locut
		self.path = path
		self.cap = cap

		self.missing = []

		self._grid = None
		self.neb_images = None

		self.decay_path_rc1 = None
		self.decay_path_rc2 = None
		self.decay_path_rc3 = None
		self.decay_path_rc4 = None

		# self.rc1_rc2_crossover = None
		# self.rc3_rc4_crossover = None

		self.read_energy_files()

		if rst_key is not None:
			self.read_separation_json()

	def read_separation_json(self):
		import json
		self.sep_dict = json.load(open(f"{self.rst_key}/sep.json"))
		print(f"{self.rst_key}/sep.json")
		self.bb_sep = self.sep_dict['DGN9-DCN1']
		print(self.sep_dict['DGN9-DCN1'])

	def read_energy_files(self):

		files = glob.glob(self.key)

		data = []

		files = sorted(glob.glob(f'{self.path}/{self.key}/window_??_??.energy'))

		# print(files)

		mout.varOut("Number of energy files",len(files))

		for file in files:

			basename = os.path.basename(file)

			i = int(basename.split('.')[0].split('_')[1])
			j = int(basename.split('.')[0].split('_')[2])
			
			with open(file) as f:
				for line in f:
					energy = float(line.split()[-1])

			data.append([i,j,energy*27])

		minima = min([p[2] for p in data])

		new_data = []
		for p in data:

			if self.cap and p[2]-minima > self.cap:
				mout.warningOut(f"Skipping {p[0]},{p[1]} with energy={p[2]-minima}")
				continue

			new_data.append([p[0],p[1],p[2]-minima])

		data = new_data

		self.point_list = data

	def get_grid(self):

		if self._grid is not None:
			return self._grid

		x_list = sorted(list(set([p[0] for p in self.point_list])))
		y_list = sorted(list(set([p[1] for p in self.point_list])))

		z = []

		for this_y in y_list:

			y_slice = []

			for this_x in x_list:

				matches = [p for p in self.point_list if p[1] == this_y and p[0] == this_x]

				if len(matches) == 1:
					p = matches[0]
					y_slice.append(p)

				elif len(matches) == 0:
					y_slice.append([this_x,this_y,None])
					self.missing.append([this_x,this_y])
					mout.warningOut(f"Missing data_point {this_x=} {this_y=}")
				else:
					mout.errorOut("multiple matches",fatal=True)

			z.append([p[2] for p in sorted(y_slice,key=lambda p: p[0])])
		
		for p in self.point_list:
			p[0] = self.convert_to_rc([p[0]])[0]
			p[1] = self.convert_to_rc([p[1]])[0]

		x_list = self.convert_to_rc(x_list)
		y_list = self.convert_to_rc(y_list)

		self._grid = x_list,y_list,z

		return x_list,y_list,z

	def plot_heatmap(self,fig=None,interpolated=False,show=True,trim=False,legend=False):

		if fig is None:
			fig = go.Figure()

		if not interpolated:
			x,y,z = self.get_grid()

			if trim:
				x,y,z = trim_grid((x,y,z),hicut=self.hicut)

			trace = go.Heatmap(name=self.key,x=x,y=y,z=z,colorbar=dict(orientation='h'))
		else:
			x,y,z = self.get_interpolated_grid()

			if trim:
				x,y,z = trim_grid((x,y,z),hicut=self.hicut)

			if self.neb_images:
				interpolator = self.get_interpolator()

				neb_energies = self.get_neb_energies()

				min_z = float(min(neb_energies))
				max_z = float(max(neb_energies))
				rng = max_z - min_z

				contours=dict(start=min_z,end=max_z + 0.25*rng,size=rng/15)

				trace = go.Contour(name=self.key,x=x,y=y,z=z,contours=contours,showscale=False)

				fig.add_trace(trace)

				x,y = self.get_neb_rcpath()

				trace = go.Scatter(name=self.key,x=x,y=y,marker_color='white',line_color='white',mode='markers+lines')

			else:
				trace = go.Heatmap(name=self.key,x=x,y=y,z=z)

			if self.decay_path_rc1:

				fig.add_trace(trace)

				distances = []
				times = []
				x,y = [],[]
				for i,(a,b) in enumerate(zip(self.decay_path_rc1,self.decay_path_rc3)):
					d = np.linalg.norm(np.array([a,b])-np.array(self.can.rcs))
					distances.append(d)
					x.append(a)
					times.append(i)
					y.append(b)
					if d < 0.02:
						break

				color = [f'rgb(0.0,{i/len(times):.2f},{1.0-i/len(times):.2f})' for i,_ in enumerate(times)]

				trace = go.Scatter(name='QM/MM MD Decay',x=x,y=y,marker=dict(color=color),line=dict(color='black'),mode='markers+lines')
				# trace = go.Scatter(name='QM/MM MD Decay',x=x,y=y,marker_color='green',line_color='green',mode='markers+lines')

		# plot the original surface
		fig.add_trace(trace)

		if self.missing:

			x = []
			y = []
			for p in self.missing:
				rc1 = self.convert_to_rc([p[0]])[0]
				rc3 = self.convert_to_rc([p[1]])[0]

				if rc1 < self.locut:
					continue
				if rc3 < self.locut:
					continue
				if rc1 > self.hicut:
					continue
				if rc3 > self.hicut:
					continue

				x.append(rc1)
				y.append(rc3)

			trace = go.Scatter(name='missing',x=x,y=y,mode='markers',marker_color='red',marker_symbol='cross')
			fig.add_trace(trace)

		fig.update_layout(xaxis_title='RC1')
		fig.update_layout(yaxis_title='RC2')

		if legend: 
			fig.update_layout(title=f'{self.key}')

		fig.update_layout(
			width=600,
			height=600,
			# autosize=True,
			xaxis_constrain="domain",
			yaxis_constrain="domain",
			yaxis_scaleanchor="x",
			yaxis_scaleratio=1,
			margin=dict(l=5,r=5,b=5,t=5),
			showlegend=legend,
			# margin=dict(pad=0),
			# xaxis_automargin=True,
			# yaxis_automargin=True,
		)

		if show:
			fig.show()

		return fig

	def extend_walls(self,mid_walls):
		mout.out(f'placing walls {self.wall_height}, {self.locut - 0.2}:{self.hicut + 0.2}')
		self.point_list.append([self.locut - 0.2,self.locut - 0.2,self.wall_height])
		self.point_list.append([self.locut - 0.2,self.hicut + 0.2,self.wall_height])
		self.point_list.append([self.hicut + 0.2,self.hicut + 0.2,self.wall_height])
		self.point_list.append([self.hicut + 0.2,self.locut - 0.2,self.wall_height])
		
		if mid_walls:
			self.point_list.append([self.locut - 0.2,(self.locut + self.hicut)/2,self.wall_height])
			self.point_list.append([self.hicut + 0.2,(self.locut + self.hicut)/2,self.wall_height])
			self.point_list.append([(self.locut + self.hicut)/2,self.locut - 0.2,self.wall_height])
			self.point_list.append([(self.locut + self.hicut)/2,self.hicut + 0.2,self.wall_height])

	def get_interpolated_grid(self):

		grid_x, grid_y = np.mgrid[scan_start:scan_end:interpolation_steps*1j, scan_start:scan_end:interpolation_steps*1j]

		points = [[p[0], p[1]] for p in self.point_list]
		values = [p[2] for p in self.point_list]
		
		grid_z0 = griddata(points, values, (grid_x,grid_y), method='cubic')

		x_list = sorted(list(set(grid_x.ravel())))
		y_list = sorted(list(set(grid_y.ravel())))

		return x_list, y_list, grid_z0.T

	def get_interpolator(self):
		self.interpolator = CloughTocher2DInterpolator([[p[0], p[1]] for p in self.point_list],[p[2] for p in self.point_list])
		return self.interpolator

	def shift_grid(self,energy):
		for p in self.point_list:
			p[2] -= energy

	def convert_to_rc(self,data):
		lookup = np.arange(scan_start,scan_end,scan_step)
		data = [round(lookup[x],1) for x in data]
		return data

	def rcs_to_windows(self,rcs):

		lookup = np.arange(scan_start,scan_end,scan_step)

		i = mp.signal.closest_index(rcs[0],lookup,numpy=True)
		j = mp.signal.closest_index(rcs[1],lookup,numpy=True)

		return i,j

	def get_calculator(self):
		return hijack.FakeSurfaceCalculator(pmf=self.get_interpolator())

	def optimize(self,atoms):

		atoms.set_calculator(self.get_calculator())

		dyn = BFGS(atoms=atoms,logfile=f'{self.key}_{atoms.name}_opt.log')

		mout.headerOut(f"Running optimiser ({atoms.name})...")
		dyn.run()

	def neb(self,n_images,k,reactant,product,fmax=0.04):

		self.neb_images = [reactant.copy() for i in range(n_images//2)] + [product.copy() for i in range(n_images//2)]

		interpolate(self.neb_images)

		neb = NEB(self.neb_images,k=k)

		for image in self.neb_images[1:-1]:
			# image.set_constraints(locut=0.91,hicut=2.28)
			image.set_calculator(self.get_calculator())

		optimizer = BFGS(neb, trajectory=f'{self.key}_neb.traj', logfile=f'{self.key}_neb.log')

		mout.headerOut(f"Running NEB... logfile={self.key}_neb.log")

		try:
			optimizer.run(fmax=fmax,steps=2000)
		except hijack.NaNEncounteredInPmfError:
			mout.errorOut("NEB did not finish correctly!")

		import scipy.signal as sps
		self.neb_minima_indices = [0] + list(sps.argrelextrema(np.array(self.get_neb_energies()), np.less)[0]) + [-1]
		self.neb_maxima_indices = list(sps.argrelextrema(np.array(self.get_neb_energies()), np.greater)[0])

		self.neb_stationary_x = [self.get_neb_rcdist()[i] for i in self.neb_minima_indices + self.neb_maxima_indices]
		self.neb_stationary_y = [self.get_neb_energies()[i] for i in self.neb_minima_indices + self.neb_maxima_indices]

		self.neb_stationary_points = [[x,y] for x,y in zip(self.get_neb_rcdist(),self.get_neb_energies())]

	def get_neb_energies(self):
		interpolator = self.get_interpolator()
		return [interpolator(x,y) for x,y in zip(*self.get_neb_rcpath())]

	def get_neb_rcdist(self,normalise=None,walls=None):

		# renormalise to distance between canonical and tautomer

		path = [[x,y] for x,y in zip(*self.get_neb_rcpath(walls=walls))]

		distances = [0.0]
		for i,p in enumerate(path[1:]):
			d = np.array(p) - np.array(path[i])
			distances.append(distances[-1]+np.linalg.norm(d))

		if normalise is not None:
			distances = list(normalise*np.array(distances)/distances[-1])

		return distances

	def get_neb_candist(self):

		# renormalise to distance between canonical and tautomer

		path = [[x,y] for x,y in zip(*self.get_neb_rcpath())]

		distances = []
		for p in path:

			d = np.array(p) - np.array(self.can.rcs)

			distances.append(np.linalg.norm(d))

		return distances

	def get_neb_rcpath(self,walls=None):

		if walls is not None:

			images = self.neb_images

			# canonical

			can_wall_images = []

			vec = np.array(images[0].rcs) - np.array(images[1].rcs)

			for i in range(walls[0]):
				rcs = list(np.array(images[0].rcs) + (i+1)*vec)
				can_wall_images.append(rcs)

			# tautomer

			tau_wall_images = []

			vec = np.array(images[-1].rcs) - np.array(images[-2].rcs)

			for i in range(walls[1]):
				rcs = list(np.array(images[-1].rcs) + (i+1)*vec)
				tau_wall_images.append(rcs)

			# print(reversed(can_wall_images),[i.rcs for i in self.neb_images],tau_wall_images)

			combined = list(reversed(can_wall_images)) + [i.rcs for i in self.neb_images] + tau_wall_images

			return [p[0] for p in combined], [p[1] for p in combined]

		else:
			return [i.rcs[0] for i in self.neb_images], [i.rcs[1] for i in self.neb_images]

	def load_decay_path(self,file):
		xvg = mp.parseXVG(file,yscale=10.0)
		self.decay_path_rc1 = xvg.columns['1']
		self.decay_path_rc2 = xvg.columns['2']
		self.decay_path_rc3 = xvg.columns['3']
		self.decay_path_rc4 = xvg.columns['4']

	def load_decay_qm_energies(self,file):
		self.decay_energies = mp.signal.parseDat(file,num_columns=1,header_rows=0)
		self.decay_energies = [e*27 for e in self.decay_energies]

	def get_normalised_neb_interpolator(self):

		raw_neb_distance = self.get_neb_rcdist()
		raw_neb_energies = self.get_neb_energies()	

		if len(self.neb_maxima_indices) == 1:

			# split the data around ts2
			pre_ts2_distance = np.array(raw_neb_distance[:self.neb_maxima_indices[-1]+1])
			post_ts2_distance = np.array(raw_neb_distance[self.neb_maxima_indices[-1]:])

			# rescale
			pre_ts2_distance = 1.5*pre_ts2_distance/pre_ts2_distance[-1]
			post_ts2_distance = 0.5*(post_ts2_distance-post_ts2_distance[0])/(post_ts2_distance-post_ts2_distance[0])[-1]+1.5

			x_data = list(pre_ts2_distance)[:-1] + list(post_ts2_distance)[0:]

		else:

			# split the data around stationary points
			pre_ts1_distance = np.array(raw_neb_distance[:self.neb_maxima_indices[0]+1])
			pre_zwit_distance = np.array(raw_neb_distance[self.neb_maxima_indices[0]:self.neb_minima_indices[1]+1])
			post_zwit_distance = np.array(raw_neb_distance[self.neb_minima_indices[1]:self.neb_maxima_indices[-1]+1])
			post_ts2_distance = np.array(raw_neb_distance[self.neb_maxima_indices[-1]:])

			# rescale
			pre_ts1_distance = 0.5*pre_ts1_distance/pre_ts1_distance[-1]
			pre_zwit_distance = 0.5*(pre_zwit_distance-pre_zwit_distance[0])/(pre_zwit_distance-pre_zwit_distance[0])[-1]+0.5
			post_zwit_distance = 0.5*(post_zwit_distance-post_zwit_distance[0])/(post_zwit_distance-post_zwit_distance[0])[-1]+1.0
			post_ts2_distance = 0.5*(post_ts2_distance-post_ts2_distance[0])/(post_ts2_distance-post_ts2_distance[0])[-1]+1.5

			x_data = list(pre_ts1_distance)[:-1] + list(pre_zwit_distance)[0:-1] + list(post_zwit_distance)[0:-1] + list(post_ts2_distance)[0:]
		
		y_data = raw_neb_energies

		return interp1d(x_data, y_data, kind='quadratic', assume_sorted=True)

	def get_gmx_like_neb_path(self,walls=None):

		gmx_rc = []

		x,y = self.get_neb_rcpath(walls=walls)

		for rc1,rc2 in zip(x,y):
			gmx_rc.append((rc1+rc2)/2)

		return gmx_rc

def get_string_trace(path,interpolator,steps=None,name=''):

	xdata = []
	ydata = []
	zdata = []

	if steps:

		new_path = []

		for i,(x,y) in enumerate(path):

			if i == 0:
				continue

			for j in range(steps):

				this_x = (path[i][0] - path[i-1][0]) * j/steps + path[i-1][0]
				this_y = (path[i][1] - path[i-1][1]) * j/steps + path[i-1][1]

				new_path.append([this_x,this_y])

		path = new_path

	for x,y in path:

		xdata.append(x)
		ydata.append(y)

		e = interpolator(x,y)

		zdata.append(e)

	return path, zdata, go.Scatter3d(name=name,x=xdata,y=ydata,z=zdata,line_color='black')

def profile_trace(path,energies,name=''):

	distances = [0.0]
	for i,p in enumerate(path[1:]):
		d = np.array(p) - np.array(path[i-1])
		distances.append(distances[-1]+np.linalg.norm(d))

	trace = go.Scatter(name=name,x=distances,y=energies)

	return trace

def trim_grid(grid,locut=0.9,hicut=2.0,wall_height=1000):

	x_list = grid[0]
	y_list = grid[1]

	grid_z0 = grid[2].T.tolist()

	new_grid = []

	for y_val,y_slice in zip(y_list,grid_z0):

		if y_val < locut or y_val > hicut:
			continue

		new_grid.append([y for x,y in zip(x_list,y_slice) if x >= locut and x <= hicut and y < wall_height])

	x_list = [x for x in x_list if x >= locut or x <= hicut]
	y_list = [y for y in y_list if y >= locut or y <= hicut]

	return x_list,y_list,np.array(new_grid).T

def main():

	dna_scans = [
					dict(key='B3LYP_2DM_mid_L01',rst_key='2DM_grid_mid_L01'), # good
					dict(key='B3LYP_2DM_mid_L02',rst_key='2DM_grid_mid_L02'), # good (slow)
					dict(key='B3LYP_2DM_mid_L03',rst_key='2DM_grid_mid_L03'), # good
					# dict(key='B3LYP_2DM_mid_L04',rst_key='2DM_grid_mid_L04'), # focked (many missing points)
					dict(key='B3LYP_2DM_mid_L05',rst_key='2DM_grid_mid_L05'), # good
					# dict(key='B3LYP_2DM_L01_c12',rst_key='2DM_grid_L03'), # no stable tautomer
					dict(key='BLYP_2DM_L03',rst_key='2DM_grid_L03'), # alternative zwitterion is not stable
					dict(key='B3LYP_2DM_L03_c4',rst_key='2DM_grid_L03'), # alternative zwitterion is not stable
					# dict(key='B3LYP_2DM_L04',rst_key='2DM_grid_L04'), # incomplete/junk
					dict(key='B3LYP_2DM_L05',rst_key='2DM_grid_L05'),
					dict(key='B3LYP_2DM_L11',rst_key='2DM_grid_L11'), # alternative zwitterion is not stable
	 			]

	pcra_scans = [
					dict(key='B3LYP_2CM_HL05',rst_key='2CM_grid_HL05',neb_fmax=0.0555), # good
					dict(key='B3LYP_2CM_HL06',rst_key='2CM_grid_HL06'), # really slow & alternate path
					dict(key='B3LYP_2CM_grid_new',rst_key='2CM_rst_grid'),
					dict(key='B3LYP_2CM_HL01',rst_key='2CM_grid_HL01'),
					dict(key='B3LYP_2CM_HL02',rst_key='2CM_grid_HL02',wall_height=20),
					dict(key='B3LYP_2CM_HL03',rst_key='2CM_grid_HL03',neb_fmax=0.055), # zwitterion unstable
					dict(key='B3LYP_2CM_HL04',rst_key='2CM_grid_HL04'), # alternate path
	 			]

	N624A_scans = [
					dict(key='B3LYP_2CM_mid_HL_N624A12',rst_key='2CM_grid_HL_N624A12'),
					dict(key='B3LYP_2CM_mid_HL_N624A13',rst_key='2CM_grid_HL_N624A13'),
					dict(key='B3LYP_2CM_mid_HL_N624A14',rst_key='2CM_grid_HL_N624A14'),
					dict(key='B3LYP_2CM_mid_HL_N624A11',rst_key='2CM_grid_HL_N624A11'),
					]


	if recalculate:

		# DNA
		hicut = 2.1

		for scan_d in dna_scans:
			can = hijack.FakeAtoms(rcs=[1.0,1.0],name="can")
			zwi = hijack.FakeAtoms(rcs=[1.0,1.8],name="zwi")
			alt = hijack.FakeAtoms(rcs=[1.8,1.0],name="alt")
			tau = hijack.FakeAtoms(rcs=[1.8,1.8],name="tau")
			wall_height = scan_d['wall_height'] if 'wall_height' in scan_d else 10
			neb_fmax = scan_d['neb_fmax'] if 'neb_fmax' in scan_d else 0.04
			scan = scan_routine(scan_d['key'], scan_d['rst_key'], can, zwi, alt, tau, hicut=hicut,wall_height=wall_height,neb_fmax=neb_fmax)
			pickle.dump(scan,open(f"{scan.key}.pickle",'wb'))

		# # PcrA
		# hicut = 2.4

		# for scan_d in pcra_scans:
		# 	can = hijack.FakeAtoms(rcs=[1.03,1.02],name="can")
		# 	zwi = hijack.FakeAtoms(rcs=[1.03,2.0],name="zwi")
		# 	alt = hijack.FakeAtoms(rcs=[2.3,1.06],name="alt")
		# 	tau = hijack.FakeAtoms(rcs=[2.3,2.07],name="tau")
		# 	wall_height = scan_d['wall_height'] if 'wall_height' in scan_d else 10
		# 	neb_fmax = scan_d['neb_fmax'] if 'neb_fmax' in scan_d else 0.04
		# 	scan = scan_routine(scan_d['key'], scan_d['rst_key'], can, zwi, alt, tau, hicut=hicut,wall_height=wall_height,neb_fmax=neb_fmax)
		# 	pickle.dump(scan,open(f"{scan.key}.pickle",'wb'))

		# # N624A
		# hicut = 2.4

		# for scan_d in N624A_scans:
		# 	can = hijack.FakeAtoms(rcs=[1.03,1.02],name="can")
		# 	zwi = hijack.FakeAtoms(rcs=[1.03,2.0],name="zwi")
		# 	alt = hijack.FakeAtoms(rcs=[2.3,1.06],name="alt")
		# 	tau = hijack.FakeAtoms(rcs=[2.3,2.07],name="tau")
		# 	wall_height = scan_d['wall_height'] if 'wall_height' in scan_d else 10
		# 	neb_fmax = scan_d['neb_fmax'] if 'neb_fmax' in scan_d else 0.04
		# 	scan = scan_routine(scan_d['key'], scan_d['rst_key'], can, zwi, alt, tau, hicut=hicut,wall_height=wall_height,neb_fmax=neb_fmax)
		# 	pickle.dump(scan,open(f"{scan.key}.pickle",'wb'))

	else:


		B3LYP_2DM_mid_L01 = pickle.load(open('B3LYP_2DM_mid_L01.pickle','rb'))
		B3LYP_2DM_mid_L02 = pickle.load(open('B3LYP_2DM_mid_L02.pickle','rb'))
		B3LYP_2DM_mid_L03 = pickle.load(open('B3LYP_2DM_mid_L03.pickle','rb'))
		B3LYP_2DM_mid_L05 = pickle.load(open('B3LYP_2DM_mid_L05.pickle','rb'))

		BLYP_2DM_L03 = pickle.load(open('BLYP_2DM_L03.pickle','rb'))
		B3LYP_2DM_L03_c4 = pickle.load(open('B3LYP_2DM_L03_c4.pickle','rb'))
		B3LYP_2DM_L05 = pickle.load(open('B3LYP_2DM_L05.pickle','rb'))
		B3LYP_2DM_L11 = pickle.load(open('B3LYP_2DM_L11.pickle','rb'))

		B3LYP_2CM_grid_new = pickle.load(open('B3LYP_2CM_grid_new.pickle','rb'))
		B3LYP_2CM_HL01 = pickle.load(open('B3LYP_2CM_HL01.pickle','rb'))
		B3LYP_2CM_HL02 = pickle.load(open('B3LYP_2CM_HL02.pickle','rb'))
		B3LYP_2CM_HL03 = pickle.load(open('B3LYP_2CM_HL03.pickle','rb'))
		B3LYP_2CM_HL04 = pickle.load(open('B3LYP_2CM_HL04.pickle','rb'))
		B3LYP_2CM_HL05 = pickle.load(open('B3LYP_2CM_HL05.pickle','rb'))
		B3LYP_2CM_HL06 = pickle.load(open('B3LYP_2CM_HL06.pickle','rb'))
		
		B3LYP_2CM_mid_HL_N624A11 = pickle.load(open('B3LYP_2CM_mid_HL_N624A11.pickle','rb'))
		B3LYP_2CM_mid_HL_N624A12 = pickle.load(open('B3LYP_2CM_mid_HL_N624A12.pickle','rb'))
		B3LYP_2CM_mid_HL_N624A13 = pickle.load(open('B3LYP_2CM_mid_HL_N624A13.pickle','rb'))
		B3LYP_2CM_mid_HL_N624A14 = pickle.load(open('B3LYP_2CM_mid_HL_N624A14.pickle','rb'))

		# B3LYP_2DM_L03_c4.rc1_rc2_decay_timerange = [0,131]
		# B3LYP_2DM_L05.rc1_rc2_decay_timerange = [0,139]
		# L11 complicated
		# HL01 complicated

		# for scan in [
		# 	B3LYP_2DM_mid_L01,
		# 	B3LYP_2DM_mid_L02,
		# 	B3LYP_2DM_mid_L03,
		# 	B3LYP_2DM_mid_L05,
		# 	B3LYP_2DM_L03_c4,
		# 	B3LYP_2DM_L05,
		# 	B3LYP_2DM_L11,
		# 	BLYP_2DM_L03,
		# 	B3LYP_2CM_HL01,
		# 	B3LYP_2CM_HL02,
		# 	B3LYP_2CM_HL03,
		# 	B3LYP_2CM_HL04,
		# 	B3LYP_2CM_HL05,
		# 	B3LYP_2CM_HL06,
		# 	# B3LYP_2CM_grid_new, # no decay
		# 	B3LYP_2CM_mid_HL_N624A11,
		# 	B3LYP_2CM_mid_HL_N624A12,
		# 	B3LYP_2CM_mid_HL_N624A13,
		# 	B3LYP_2CM_mid_HL_N624A14,
		# 	]:
			
		# 	decay_pullx = glob.glob(f'../TautomerStability/{scan.key}_vel_decay/pullx.xvg')
		# 	for file in decay_pullx:
		# 		scan.load_decay_path(file)
			
		# 	decay_pullx = glob.glob(f'../TautomerStability/{scan.key}_vel_decay/qm_energies.dat')
		# 	for file in decay_pullx:
		# 		scan.load_decay_qm_energies(file)

		# 	fig = scan.plot_heatmap(interpolated=True,show=False,trim=True)
		# 	mp.write(f"heatmaps/{scan.key}.pdf",fig)
		# 	mout.varOut("bb_sep",scan.bb_sep)

		# 	save_neb_profile(scan)

		# 	if decay_pullx:
		# 		# plot_neb_vs_decay(scan)
		# 		plot_decay_rcs(scan,show=False)

		# exit()

		pickle.dump(B3LYP_2DM_mid_L01,open('B3LYP_2DM_mid_L01.pickle','wb'))
		pickle.dump(B3LYP_2DM_mid_L02,open('B3LYP_2DM_mid_L02.pickle','wb'))
		pickle.dump(B3LYP_2DM_mid_L03,open('B3LYP_2DM_mid_L03.pickle','wb'))
		pickle.dump(B3LYP_2DM_mid_L05,open('B3LYP_2DM_mid_L05.pickle','wb'))

		pickle.dump(B3LYP_2CM_mid_HL_N624A11,open('B3LYP_2CM_mid_HL_N624A11.pickle','wb'))
		pickle.dump(B3LYP_2CM_mid_HL_N624A12,open('B3LYP_2CM_mid_HL_N624A12.pickle','wb'))
		pickle.dump(B3LYP_2CM_mid_HL_N624A13,open('B3LYP_2CM_mid_HL_N624A13.pickle','wb'))
		pickle.dump(B3LYP_2CM_mid_HL_N624A14,open('B3LYP_2CM_mid_HL_N624A14.pickle','wb'))

		pickle.dump(BLYP_2DM_L03,open(f'{BLYP_2DM_L03.key}.pickle','wb'))
		pickle.dump(B3LYP_2DM_L03_c4,open(f'{B3LYP_2DM_L03_c4.key}.pickle','wb'))
		pickle.dump(B3LYP_2DM_L05,open(f'{B3LYP_2DM_L05.key}.pickle','wb'))
		pickle.dump(B3LYP_2DM_L11,open(f'{B3LYP_2DM_L11.key}.pickle','wb'))
		pickle.dump(B3LYP_2CM_grid_new,open(f'{B3LYP_2CM_grid_new.key}.pickle','wb'))
		pickle.dump(B3LYP_2CM_HL01,open(f'{B3LYP_2CM_HL01.key}.pickle','wb'))
		pickle.dump(B3LYP_2CM_HL02,open(f'{B3LYP_2CM_HL02.key}.pickle','wb'))
		pickle.dump(B3LYP_2CM_HL03,open(f'{B3LYP_2CM_HL03.key}.pickle','wb'))
		pickle.dump(B3LYP_2CM_HL04,open(f'{B3LYP_2CM_HL04.key}.pickle','wb'))
		pickle.dump(B3LYP_2CM_HL05,open('B3LYP_2CM_HL05.pickle','wb'))
		pickle.dump(B3LYP_2CM_HL06,open('B3LYP_2CM_HL06.pickle','wb'))

		# for scan in [B3LYP_2DM_mid_L01,B3LYP_2DM_mid_L02,B3LYP_2DM_mid_L03,B3LYP_2DM_mid_L05,B3LYP_2DM_L03_c4,B3LYP_2DM_L05,B3LYP_2DM_L11,B3LYP_2CM_HL01,B3LYP_2CM_HL02,B3LYP_2CM_HL03,B3LYP_2CM_HL04]:
		for scan in [
			B3LYP_2DM_mid_L01,
			# B3LYP_2DM_mid_L02,
			# B3LYP_2DM_mid_L03,
			# B3LYP_2DM_mid_L05,
			# B3LYP_2DM_L03_c4,
			# B3LYP_2DM_L05,
			# B3LYP_2DM_L11,
			# BLYP_2DM_L03,
			# B3LYP_2CM_HL01,
			# B3LYP_2CM_HL02,
			# B3LYP_2CM_HL03,
			# B3LYP_2CM_HL04,
			# B3LYP_2CM_HL05,
			# B3LYP_2CM_HL06,
			# B3LYP_2CM_mid_HL_N624A11,
			# B3LYP_2CM_mid_HL_N624A12,
			# B3LYP_2CM_mid_HL_N624A13,
			# B3LYP_2CM_mid_HL_N624A14
		]:
			# create_decay_input_gro(scan, *scan.tau.rcs,show=False)
			# create_ts_inputs(scan)
			wigner_correction(scan)
			# exit()

		exit()

		barrier_statistics('All DNA',[B3LYP_2DM_mid_L01,B3LYP_2DM_mid_L02,B3LYP_2DM_mid_L03,B3LYP_2DM_mid_L05,B3LYP_2DM_L03_c4,B3LYP_2DM_L05,B3LYP_2DM_L11])
		barrier_statistics('All Helicase Normal',[B3LYP_2CM_grid_new,B3LYP_2CM_HL01,B3LYP_2CM_HL02,B3LYP_2CM_HL03,B3LYP_2CM_HL05])
		barrier_statistics('All Helicase Alternate',[B3LYP_2CM_HL04,B3LYP_2CM_HL06])
		
		barrier_statistics('All N624A',[B3LYP_2CM_mid_HL_N624A11,B3LYP_2CM_mid_HL_N624A12,B3LYP_2CM_mid_HL_N624A14])
		barrier_statistics('All N624A Alternate',[B3LYP_2CM_mid_HL_N624A13])

		lifetime_summary('All DNA',[B3LYP_2DM_mid_L01,B3LYP_2DM_mid_L02,B3LYP_2DM_mid_L03,B3LYP_2DM_mid_L05,B3LYP_2DM_L03_c4,B3LYP_2DM_L05,B3LYP_2DM_L11])
		lifetime_summary('All Helicase',[B3LYP_2CM_grid_new,B3LYP_2CM_HL01,B3LYP_2CM_HL02,B3LYP_2CM_HL03,B3LYP_2CM_HL05,B3LYP_2CM_HL04,B3LYP_2CM_HL06])

		plot_lifetime_statistics([
			B3LYP_2DM_mid_L01,
			B3LYP_2DM_mid_L02,
			B3LYP_2DM_mid_L03,
			B3LYP_2DM_mid_L05,
			B3LYP_2DM_L03_c4,
			# B3LYP_2CM_mid_HL_N624A11,
			B3LYP_2DM_L05,
			B3LYP_2DM_L11,
			B3LYP_2CM_HL01,
			B3LYP_2CM_HL02,
			B3LYP_2CM_HL03,
			B3LYP_2CM_HL04,
			B3LYP_2CM_HL05,
			B3LYP_2CM_HL06
		],debug=True)

		# exit()

		# compare BLYP vs B3LYP (L03)
		# plot_compare_profile([BLYP_2DM_L03,B3LYP_2DM_L03_c4])
		# plot_compare_path([BLYP_2DM_L03,B3LYP_2DM_L03_c4])

		# # compare all DNA's
		plot_compare_profile([B3LYP_2DM_mid_L01,B3LYP_2DM_mid_L02,B3LYP_2DM_mid_L03,B3LYP_2DM_mid_L05,B3LYP_2DM_L03_c4,B3LYP_2DM_L05,B3LYP_2DM_L11],align=False,file='all_dna_profiles.pdf')
		plot_compare_profile([B3LYP_2DM_mid_L01,B3LYP_2DM_mid_L02,B3LYP_2DM_mid_L03,B3LYP_2DM_mid_L05,B3LYP_2DM_L03_c4,B3LYP_2DM_L05,B3LYP_2DM_L11],statistics=True,xvg_compare='../ArcherUmbrellaDNA/wham_louie_b2_3Oct22/bsProfs.xvg',file='all_dna_statprofs.pdf')

		# # compare all N624A's
		plot_compare_profile([B3LYP_2CM_mid_HL_N624A11,B3LYP_2CM_mid_HL_N624A12,B3LYP_2CM_mid_HL_N624A13,B3LYP_2CM_mid_HL_N624A14],align=False,file='all_N624A_profiles.pdf')
		plot_compare_profile([B3LYP_2CM_mid_HL_N624A11,B3LYP_2CM_mid_HL_N624A12,B3LYP_2CM_mid_HL_N624A13,B3LYP_2CM_mid_HL_N624A14],statistics=True,xvg_compare='../ArcherUmbrellaHelicase/wham_louie_N624A_b0_23Mar23_cut/bsProfs.xvg')

		# plot_compare_path([B3LYP_2DM_mid_L01,B3LYP_2DM_mid_L02,B3LYP_2DM_mid_L03,B3LYP_2DM_mid_L05,B3LYP_2DM_L03_c4,B3LYP_2DM_L05,B3LYP_2DM_L11])
		
		# # compare all PcrA's
		# plot_comparse_sep_properties([B3LYP_2CM_grid_new,B3LYP_2CM_HL01,B3LYP_2CM_HL02,B3LYP_2CM_HL03])
		# plot_compare_3dsep([B3LYP_2CM_grid_new,B3LYP_2CM_HL01,B3LYP_2CM_HL02,B3LYP_2CM_HL03,B3LYP_2CM_HL04])
		
		# plot_compare_profile([B3LYP_2CM_grid_new,B3LYP_2CM_HL01,B3LYP_2CM_HL02,B3LYP_2CM_HL03,B3LYP_2CM_HL04,B3LYP_2CM_mid_HL_N624A11])
		plot_compare_profile([B3LYP_2CM_grid_new,B3LYP_2CM_HL01,B3LYP_2CM_HL02,B3LYP_2CM_HL03,B3LYP_2CM_HL04,B3LYP_2CM_HL05,B3LYP_2CM_HL06],statistics=False,file='all_pcra_profile.pdf')
		plot_compare_profile([B3LYP_2CM_grid_new,B3LYP_2CM_HL01,B3LYP_2CM_HL02,B3LYP_2CM_HL03,B3LYP_2CM_HL04,B3LYP_2CM_HL05,B3LYP_2CM_HL06],statistics=True,xvg_compare='../ArcherUmbrellaHelicase/wham_louie_b0_3Oct22/bsProfs.xvg',ts1_map=1.83)
		# plot_compare_path([B3LYP_2CM_grid_new,B3LYP_2CM_HL01,B3LYP_2CM_HL02,B3LYP_2CM_HL03,B3LYP_2CM_HL04])

		# compare everything
		# plot_compare_profile([B3LYP_2DM_L03_c4,B3LYP_2DM_L05,B3LYP_2CM_grid_new,B3LYP_2CM_HL01,B3LYP_2CM_HL02,B3LYP_2CM_HL03,B3LYP_2CM_HL04])
		# plot_compare_path([B3LYP_2DM_L03_c4,B3LYP_2DM_L05,B3LYP_2CM_grid_new,B3LYP_2CM_HL01,B3LYP_2CM_HL02,B3LYP_2CM_HL03,B3LYP_2CM_HL04])

def wigner_correction(scan,T=310):

	# kappa = 1 + (hbar * beta)^2 * w_b^2 / 24
	# [kappa] = [/s^2] * [s^2]

	hbar = 1.05457182e-34 # m^2 kg / s
	k_b = 1.380649e-23 # m^2 kg / s^2 / K
	c = 299792458 # m / s
	
	beta = 1.0 / (k_b * T) # s^2 /m^2 /kg

	for index in scan.neb_maxima_indices:

		# get the vibrational analysis cp2k output filename
		pattern = f'../VibrationScan/{scan.key}_ts_{index}_2/*.out'
		cp2k_out = glob.glob(pattern)
		if not cp2k_out:
			mout.errorOut(f"No vibration analysis pattern: {pattern}")
			continue
		cp2k_out = cp2k_out[0]

		# parse cp2k output to get frequencies
		freqs = parse_cp2k_vib_out(cp2k_out)

		# get biggest negative frequency normal mode
		w_b = sorted([f for f in freqs if f < 0])[-1]

		# convert w_b [/cm] --> [/s]
		w_b = c * w_b

		# calculate kappa
		kappa = 1 + (hbar * beta)**2 * w_b**2 / 24

		print(f'{kappa=}')

def parse_cp2k_vib_out(file):
	freqs = []
	with open(file,'rt') as f:
		for line in f:
			if not line.startswith(" VIB|"):
				continue

			if line.startswith(" VIB|Frequency (cm^-1)"):
				freqs += line.split()[2:]
			elif line.startswith(" VIB|              Temperature [K]:"):
				temp = float(line.split()[-1])

	freqs = [float(f) for f in freqs]

	return freqs #, temp

def barrier_statistics(name,scans):

	mout.headerOut(name)

	can_tau_asym = []
	can_zwi_asym = []
	can_zwi_fbar = []
	can_zwi_rbar = []
	zwi_tau_asym = []
	zwi_tau_fbar = []
	zwi_tau_rbar = []
	
	for scan in scans:

		if scan.key == 'B3LYP_2CM_HL03':
			scan.neb_minima_indices = [0,7,-1]
			scan.neb_maxima_indices = [6,19]

		if len(scan.neb_minima_indices) != 3:
			print('missing minima',scan.key,scan.neb_minima_indices)
			continue
		if len(scan.neb_maxima_indices) != 2:
			print('missing maxima',scan.key,scan.neb_maxima_indices)
			continue

		energies = scan.get_neb_energies()

		can_tau_asym.append(energies[scan.neb_minima_indices[2]]-energies[scan.neb_minima_indices[0]])
		can_zwi_asym.append(energies[scan.neb_minima_indices[1]]-energies[scan.neb_minima_indices[0]])
		zwi_tau_asym.append(energies[scan.neb_minima_indices[2]]-energies[scan.neb_minima_indices[1]])

		can_zwi_fbar.append(energies[scan.neb_maxima_indices[0]]-energies[scan.neb_minima_indices[0]])
		zwi_tau_fbar.append(energies[scan.neb_maxima_indices[1]]-energies[scan.neb_minima_indices[1]])

		# rbar = energies[scan.neb_maxima_indices[0]-scan.neb_minima_indices[1]]

		can_zwi_rbar.append(energies[scan.neb_maxima_indices[0]]-energies[scan.neb_minima_indices[1]])
		zwi_tau_rbar.append(energies[scan.neb_maxima_indices[1]]-energies[scan.neb_minima_indices[2]])
		
	print_stat('can_tau_asym', can_tau_asym, positive_only=True)
	print_stat('can_zwi_asym',can_zwi_asym, positive_only=True)
	print_stat('can_zwi_fbar',can_zwi_fbar, positive_only=True)
	print_stat('can_zwi_rbar',can_zwi_rbar, positive_only=True)
	print_stat('zwi_tau_asym',zwi_tau_asym, positive_only=False)
	print_stat('zwi_tau_fbar',zwi_tau_fbar, positive_only=True)
	print_stat('zwi_tau_rbar',zwi_tau_rbar, positive_only=True)

def save_neb_profile(scan,smooth=False,show=False,subdir=None):

	# path linalg

	ase_rc = scan.get_neb_rcdist(walls=[1,3])
	rcs = scan.get_neb_rcpath(walls=[1,3])

	xs = []
	ys = []

	interpolator = scan.get_interpolator()
	
	for x,rc1,rc2 in zip(ase_rc,rcs[0],rcs[1]):
		xs.append(x)
		ys.append(interpolator(rc1,rc2))
	
	profile_interpolator = interp1d(xs, ys, kind='quadratic')

	if subdir:
		os.system(f'mkdir -p profiles/{subdir}')
		file = f'profiles/{subdir}/{scan.key}.dat'
	else:
		file = f'profiles/{scan.key}.dat'

	with open(file,'wt') as f:
		f.write("# ase_rc np.linalg.norm([rc1,rc3]) [Å], energy (eV)\n")
		for x in np.linspace(min(xs), max(xs), 200):
			f.write(f'{x} {profile_interpolator(x)}\n')

	# plot
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=xs,y=ys))

	xs = list(np.linspace(min(xs), max(xs), 100))
	ys = [profile_interpolator(x) for x in xs]
	fig.add_trace(go.Scatter(x=xs,y=ys))

	if show:
		fig.show()


	# gmx rc

	gmx_rc = scan.get_gmx_like_neb_path([1,3])
	rcs = scan.get_neb_rcpath([1,3])

	xs = []
	ys = []

	interpolator = scan.get_interpolator()
	for x,rc1,rc2 in zip(gmx_rc,rcs[0],rcs[1]):
		xs.append(x)
		ys.append(interpolator(rc1,rc2))
	profile_interpolator = interp1d(xs, ys, kind='quadratic')

	with open(f'{scan.key}/neb_profile.dat','wt') as f:
		f.write("# gmx_rc (rc1+rc3)/2 [Å], energy (eV)\n")
		interpolator = scan.get_interpolator()
		for x,rc1,rc2 in zip(gmx_rc,rcs[0],rcs[1]):
			f.write(f'{x} {interpolator(rc1,rc2)}\n')

	with open(f'{scan.key}/neb_profile_smooth.dat','wt') as f:
		f.write("# gmx_rc (rc1+rc3)/2 [Å], energy (eV)\n")
		for x in np.linspace(min(xs), max(xs), 200):
			f.write(f'{x} {profile_interpolator(x)}\n')

def plot_neb_vs_decay(scan):

	fig = go.Figure()

	NEB_path, NEB_e, trace = get_string_trace([image.rcs for image in scan.neb_images], scan.interpolator)
	
	distances = []
	for p in NEB_path:
		d = np.linalg.norm(np.array(p)-np.array(scan.can.rcs))
		distances.append(d)

	trace = go.Scatter(name='NEB',x=distances,y=NEB_e)
	# fig.add_trace(trace)

	distances = []
	times = []
	for i,(a,b) in enumerate(zip(scan.decay_path_rc1,scan.decay_path_rc3)):
		d = np.linalg.norm(np.array([a,b])-np.array(scan.can.rcs))
		distances.append(d)
		times.append(i)
		# if d < 0.1:
		# 	break

	energies = [e-scan.decay_energies[1]+NEB_e[-1] for e in scan.decay_energies]

	trace = go.Scatter(name='decay',x=distances,y=energies)
	fig.add_trace(trace)
	
	trace = go.Scatter3d(name='decay',x=times,y=distances,z=energies)
	fig.update_layout(scene=dict(yaxis_title='dist. to can. (Å)',zaxis_title='QM energy (eV)',xaxis_title='time (fs)'))
	
	fig.add_trace(trace)

	fig.update_layout(title=scan.key)
	
	fig.show()

def plot_decay_rcs(scan,show=True):

	mout.debugOut(f'plot_decay_rcs({scan.key})')

	fig = go.Figure()

	# calculate delta to can.rcs

	distances = []
	times = []
	for i,(a,b) in enumerate(zip(scan.decay_path_rc1,scan.decay_path_rc3)):
		d = np.linalg.norm(np.array([a,b])-np.array(scan.can.rcs))
		distances.append(-d)
		times.append(i)
		if d < 0.01:
			scan.decay_can_arrival = i

	trace = go.Scatter(name='dist. to can.',x=times,y=distances)
	fig.add_trace(trace)
	
	trace = go.Scatter(name='RC1',x=times,y=scan.decay_path_rc1[:len(times)])
	fig.add_trace(trace)

	trace = go.Scatter(name='RC2',x=times,y=scan.decay_path_rc2[:len(times)])
	fig.add_trace(trace)

	trace = go.Scatter(name='RC1+RC2',x=times,y=np.array(scan.decay_path_rc1[:len(times)])+np.array(scan.decay_path_rc2[:len(times)]))
	fig.add_trace(trace)

	trace = go.Scatter(name='RC1-RC2',x=times,y=np.array(scan.decay_path_rc1[:len(times)])-np.array(scan.decay_path_rc2[:len(times)]))
	fig.add_trace(trace)

	trace = go.Scatter(name='RC3',x=times,y=scan.decay_path_rc3[:len(times)])
	fig.add_trace(trace)

	trace = go.Scatter(name='RC4',x=times,y=scan.decay_path_rc4[:len(times)])
	fig.add_trace(trace)

	trace = go.Scatter(name='RC3+RC4',x=times,y=np.array(scan.decay_path_rc3[:len(times)])+np.array(scan.decay_path_rc4[:len(times)]))
	fig.add_trace(trace)

	trace = go.Scatter(name='RC3-RC4',x=times,y=np.array(scan.decay_path_rc3[:len(times)])-np.array(scan.decay_path_rc4[:len(times)]))
	fig.add_trace(trace)

	scan.rc1_rc2_crossover = times[-1]
	for t, rc1, rc2 in zip(times,scan.decay_path_rc1,scan.decay_path_rc2):
		if rc2 > rc1:
			scan.rc1_rc2_crossover = t
			break
	mout.varOut("RC1/2 crossover",scan.rc1_rc2_crossover,unit='fs')

	scan.rc3_rc4_crossover = times[-1]
	for t, rc3, rc4 in zip(times,scan.decay_path_rc3,scan.decay_path_rc4):
		if rc4 > rc3:
			scan.rc3_rc4_crossover = t
			break
	mout.varOut("RC3/4 crossover",scan.rc3_rc4_crossover,unit='fs')

	fig.add_vline(scan.rc1_rc2_crossover)
	fig.add_vline(scan.rc3_rc4_crossover)

	fig.update_layout(xaxis_title='time (fs)',yaxis_title='distance (Å)')

	fig.update_layout(title=scan.key)

	if show:
		fig.show()

def create_ts_inputs(scan):
	for index in scan.neb_maxima_indices:
		rc1,rc2 = scan.neb_images[index].rcs
		create_decay_input_gro(scan, rc1, rc2, out=f'neb.ts.{index}.pdb',show=False,pdb=True,qm_only=True)

def create_decay_input_gro(scan,rc1,rc2,show=False,out='scan.tau.vel.gro',pdb=False,filter=None,qm_only=False):

	# get a gro reference
	if '2DM' in scan.key:

		if pdb:
			sys = mp.parse('../ArcherUmbrellaDNA/L01R1_w000/L01R1_w000.pdb')
		else:
			sys = mp.parse('../ArcherUmbrellaDNA/2DM_top/2DM_rdy.gro')
		ndx = mp.parseNDX('../ArcherUmbrellaDNA/index.ndx')

	elif '2CM' in scan.key:
		if pdb:
			sys = mp.parse('../ArcherUmbrellaHelicase/HL01R1_BLYP_VDW_w000/HL01R1_BLYP_VDW_w000.pdb')
		else:
			sys = mp.parse('../ArcherUmbrellaHelicase/2CM_top/2CM_rdy.gro')
		ndx = mp.parseNDX('../ArcherUmbrellaHelicase/index_2CM.ndx')

	else:
		mout.errorOut("Unsupported",fatal=True)

	ndx.shift(-1)

	# ndx.summary()

	# set the coordinates based on window reference
	# ref_window = f'{scan.rst_key}/window_00_00.pdb'
	ref_window = f'{scan.rst_key}/confout.gro'
	sys.set_coordinates(ref_window,velocity=True)

	# get indices of relevant atoms
	index_DCH41 = ndx['DCH41'][0]
	index_DGH1  = ndx['DGH1'][0]
	index_DCN4 = ndx['DCN4'][0]
	index_DGO6 = ndx['DGO6'][0]
	index_DCN3 = ndx['DCN3'][0]
	index_DGN1 = ndx['DGN1'][0]

	# get the atoms
	DCH41 = sys.atoms[index_DCH41]
	DGH1 = sys.atoms[index_DGH1]
	DCN4 = sys.atoms[index_DCN4]
	DGO6 = sys.atoms[index_DGO6]
	DCN3 = sys.atoms[index_DCN3]
	DGN1 = sys.atoms[index_DGN1]

	assert DCH41.species == "H"
	assert DGH1.species == "H"
	assert DCN4.species == "N"
	assert DGO6.species == "O"
	assert DCN3.species == "N"
	assert DGN1.species == "N"

	# hydrogen bond vectors
	vector_DCN4_DGO6 = DGO6 - DCN4
	vector_DGN1_DCN3 = DCN3 - DGN1

	# move the hydrogens
	DCH41.position = DCN4 + rc1 * vector_DCN4_DGO6 / np.linalg.norm(vector_DCN4_DGO6)
	DGH1.position = DGN1 + rc2 * vector_DGN1_DCN3 / np.linalg.norm(vector_DGN1_DCN3)

	vel_along_rc1 = np.dot(vector_DCN4_DGO6,DCH41.np_vel)
	vel_along_rc3 = np.dot(vector_DGN1_DCN3,DGH1.np_vel)

	mout.varOut("DCH41 velocity",np.linalg.norm(DCH41.np_vel),'Å/fs')
	mout.varOut("DGH1 velocity",np.linalg.norm(DGH1.np_vel),'Å/fs')

	DCH41_kin = 0.5*1.67262192e-27*np.power(np.linalg.norm(DCH41.np_vel)*1.0e5,2)/1.602e-19
	DGH1_kin = 0.5*1.67262192e-27*np.power(np.linalg.norm(DGH1.np_vel)*1.0e5,2)/1.602e-19

	mout.varOut("DCH41 E_Kin",DCH41_kin,'eV')
	mout.varOut("DGH1 E_Kin",DGH1_kin,'eV')

	mout.varOut("vel_along_rc1",vel_along_rc1)
	mout.varOut("vel_along_rc3",vel_along_rc3)

	if qm_only:
		
		outsys = mp.AtomGroup.from_any("QM",sys['rQM'])

		# print(outsys)
		# print(outsys.summary())

		index_DGC1p = ndx["DGC1'"][0]
		index_DCC1p = ndx["DCC1'"][0]
		index_DGN9 = ndx["DGN9"][0]
		index_DCN1 = ndx["DCN1"][0]

		print(index_DGC1p)
		print(index_DGN9)

		DGC1p = sys.atoms[index_DGC1p]
		DCC1p = sys.atoms[index_DCC1p]
		DGN9 = sys.atoms[index_DGN9]
		DCN1 = sys.atoms[index_DCN1]

		DGN9.summary()
		DGC1p.summary()

		link1 = mp.mutate.get_link_atom(DGN9,DGC1p)
		outsys.add_atom(link1)

		link2 = mp.mutate.get_link_atom(DCN1,DCC1p)
		outsys.add_atom(link2)
		
		mp.write(f'{scan.key}/{out}',outsys)

		if show:
			outsys.plot3d()

	else:

		# write out a new gro file
		mp.write(f'{scan.key}/{out}',sys)

		if show:
			show_group = mp.AtomGroup.from_any("G*C*",sys.residues[13,34])
			show_group.plot3d(velocity=True,v_scale=10.0,alpha=0.5)

	return f'{scan.key}/{out}'

def plot_compare_profile(scans,statistics=False,align=False,xvg_compare=None,ts1_map=1.58,show=False,file=None):

	fig = go.Figure()

	if not statistics:

		if align:

			for scan in scans:

				# split the data around stationary points
				raw_neb_distance = scan.get_neb_rcdist()
				raw_neb_energies = scan.get_neb_energies()
				
				pre_ts1_distance = np.array(raw_neb_distance[:scan.neb_maxima_indices[0]+1])
				pre_ts1_energies = np.array(raw_neb_energies[:scan.neb_maxima_indices[0]+1])

				pre_zwit_distance = np.array(raw_neb_distance[scan.neb_maxima_indices[0]:scan.neb_minima_indices[1]+1])
				pre_zwit_energies = np.array(raw_neb_energies[scan.neb_maxima_indices[0]:scan.neb_minima_indices[1]+1])

				post_zwit_distance = np.array(raw_neb_distance[scan.neb_minima_indices[1]:scan.neb_maxima_indices[-1]+1])
				post_zwit_energies = np.array(raw_neb_energies[scan.neb_minima_indices[1]:scan.neb_maxima_indices[-1]+1])

				post_ts2_distance = np.array(raw_neb_distance[scan.neb_maxima_indices[-1]:])
				post_ts2_energies = np.array(raw_neb_energies[scan.neb_maxima_indices[-1]:])

				# rescale
				pre_ts1_distance = 0.5*pre_ts1_distance/pre_ts1_distance[-1]
				pre_zwit_distance = 0.5*(pre_zwit_distance-pre_zwit_distance[0])/(pre_zwit_distance-pre_zwit_distance[0])[-1]+0.5
				post_zwit_distance = 0.5*(post_zwit_distance-post_zwit_distance[0])/(post_zwit_distance-post_zwit_distance[0])[-1]+1.0
				post_ts2_distance = 0.5*(post_ts2_distance-post_ts2_distance[0])/(post_ts2_distance-post_ts2_distance[0])[-1]+1.5

				x_data = list(pre_ts1_distance) + list(pre_zwit_distance) + list(post_zwit_distance) + list(post_ts2_distance)
				y_data = list(pre_ts1_energies) + list(pre_zwit_energies) + list(post_zwit_energies) + list(post_ts2_energies)

				trace = go.Scatter(name='NEB Profile',legendgroup=scan.key,legendgrouptitle_text=f'{scan.key} {scan.bb_sep:.1f}',x=x_data,y=y_data)
				fig.add_trace(trace)

			fig.update_layout(xaxis_title='Normalised Transfer Coordinate',yaxis_title='Energy (eV)',title=f'{" ".join([s.key for s in scans])}')

		else:

			for scan in scans:

				# trace = go.Scatter(name='Stationary Points',legendgroup=scan.key,legendgrouptitle_text=f'{scan.key} {scan.bb_sep:.1f}',x=scan.neb_stationary_x,y=scan.neb_stationary_y,mode='markers')
				# fig.add_trace(trace)

				# trace = go.Scatter(name='NEB Profile',legendgroup=scan.key,legendgrouptitle_text=f'{scan.key} {scan.bb_sep:.1f}',x=scan.get_neb_rcdist(),y=scan.get_neb_energies())
				trace = go.Scatter(name=scan.key,x=scan.get_neb_rcdist(),y=scan.get_neb_energies())
				fig.add_trace(trace)

			# fig.update_layout(xaxis_title='Path Distance',yaxis_title='Energy (eV)',title=f'{" ".join([s.key for s in scans])}')
			fig.update_layout(xaxis_title='Path Distance',yaxis_title='Energy (eV)')

	else:

		# use normalised transfer coordinate

		x_data = []
		y_mean_data = []
		y_std_max_data = []
		y_std_min_data = []
		y_min_data = []
		y_max_data = []

		interpolators = [scan.get_normalised_neb_interpolator() for scan in scans]

		for x in np.arange(0,2,0.01):

			sliced = [i(x) for i in interpolators]

			x_data.append(x)

			mean = np.mean(sliced)
			std = np.std(sliced)

			y_mean_data.append(mean)
			y_std_max_data.append(mean+std)
			y_std_min_data.append(mean-std)
			y_max_data.append(max(sliced))
			y_min_data.append(min(sliced))

		color = 'rgb(0,0,0)' # default is black

		fig.add_trace(go.Scatter(x=x_data,y=y_max_data,name="max",line=dict(width=0,color=color)))
		fig.add_trace(go.Scatter(x=x_data,y=y_min_data,name="min",fill='tonexty',line=dict(width=0,color=color),fillcolor=color.replace('rgb','rgba').replace(')',',0.15)')))
		fig.add_trace(go.Scatter(x=x_data,y=y_std_max_data,name="mean+std",line=dict(width=0,color=color)))
		fig.add_trace(go.Scatter(x=x_data,y=y_std_min_data,name="mean-std",fill='tonexty',line=dict(width=0,color=color),fillcolor=color.replace('rgb','rgba').replace(')',',0.3)')))
		fig.add_trace(go.Scatter(x=x_data,y=y_mean_data,name="mean",line=dict(color=color,width=4)))

		fig.add_trace(go.Scatter(x=x_data,y=y_max_data,name="max",line=dict(width=0,color=color)))
		fig.add_trace(go.Scatter(x=x_data,y=y_min_data,name="min",fill='tonexty',line=dict(width=0,color=color),fillcolor=color.replace('rgb','rgba').replace(')',',0.15)')))
		fig.add_trace(go.Scatter(x=x_data,y=y_std_max_data,name="mean+std",line=dict(width=0,color=color)))
		fig.add_trace(go.Scatter(x=x_data,y=y_std_min_data,name="mean-std",fill='tonexty',line=dict(width=0,color=color),fillcolor=color.replace('rgb','rgba').replace(')',',0.3)')))
		fig.add_trace(go.Scatter(x=x_data,y=y_mean_data,name="mean",line=dict(color=color,width=4)))

		fig.update_layout(xaxis_title='Normalised Transfer Coordinate',yaxis_title='Energy (eV)',title=f'{" ".join([s.key for s in scans])}')

	if xvg_compare:

		bs_collection = mp.parseXVG(xvg_compare,yscale=1/23)

		bs_collection.align_ydata(min)

		bs_collection.smooth()

		bs_collection.map_xdata(1.05,ts1_map,0.0,1.0)

		fig = bs_collection.plotly(fig=fig,statistics=True,color='rgb(1.0,0.0,0.0)',no_layout=True)

	if show:
		fig.show()

	if file:
		mp.write(file,fig)

def plot_compare_path(scans):

	fig = go.Figure()

	for scan in scans:
		x,y = scan.get_neb_rcpath()
		trace = go.Scatter(name=scan.key,x=x,y=y)
		fig.add_trace(trace)

	fig.update_layout(xaxis_title='RC1',yaxis_title='RC2',title=f'{" ".join([s.key for s in scans])}')

	fig.show()

def plot_comparse_sep_properties(scans):

	ts1_x = []
	ts1_y = []

	ts2_x = []
	ts2_y = []

	zwi_x = []
	zwi_y = []

	tau_x = []
	tau_y = []

	for scan in sorted(scans,key=lambda x: x.bb_sep):

		if len(scan.neb_stationary_y) == 5:

			ts1_x.append(scan.bb_sep)
			ts1_y.append(scan.neb_stationary_y[1])
			
			ts2_x.append(scan.bb_sep)
			ts2_y.append(scan.neb_stationary_y[3])
			
			zwi_x.append(scan.bb_sep)
			zwi_y.append(scan.neb_stationary_y[2])

			tau_x.append(scan.bb_sep)
			tau_y.append(scan.neb_stationary_y[4])

		elif len(scan.neb_stationary_y) == 3:
			
			ts2_x.append(scan.bb_sep)
			ts2_y.append(scan.neb_stationary_y[1])
			
			tau_x.append(scan.bb_sep)
			tau_y.append(scan.neb_stationary_y[2])


	fig = go.Figure()

	fig.add_trace(go.Scatter(name='ts1',x=ts1_x,y=ts1_y,mode='markers'))
	fig.add_trace(go.Scatter(name='ts2',x=ts2_x,y=ts2_y,mode='markers'))
	fig.add_trace(go.Scatter(name='zwi',x=zwi_x,y=zwi_y,mode='markers'))
	fig.add_trace(go.Scatter(name='tau',x=tau_x,y=tau_y,mode='markers'))

	fig.update_layout(xaxis_title='BB Separation',yaxis_title='Energy (eV)',title=f'{" ".join([s.key for s in scans])}')

	fig.show()

def plot_compare_3dsep(scans):

	fig = go.Figure()

	for scan in scans:

		y = [scan.bb_sep for y in scan.neb_stationary_y]
		trace = go.Scatter3d(name='Stationary Points',legendgroup=scan.key,legendgrouptitle_text=f'{scan.key} {scan.bb_sep:.1f}',x=scan.neb_stationary_x,y=y,z=scan.neb_stationary_y,mode='markers')
		fig.add_trace(trace)

		y = [scan.bb_sep for y in scan.get_neb_energies()]
		trace = go.Scatter3d(name='NEB Profile',legendgroup=scan.key,legendgrouptitle_text=f'{scan.key} {scan.bb_sep:.1f}',x=scan.get_neb_rcdist(),y=y,z=scan.get_neb_energies(),mode='lines')
		fig.add_trace(trace)

	fig.update_layout(xaxis_title='Path Distance',yaxis_title='Energy (eV)',title=f'{" ".join([s.key for s in scans])}')

	fig.show()

def scan_routine(key,rst_key,can,zwi,alt,tau, wall_height = 10, neb_fmax=0.04, hicut=2.0,path='.',cap=None,mid_walls=False):

	mout.headerOut(f"@@@ {key}")

	os.system("rm *.traj")

	n_images = 30
	k = 60

	scan = Scan(key,rst_key=rst_key,wall_height=wall_height,hicut=hicut,path=path,cap=cap)

	# read the energies
	d = scan.point_list

	# plot raw heatmap
	fig = scan.plot_heatmap()

	mp.write(f"{path}/{scan.key}/raw_heatmap.pdf",fig)

	# build the wall
	scan.extend_walls(mid_walls)

	# interpolation
	cubic_grid = scan.get_interpolated_grid()
	clough_interpolator = scan.get_interpolator()

	trimmed_grid = trim_grid(cubic_grid,hicut=scan.hicut)

	### Optimizer hijack

	endpoints = [can,tau,zwi,alt]

	scan.can = can
	scan.tau = tau
	scan.zwi = zwi
	scan.alt = alt

	calc = hijack.FakeSurfaceCalculator(pmf=clough_interpolator)

	for atoms in endpoints:

		scan.optimize(atoms)

		mout.varOut(f"Optimised ({atoms.name}) RCS",atoms.rcs)
		mout.varOut(f"Optimised ({atoms.name}) energy",clough_interpolator(*atoms.rcs))

	if abs(zwi.rcs[0] - tau.rcs[0]) <= 0.01 and abs(zwi.rcs[1] - tau.rcs[1]) <= 0.01:
		mout.warningOut("Tautomeric state is not stable!")

	if abs(zwi.rcs[0] - can.rcs[0]) <= 0.01 and abs(zwi.rcs[1] - can.rcs[1]) <= 0.01:
		mout.warningOut("Zwitterion state is not stable!")

	if abs(alt.rcs[0] - can.rcs[0]) <= 0.01 and abs(alt.rcs[1] - can.rcs[1]) <= 0.01:
		mout.warningOut("Alternative zwitterion state is not stable!")

	# shift the grid
	scan.shift_grid(clough_interpolator(*can.rcs))
	cubic_grid = scan.get_interpolated_grid()
	clough_interpolator = scan.get_interpolator()
	trimmed_grid = trim_grid(cubic_grid,hicut=scan.hicut)

	### 2d surface

	fig = scan.plot_heatmap(interpolated=True,show=False)

	trace = go.Scatter(name='Canonical',x=[can.rcs[0]],y=[can.rcs[1]])
	fig.add_trace(trace)

	trace = go.Scatter(name='Zwitterion',x=[zwi.rcs[0]],y=[zwi.rcs[1]])
	fig.add_trace(trace)

	trace = go.Scatter(name='Alternate',x=[alt.rcs[0]],y=[alt.rcs[1]])
	fig.add_trace(trace)

	trace = go.Scatter(name='Tautomer',x=[tau.rcs[0]],y=[tau.rcs[1]])
	fig.add_trace(trace)

	fig.update_layout(title=f'{key}: Optimised Minima')

	mp.write(f"{path}/{scan.key}/heatmap_minima.pdf",fig)

	# fig.show()

	### NEB hijack

	# run the NEB
	scan.neb(n_images=n_images,k=k,reactant=can,product=tau,fmax=neb_fmax)

	fig = scan.plot_heatmap(interpolated=True,show=False,trim=True)

	x,y = scan.get_neb_rcpath()
	fig.add_trace(go.Scatter(name='NEB',x=x,y=y))

	fig.update_layout(
		xaxis_constrain="domain",
		yaxis_constrain="domain",
		yaxis_scaleanchor="x",
		yaxis_scaleratio=1,
	)

	fig.add_trace(trace)

	fig.update_layout(title=f'{key}: Optimised Minima & Path')

	mp.write(f"{path}/{scan.key}/heatmap_path.pdf",fig)

	# fig.show()

	### basic paths

	concerted_path = [can.rcs,tau.rcs]
	concerted_path, concerted_e, trace = get_string_trace(concerted_path, clough_interpolator,50,'concerted')

	asynchronous_path = [can.rcs,zwi.rcs,tau.rcs]
	asynchronous_path, asynchronous_e, trace = get_string_trace(asynchronous_path, clough_interpolator,25,'asynchronous')

	alternate_path = [can.rcs,alt.rcs,tau.rcs]
	alternate_path, alternate_e, trace = get_string_trace(alternate_path, clough_interpolator,25,'alternate')

	### 3d surface plots

	fig = go.Figure()

	NEB_path, NEB_e, trace = get_string_trace([image.rcs for image in scan.neb_images], scan.get_interpolator())
	fig.add_trace(trace)

	trace = go.Surface(name='cubic',x=trimmed_grid[0],y=trimmed_grid[1],z=trimmed_grid[2],
	# trace = go.Surface(name='cubic',x=cubic_grid[0],y=cubic_grid[1],z=cubic_grid[2],
			contours = {
				"z": {"show": True, "start": 0.0, "end": 6.0, "size": 0.1, "color": 'white', 'width': 1},
			}
		)
	fig.add_trace(trace)

	trace = go.Scatter3d(name='Canonical',x=[can.rcs[0],can.rcs[0]],y=[can.rcs[1],can.rcs[1]],z=[-1.0,6.0],mode='lines')
	fig.add_trace(trace)
	trace = go.Scatter3d(name='Canonical',x=[tau.rcs[0],tau.rcs[0]],y=[tau.rcs[1],tau.rcs[1]],z=[-1.0,6.0],mode='lines')
	fig.add_trace(trace)
	trace = go.Scatter3d(name='Alternate',x=[alt.rcs[0],alt.rcs[0]],y=[alt.rcs[1],alt.rcs[1]],z=[-1.0,6.0],mode='lines')
	fig.add_trace(trace)
	trace = go.Scatter3d(name='Zwitterion',x=[zwi.rcs[0],zwi.rcs[0]],y=[zwi.rcs[1],zwi.rcs[1]],z=[-1.0,6.0],mode='lines')
	fig.add_trace(trace)

	fig.update_xaxes(range=[scan.locut,scan.hicut],constrain='domain')
	fig.update_yaxes(range=[scan.locut,scan.hicut],constrain='domain')

	fig.update_layout(title=f'{key}: Optimised Minima & Path')

	fig.update_layout(xaxis_title='RC1')
	fig.update_layout(yaxis_title='RC2')

	fig.show()

	### 1d profiles

	fig = go.Figure()

	trace = profile_trace(name="concerted",path=concerted_path,energies=concerted_e)
	fig.add_trace(trace)

	trace = profile_trace(name="asynchronous",path=asynchronous_path,energies=asynchronous_e)
	fig.add_trace(trace)

	trace = profile_trace(name="alternate",path=alternate_path,energies=alternate_e)
	fig.add_trace(trace)

	trace = profile_trace(name="NEB",path=[image.rcs for image in scan.neb_images],energies=NEB_e)
	fig.add_trace(trace)

	mp.write(f"{path}/{scan.key}/profiles.pdf",fig)

	# get decay trajectory
	decay_pullx = glob.glob(f'../TautomerStability/{scan.key}_decay/pullx.xvg')
	for file in decay_pullx:
		scan.load_decay_path(file)

	# fig.show()

	return scan

def lifetime_summary(name,scans):

	normal_rc1_rc2 = []
	normal_rc3_rc4 = []

	alternate_rc1_rc2 = []
	alternate_rc3_rc4 = []

	for scan in scans:

		try:

			if scan.rc1_rc2_crossover < scan.rc3_rc4_crossover:
				normal_rc1_rc2.append(scan.rc1_rc2_crossover)
				normal_rc3_rc4.append(scan.rc3_rc4_crossover-scan.rc1_rc2_crossover)
			elif scan.rc1_rc2_crossover == scan.rc3_rc4_crossover:
					continue
			else:
				print(scan.key,scan.rc1_rc2_crossover,scan.rc3_rc4_crossover)
				alternate_rc1_rc2.append(scan.rc1_rc2_crossover)
				alternate_rc3_rc4.append(scan.rc3_rc4_crossover)
			
		except AttributeError:
			print(scan.key)
			continue

	print_stat('normal_rc1_rc2', normal_rc1_rc2, positive_only=True)
	print_stat('normal_rc3_rc4', normal_rc3_rc4, positive_only=True)

	if alternate_rc1_rc2:
		print_stat('alternate_rc1_rc2', alternate_rc1_rc2, positive_only=True)
	if alternate_rc3_rc4:
		print_stat('alternate_rc3_rc4', alternate_rc3_rc4, positive_only=True)

def plot_lifetime_statistics(scans,debug=False):

	fig = go.Figure()

	dna_rc1_rc2_lifetimes = []
	dna_rc3_rc4_lifetimes = []
	dna_can_arrival = []

	pcra_rc1_rc2_lifetimes = []
	pcra_rc3_rc4_lifetimes = []
	pcra_can_arrival = []

	for scan in scans:

		print(scan.key)

		# fig.add_hline(0)

		if '2CM' in scan.key:

			# if 'N624A' in scan.key:

			color='red'

			pcra_rc1_rc2_lifetimes.append(scan.rc1_rc2_crossover)
			pcra_rc3_rc4_lifetimes.append(scan.rc3_rc4_crossover)

			try:
				pcra_can_arrival.append(scan.decay_can_arrival)
			except AttributeError:
				mout.warningOut(f"{scan.key} has no decay_can_arrival")

			# else:

			# color='red'

			# n624a_rc1_rc2_lifetimes.append(scan.rc1_rc2_crossover)
			# n624a_rc3_rc4_lifetimes.append(scan.rc3_rc4_crossover)

			# try:
			# 	n624a_can_arrival.append(scan.decay_can_arrival)
			# except AttributeError:
			# 	mout.warningOut(f"{scan.key} has no decay_can_arrival")

		else:
			color='blue'

			dna_rc1_rc2_lifetimes.append(scan.rc1_rc2_crossover)
			dna_rc3_rc4_lifetimes.append(scan.rc3_rc4_crossover)

			try:
				dna_can_arrival.append(scan.decay_can_arrival)
			except AttributeError:
				mout.warningOut(f"{scan.key} has no decay_can_arrival")

		# trace = go.Bar(x=[f'{scan.key} (rc1_rc2)'],y=[scan.rc1_rc2_crossover],marker_color=color)
		# fig.add_trace(trace)
		# trace = go.Bar(x=[f'{scan.key} (rc3_rc4)'],y=[scan.rc3_rc4_crossover],marker_color=color)
		# fig.add_trace(trace)

	fig.update_layout(scattermode="group",yaxis_title='Time [fs]')

	x_bar = ['Proton 1 Crossover','Proton 2 Crossover','GC Arrival']

	means = []
	stdevs = []
	means.append(np.mean(dna_rc1_rc2_lifetimes))
	stdevs.append(np.std(dna_rc1_rc2_lifetimes)/np.sqrt(len(dna_rc1_rc2_lifetimes)))
	means.append(np.mean(dna_rc3_rc4_lifetimes))
	stdevs.append(np.std(dna_rc3_rc4_lifetimes)/np.sqrt(len(dna_rc3_rc4_lifetimes)))
	means.append(np.mean(dna_can_arrival))
	stdevs.append(np.std(dna_can_arrival)/np.sqrt(len(dna_can_arrival)))

	print(dna_rc1_rc2_lifetimes)
	print(dna_rc3_rc4_lifetimes)
	print(dna_can_arrival)

	trace = go.Bar(name='Average',
		x=x_bar,y=means,error_y=dict(type='data',array=stdevs,visible=True),
		marker_color='blue',marker_opacity=0.6,
		offsetgroup="dna",
		legendgroup="dna",
		legendgrouptitle_text="DNA")
	fig.add_trace(trace)
	
	x_scatter = ['Proton 1 Crossover']*len(dna_rc1_rc2_lifetimes) + ['Proton 2 Crossover']*len(dna_rc3_rc4_lifetimes) + ['GC Arrival']*len(dna_can_arrival)

	trace = go.Scatter(name='DNA',x=x_scatter,y=dna_rc1_rc2_lifetimes+dna_rc3_rc4_lifetimes+dna_can_arrival,
		mode='markers',marker_color='blue',
		offsetgroup="dna",
		legendgroup="dna",
		legendgrouptitle_text="DNA")
	fig.add_trace(trace)

	means = []
	stdevs = []
	means.append(np.mean(pcra_rc1_rc2_lifetimes))
	stdevs.append(np.std(pcra_rc1_rc2_lifetimes)/np.sqrt(len(pcra_rc1_rc2_lifetimes)))
	means.append(np.mean(pcra_rc3_rc4_lifetimes))
	stdevs.append(np.std(pcra_rc3_rc4_lifetimes)/np.sqrt(len(pcra_rc3_rc4_lifetimes)))
	means.append(np.mean(pcra_can_arrival))
	stdevs.append(np.std(pcra_can_arrival)/np.sqrt(len(pcra_can_arrival)))
	
	trace = go.Bar(name='Average',
		x=x_bar,y=means,error_y=dict(type='data',array=stdevs,visible=True),
		marker_color='red',marker_opacity=0.6,
		offsetgroup="pcra",
		legendgroup="pcra",
		legendgrouptitle_text="PcrA")
	fig.add_trace(trace)
	
	x_scatter = ['Proton 1 Crossover']*len(pcra_rc1_rc2_lifetimes) + ['Proton 2 Crossover']*len(pcra_rc3_rc4_lifetimes) + ['GC Arrival']*len(pcra_can_arrival)

	trace = go.Scatter(name='PcrA',x=x_scatter,y=pcra_rc1_rc2_lifetimes+pcra_rc3_rc4_lifetimes+pcra_can_arrival,
		mode='markers',marker_color='red',
		offsetgroup="pcra",
		legendgroup="pcra",
		legendgrouptitle_text="PcrA")
	fig.add_trace(trace)

	# means = []
	# stdevs = []
	# means.append(np.mean(n624a_rc1_rc2_lifetimes))
	# stdevs.append(np.std(n624a_rc1_rc2_lifetimes)/np.sqrt(len(n624a_rc1_rc2_lifetimes)))
	# means.append(np.mean(n624a_rc3_rc4_lifetimes))
	# stdevs.append(np.std(n624a_rc3_rc4_lifetimes)/np.sqrt(len(n624a_rc3_rc4_lifetimes)))
	# means.append(np.mean(n624a_can_arrival))
	# stdevs.append(np.std(n624a_can_arrival)/np.sqrt(len(n624a_can_arrival)))
	
	# trace = go.Bar(name='Average',
	# 	x=x_bar,y=means,error_y=dict(type='data',array=stdevs,visible=True),
	# 	marker_color='red',marker_opacity=0.6,
	# 	offsetgroup="n624a",
	# 	legendgroup="n624a",
	# 	legendgrouptitle_text="n624a")
	# fig.add_trace(trace)
	
	# x_scatter = ['Proton 1 Crossover']*len(n624a_rc1_rc2_lifetimes) + ['Proton 2 Crossover']*len(n624a_rc3_rc4_lifetimes) + ['GC Arrival']*len(n624a_can_arrival)

	# trace = go.Scatter(name='n624a',x=x_scatter,y=n624a_rc1_rc2_lifetimes+n624a_rc3_rc4_lifetimes+n624a_can_arrival,
	# 	mode='markers',marker_color='red',
	# 	offsetgroup="n624a",
	# 	legendgroup="n624a",
	# 	legendgrouptitle_text="PcrA")
	# fig.add_trace(trace)

	fig.show()

	mp.write(f"lifetime_summary.pdf",fig)

if __name__ == '__main__':
	main()

"""

Repeat decays w/ longer runs: 

B3LYP_2DM_L11
B3LYP_2CM_HL01


"""