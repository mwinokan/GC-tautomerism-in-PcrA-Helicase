#!/usr/bin/env python3

import time
import os
import numpy as np
import mout
import molparse as mp
import plotly.graph_objects as go
import json

coord_gro = None

scan_start = 0.9
scan_step = 0.1

# ls ../ArcherUmbrellaHelicase/HL*_w000/confout.gro

scan_end = 2.5
output_directory = '2CM_grid_HL05'
input_pdb = '../ArcherUmbrellaHelicase/HL04R1_BLYP_VDW_w000/HL04R1_BLYP_VDW_w000.pdb'
coord_gro = '../ArcherUmbrellaHelicase/HL05R1_BLYP_VDW_w000/confout.gro'
input_ndx = '../ArcherUmbrellaHelicase/index_2CM.ndx'

# ls ../ArcherUmbrellaDNA/L??R1_w000/confout.gro

# scan_end = 2.1
# output_directory = '2DM_grid_mid_L06'
# input_pdb = '../ArcherUmbrellaDNA/L06R1_w000/L06R1_w000.pdb'
# # coord_gro = '../ArcherUmbrellaDNA/L06R1_w000/confout.gro'
# coord_gro = '../ArcherUmbrellaDNA/L06R1_w000/confmid.gro'
# input_ndx = '../ArcherUmbrellaDNA/index.ndx'

# read ndx
ndx = mp.parseNDX(input_ndx)
ndx.summary()

# get indices of relevant atoms
index_DCH41 = ndx['DCH41'][0] - 1
index_DGH1  = ndx['DGH1'][0] - 1
index_DCN4 = ndx['DCN4'][0] - 1
index_DGO6 = ndx['DGO6'][0] - 1
index_DCN3 = ndx['DCN3'][0] - 1
index_DGN1 = ndx['DGN1'][0] - 1

mout.headerOut("INDICES")
print(f'{index_DCH41=}')
print(f'{index_DGH1=}')
print(f'{index_DCN4=}')
print(f'{index_DGO6=}')
print(f'{index_DCN3=}')
print(f'{index_DGN1=}')

# read pdb from US window
sys = mp.parse(input_pdb)

if coord_gro:
	sys.set_coordinates(coord_gro)

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

print(f'{vector_DCN4_DGO6=}',f'norm={np.linalg.norm(vector_DCN4_DGO6):.2f} Å')
print(f'{vector_DGN1_DCN3=}',f'norm={np.linalg.norm(vector_DGN1_DCN3):.2f} Å')

# create the grid of RC1 and RC3
grid_pairs = []
index_pairs = []
for i,rc1_value in enumerate(np.arange(scan_start,scan_end,scan_step)):
	for j,rc3_value in enumerate(np.arange(scan_start,scan_end,scan_step)):
		index_pairs.append([i,j])
		grid_pairs.append([rc1_value,rc3_value])

print(f'Number of grid-points {len(grid_pairs)}')

# grid figure
fig = go.Figure()

fig.add_trace(go.Scatter(name="Full H-Bond",x=[0,np.linalg.norm(vector_DCN4_DGO6)],y=[0,np.linalg.norm(vector_DGN1_DCN3)]))
fig.add_trace(go.Scatter(name="Scanning Grid",x=[p[0] for p in grid_pairs],y=[p[1] for p in grid_pairs],mode='markers'))

fig.update_layout(
	title="Diabatic Scanning Grid (Helicase)",
	xaxis_title="RC1 [Å]",
	yaxis_title="RC3 [Å]",
	yaxis_scaleanchor="x",
	yaxis_scaleratio=1,
)

fig.show()

# make a directory
os.makedirs(output_directory,exist_ok=True)

# infer new proton positions
for count, ((i, j), (rc1_value, rc3_value)) in enumerate(zip(index_pairs,grid_pairs)):

	start = time.perf_counter()

	outfile = f'{output_directory}/window_{i:02}_{j:02}.pdb'

	# new proton position will be the donor atom's position 
	# plus a scaled h-bond vector
	DCH41.position = DCN4 + rc1_value * vector_DCN4_DGO6 / np.linalg.norm(vector_DCN4_DGO6)
	DGH1.position = DGN1 + rc3_value * vector_DGN1_DCN3 / np.linalg.norm(vector_DGN1_DCN3)

	if not count:

		first_file = outfile

		mp.write(outfile,sys,verbosity=0)

	else:

		mp.modifyPDB(outfile,[DCH41,DGH1],copy_from=first_file)

	end = time.perf_counter()

	mout.progress(count,len(grid_pairs),prepend=f'{i=:02} {j=:02}',append=f' {end-start:.1f} sec')

mout.progress(count,count,prepend=f'{i=:02} {j=:02}',append='         DONE')
