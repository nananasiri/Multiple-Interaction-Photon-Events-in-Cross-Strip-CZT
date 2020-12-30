import sys
from Nana_NEC_functions import *
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

"""
col5: panel ID
col6: module ID, in czt it is crystal ID
col7: intersection of anode cathode cellID
col13-15: x,y,z
"""

fname = sys.argv[1] # Go to Run/edit Configuration set data!

#build the table of all potential module positions, [height,lateral]
mtable = build_module_table() #mtabole dictionary has!

# print('mtable[0]', mtable['0']) # [0, 0, -40.0, -32.5]
# print('mtable[149]', mtable['149']) #[29, 4, 120.0, 117.5]


#build the table of all potential anode/cathode positions, [A,C]
actable = build_anode_cathode_table()
# print(actable['0'])
# print(actable['24'])
#=true if you want to print values to console
DEBUG = True #True nana
allbin = False #=False if you want cross strip 1mm x 5mm nana NOT PIXLATED!!!!

crystal = 5.0 #5.01 #the crystal itself is 5.0, just when repeating it uses 5.01
#bin the z position: divide the crystal into ~1.0mm pieces
dz = crystal / 10.
#print('dz:', dz, 'crystal:', crystal)
anode = 1.
cathode = 5.
dx = 1. #if pixellation is used, follow this size (mm)
dcat = 3.08 #if pixellation is used, follow this "Cathode" size (mm)
latcrystal = 40. #lateral size of crystal

dxyarray = []
dcatarray = []
for i in range(int(latcrystal / dx)):

	dxyarray.append(i*dx - latcrystal/2. + dx/2)
	dcatarray.append(i*dcat - latcrystal/2. + dcat/2)

dxyarray =  np.asarray(dxyarray)
dcatarray =  np.asarray(dcatarray)
# print('dxyarray=', dxyarray[1:4])
# print('dcatarray:::', dcatarray[1:4])
# print dxyarray
#exit()

# for i in mtable:
# 	print(i,mtable[i])
# for i in actable:
# 	print(i,actable[i])
#exit()

pan = {'0':0,'1':0} # col 5 = 0 0r 1
# print('pannnn', pan)
# print(pan)
pan2 = {'0':[],'1':[]}

with open(fname, 'r') as r, open(fname+'_bin'+str(dx)+'.dat','w') as w:
	for kk,line in enumerate(r): # har "line" ye khat az dataset has
		if kk > 1e7:# aeb = a * 10 b(tawan)
			break
			#exit()
		#parse the line
		l = parse_GATE_Hits_line(line) ##get rid of the spaces
		# print('LLL', l)
		#find panel's center wrt origin of phantom
		center = get_panel_center4(l[5])
		#find module height and lateral position
		# print('l[5]====', l[5])

		m = mtable[l[6]]
		# print('mtable::', mtable)
		# print('l[6],,,', l[6])
		# print('MMM,,,', m)
		#find crystal position
		cr = int(float(l[8]))	#l[7] submodule ID

		zcr = get_crystal_offset_position(cr) # cr : anode cathod numbero migire
		# print('cr:""""', cr)
		# print('zcr:""""', zcr)
		# find anode/cathode
		ac = actable[l[7]] # l[8] pixel ID
		# print("L[7]:", l[7])
		# print('AC:::', ac[l[8]])

		xreal,yreal,zreal = l[13:16]
		xreal = float(xreal)
		yreal = float(yreal)
		zreal = float(zreal)
		# print(xreal, yreal, zreal)
		ztest = zcr + m[3] + center[2]	# transferring crustal z from module frame to inertial frame
		panel = l[5]
		xtest0 = []
		if allbin == False:
			#estimate boundaries and make sure that point is within boundaries
			if panel == '0': #y-Anodes, x-Cathodes
				ytest = center[1] + (m[2] + ac[3]) # m[2] comes from mtable(L[6]:crystal ID 0< <150, ac[3] comes from [j i x y] which here ac[3] is y
				# print('center[1]:', center[1])
				# print('m[2]:', m[2])
				# print('ac[3]:', ac[3])
				xtest = center[0] + ac[2]
				# print('center[0]:', center[0])
				# print('ac[22222]:', ac[2]) # ac[2] = -17.5
				# print("xtest= ", xtest, "ytest=", ytest)
				# xtest0.append(xtest)
				# print('center[0]:', center[0])
				# print('xtest:', xtest)
				if xreal < xtest - cathode or xreal > xtest + cathode or yreal < ytest - anode or yreal > ytest + anode:

					print('real point outside of estimate Panel 0')
			# elif panel == '2': #y-Cathodes, x-Anodes
			# 	ytest = center[1] + ac[2]
			# 	xtest = center[0] - (m[2] + ac[3])
			# 	if xreal < xtest - anode or xreal > xtest + anode or yreal < ytest - cathode or yreal > ytest + cathode:
			# 		print('real point outside of estimate')
			elif panel == '1': #y-Anodes, x-Cathodes
				ytest = center[1] - (m[2] + ac[3])
				xtest = center[0] - ac[2]
				if xreal < xtest - cathode or xreal > xtest + cathode or yreal < ytest - anode or yreal > ytest + anode:
					print('real point outside of estimate Panel 1')
			# elif panel == '3': #y-Cathodes, x-Anodes
			# 	ytest = center[1] - ac[2]
			# 	xtest = center[0] + (m[2] + ac[3])
			# 	if xreal < xtest - anode or xreal > xtest + anode or yreal < ytest - cathode or yreal > ytest + cathode:
			# 		print('real point outside of estimate')
		else:
			if panel == '0': #y-Anodes, x-Cathodes
				ytest = center[1] + m[2]
				xtest = center[0]
				xarray = xtest + dcatarray
				yarray = ytest + dxyarray
			elif panel == '2': #y-Cathodes, x-Anodes
				ytest = center[1]
				xtest = center[0] - (m[2])
				xarray = xtest + dxyarray
				yarray = ytest + dcatarray
			elif panel == '1': #y-Anodes, x-Cathodes
				ytest = center[1] - (m[2])
				xtest = center[0]
				xarray = xtest + dcatarray
				yarray = ytest + dxyarray
			elif panel == '3': #y-Cathodes, x-Anodes
				ytest = center[1]
				xtest = center[0] + (m[2])
				xarray = xtest + dxyarray
				yarray = ytest + dcatarray


			#find the closest values in x and y
			idx = (np.abs(xarray-xreal)).argmin()
			xtest = xarray[idx]
			idx = (np.abs(yarray-yreal)).argmin()
			ytest = yarray[idx]

			#bin the y position
#			for i in range(int(latcrystal)/int(dx)):
#				#go from the side to see which z-bin this point is closest to
#				yguess = ytest - latcrystal/2. + dx/2. + i*dx
#				if abs(xreal - xguess) <= dx/2.:
#					xtest = xguess
#					break
#			#bin the x position
#			for i in range(5):
#				#go from bottom to top to see which z-bin this point is closest to
#				zguess = ztest - crystal/2. + dz/2. + i*dz
#				if abs(zreal - zguess) <= dz/2.:
#					ztest = zguess
#					break
		# if zreal < ztest - crystal or zreal > ztest + crystal:
		# 	print('real point outside of estimate')
		#bin the z position
		for i in range(5):
			#go from bottom to top to see which z-bin this point is closest to
			zguess = ztest - crystal/2. + dz/2. + i*dz
			if abs(zreal - zguess) <= dz/2.:
				ztest = zguess
				break

		#13-15 are the x,y,z coords
		w.write(' '+l[0]+' '+l[1]+' '+l[2]+' '+l[3]+' '+l[4]+' '+l[5]+' '+l[6]+' '+l[7]+' '+l[8]+' '+l[9]+' '+l[10]+' '+l[11]+' '+l[12]+' '+str(xtest)+' '+str(ytest)+' '+str(ztest)+' '+l[16]+' '+l[17]+' '+l[18]+' '+l[19]+' '+l[20]+' '+l[21]+' '+l[22]+' '+l[23]+' '+l[24])
		#w.write(' '+l[0]+' '+l[1]+' '+l[2]+' '+l[3]+' '+l[4]+' '+l[5]+' '+l[6]+' '+l[7]+' '+l[8]+' '+l[9]+' '+l[10]+' '+l[11]+' '+l[12]+' '+l[13]+' '+l[14]+' '+l[15]+' '+l[16]+' '+l[17]+' '+l[18]+' '+l[19]+' '+l[20]+' '+l[21]+' '+l[22]+' '+l[23]+' '+l[24])

		#set point equal to center of boundary in x/y/z

		if DEBUG == True:
			# print(l)
			# print(center)
			# print(m)
			# print(ac)
			# print(cr)
			# print(zcr)
			# # print(xarray[0],xarray[-1],yarray[0],yarray[-1])
			# print(xtest,ytest,ztest)
			# print(xreal,yreal,zreal)
			# print(panel)
			pan[str(panel)] += 1
			# pan2[str(panel)].append([xreal, yreal, zreal])
			pan2[str(panel)].append([xtest, ytest, ztest])

if DEBUG == True:
	for i in pan2:
		pan2[i] = np.asarray(pan2[i])
	# print('PAN2: ', pan2['0'],pan2['1'],pan2['2'],pan2['3'])

	# print(100*'-')
	# print(pan2['2'].shape)
	# print('pan2[2]', pan2['1'])

	plt.plot(pan2['0'][:,0],pan2['0'][:,1],'y')
	plt.plot(pan2['1'][:,0],pan2['1'][:,1],'k')
	# plt.plot(pan2['2'][:,0],pan2['2'][:,1],'b')#IndexError: too many indices for array
	# plt.plot(pan2['3'][:,0],pan2['3'][:,1],'k')



	plt.legend(['0','1'])

	num = []
	t = 0
	for r in xtest0:
		if r not in num:
			num.append(r)
			t += 1
	# plt.show()


	ax = plt.axes(projection='3d')
	#
	# print(100*'*')
	# print(pan2['0'][0][:])
	xdata = []
	ydata = []
	zdata = []
	for x in pan2['0']:
		xdata.append(x[0])
		ydata.append(x[1])
		zdata.append(x[2])
	panel1_plt = ax.scatter3D(xdata, ydata, zdata, cmap='Reds')

	xdata = []
	ydata = []
	zdata = []
	for x in pan2['1']:
		xdata.append(x[0])
		ydata.append(x[1])
		zdata.append(x[2])
	panel2_plt = ax.scatter3D(xdata, ydata, zdata, cmap='Blues')
	plt.legend((panel1_plt, panel2_plt), ('Panel 1', 'Panel 2'))

	ax.set_xlabel('X-Real')
	ax.set_ylabel('Y-Real')
	ax.set_zlabel('Z-Real')

	# ax.set_xlabel('X-Real')
	# ax.set_ylabel('Y-Real')
	# ax.set_zlabel('Z-Real')

	plt.show()
	# print(len(pan2['0'][0]))
	# print(pan2['0'])
	#
	# for x in pan2['0']:
	# 	print(x)
