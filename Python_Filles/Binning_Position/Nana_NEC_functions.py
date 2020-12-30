# file that stores functions for calculating things related to NEC
import math
import numpy as np

##-------------------- CONSTANTS --------------------------

# mec2 = 9.10938291e-31*(299792458*299792458)*6.24150934e15 #me*c^2 (keV)
mec2 = 510.9989273362467  # equivalent to above, but doesn't give errors
Ecomplim = 340.66  # kev, the max the first compton can be
percabove = 1.08  # percentage above that is still allowable


##-------------------- GENERAL --------------------------

# cos(thetaLOR) = dotproduct(A,B)/(norm(A)*norm(B))
def calc_thetaLOR_3D(x0, y0, z0, x1, y1, z1, x2, y2, z2):
    vec1 = [x1 - x0, y1 - y0, z1 - z0]
    vec2 = [x2 - x1, y2 - y1, z2 - z1]
    norm1 = math.sqrt(vec1[0] * vec1[0] + vec1[1] * vec1[1] + vec1[2] * vec1[2])
    norm2 = math.sqrt(vec2[0] * vec2[0] + vec2[1] * vec2[1] + vec2[2] * vec2[2])
    if norm1 == 0 or norm2 == 0:
        return -1000000
    thetaLOR = math.acos((vec1[0] * vec2[0] + vec1[1] * vec2[1] + vec1[2] * vec2[2]) / (norm1 * norm2))
    return thetaLOR


# thetaE
# E1 = 511 #=mec2 #energy of initial photon
# Edep = energy deposited in first compton interaction (E2=E1-Edep)
def calc_thetaE(E1, Edep):
    # if the energy is big, but within limit, set to maximum
    if Edep > Ecomplim:
        E2 = E1 - Ecomplim
    else:
        E2 = E1 - Edep
    thetaE = math.acos(1.0 - mec2 * (1.0 / E2 - 1.0 / E1))
    return thetaE


# find if it was C or P depending on the number of entries
def C_or_P(entries):
    if len(entries) == 1:
        return 'P'
    if len(entries) == 2:
        return 'C'
    # if neither C or P return N
    return 'N'


# check if the energy is above the threshold (check all thresholds)
def check_Ethresh(Ethresh, energy, E, index, cut_list):
    if energy < Ethresh:  # if it's below threshold, mark that it was cut
        cut_list.append(index + 1)
        return 0
    else:
        return energy


##-------------------- SPECIFIC FOR GATE SIMULATION --------------------------

# Hits o/p:
# col0: ID of run (timeslice)
# col1: ID of event PHOTON
# col2: ID of primary particle
# col3: ID of source which emitted primary
# col4 to 6+4(10): volume IDs
# col4: ID of volume for base level
# col5: ID of rsector
# col6: ID of module in hamun crystal has!
# col7: ID of submodule tu czt submodule nadarim!
# col8: ID of crystal! Dr shiva said it is Zero!
# col9: ID of layer
# col10: time stamp
# col11: energy deposited
# col12: range of particle
# col13-15: x,y,z inja 13-14-15 has na az 14 nana
# col16: geant4 code
# col17 ID of particle
# col18: ID of mother of particle
# col19: ID of photon?
# col20: number of compton in phantom before reaching detector
# col21: number of rayleigh in phantom before reaching detector
# col22: name of process
# col23: name of last volume where compton occurs
# col24: name of last volume where rayleigh occurs

# the Hits.dat file contains many spaces, making it so you can't use .split(' ')
# instead you can use this function to parse the line, it looks for all
# combinations of spaces between columns
# input: string line from Hits.dat file
# output: parsed line (array of columns)
def parse_GATE_Hits_line(line):
    # get rid of the spaces in the line, and replace them with commas
    line = line.replace('       ', ',')
    line = line.replace('      ', ',')
    line = line.replace('     ', ',')
    line = line.replace('    ', ',')
    line = line.replace('   ', ',')
    line = line.replace('  ', ',')
    line = line.replace(' ', ',')
    l = line.split(',')[1:]
    return l


def cut_Ethresh(Ethresh, group):
    groupEt = []
    for i, l in enumerate(group):
        if float(l[11]) * 1000. > Ethresh:
            groupEt.append(l)
    return groupEt


# find if any of the lines have: annihil,ElectronIonisation,Transportation,RayleighScattering
# we don't want to process those
# return 'False' if there is a bad line, return 'True' otherwise
def find_bad_lines(ph, counter={}):
    # the column for type depends on data format
    if len(ph[0]) > 9:  # > 8 ###changed 07-10-2016 to allow for adding the full line to group
        col = 22
    else:
        col = 6

    for l in ph:
        if l[col] == 'annihil' or l[col] == 'ElectronIonisation' or l[col] == 'Transportation' or l[
            col] == 'RayleighScattering':
            if 'annihil' in counter:
                counter[l[col]] = counter[l[col]] + 1
            return True  # say Yes, there was a bad line

    return False  # if no bad lines, return False


# find the total energy of a photon
def find_total_energy(ph, Et=0):
    energy = 0
    # sum the energy of all the depositions
    for l in ph:
        en = float(l[11]) * 1000.  # x1000 because it was in MeV
        if en > Et:
            energy += en

    energy = energy
    return energy


# this function assumes 4 panel geometry  majid check again
# input: panel # from line of GATE Hits.dat file
def get_panel_center4(panel):
    panel = int(float(panel))                # convert to int
    if panel == 0:
        return [120, -40, 0]                     # nana: [x, y, z] return [60, -21, 0] check

    elif panel == 1:
        return [-120, 40, 0]                #return [-60, 21, 0] check
    else:
        print('Wrong panel type')
        return [False, False, False]


# build a table to know which anode and cathode an interaction is from, it is
# essentially a decoder
# it is based on C+A*8 = ID in GATE
# also find the offset (assuming panel 0)
# output = [A,C, x,y] where x,y are offsets assuming panel 0
def build_anode_cathode_table():
    actable = {}
    for i in range(8):  # 8 ta cathode darim
        for j in range(39):  # 39 ta anode
            x, y = get_ac_offset_position([j, i])
            actable[str(i + j * 8)] = [j, i, x, y]  # Output = [A,C, x,y]
    return actable


# build table to know which module it is (in z and laterally)
# and find the offset assuming panel 0

def build_module_table():
    mtable = {}
    for i in range(5):                                          # lateral 5 ta 4 cm darim:) nana
        for j in range(30):                                             # height 30 * 0.5  = 15 cm
            y, z = get_module_offset_position([j, i])
            mtable[str(i + j * 5)] = [j, i, y, z]                            # =[height,lateral, y,z]
    return mtable


# get the offset due to Anode/Cathode w.r.t. center of crystal
# input = [A,C]
# output = [x,y] (assumes panel 0)
def get_ac_offset_position(ac):  # done:) Dr.
    # the cathode is 5mm and goes to right to the edge of the crystal
    cathode = 5.
    c_offset = 0. - 40. / 2.  # half the crystal
    # anode is 1mm and there are 39 (in a 4cm crystal) so it doesn't hit the
    # edge, there is an offset of 0.5mm on each side
    anode = 1.
    a_offset = 0.5 - 40. / 2.

    x = (cathode / 2.) + cathode * ac[1] + c_offset
    y = (anode / 2.) + anode * ac[0] + a_offset
    return [x, y]


# get the offset due to module, assuming panel 0
# input = [height,lateral]
# output = [y,z] (these are the offsets from the center of panel to center of
# module)
'''  ino az awal neveshtim.
def get_module_offset_position(m):#??? DR ino neveshte bud man payini.
	#z = 0 is center of panel, so the offset is either above or below 0
	zmodule = 10.1 #mm ma 5mm nana
	#there are 3 modules, so the center one is exactly at center of panel
	ymodule = 40.01 #mm ma 40 mm nana

	if m[0] > 3: #above 0mm in z Inja
		z = zmodule/2. + (m[0] - 4)*zmodule
	else: #below 0
		z = -zmodule/2. - (3-m[0])*zmodule

	y = (m[1] - 1)*ymodule

	return [y,z]
'''


def get_module_offset_position(m):  # nana ghalat
    # z = 0 is center of panel, so the offset is either above or below 0
    zmodule = 5.0 # 10.01 bud, 5mm mishe chon 0.5 has Z
    # there are 3 modules, so the center one is exactly at center of panel
    ymodule = 40.0

    if m[0] >= 5:  # above 0mm in z Inja ghalat 15 bud
        z = zmodule / 2 + (m[0] - 6) * zmodule

    else:  # below 0
        z = -zmodule / 2 - (6 - m[0]) * zmodule
    y = (m[1] - 1) * ymodule  # m[1]=i
    return [y, z]


# get the offset in z of the cyrstal, assume you know the center of the module
def get_crystal_offset_position(cr):
    zcrystal =  5 # 5.01  # mm
    z = 0   # because we only have one submodule in each module

    # if cr == 0:
    #     z = -zcrystal / 2.
    # elif cr == 1:
    #     z = zcrystal / 2.
    # else:
    #     print('What is crystal? ', cr)
    return z


# return the x,y,z values from a linesplit GATE line
# default separator is a space
def return_xyz(ph, separator=' '):
    return ph[13] + separator + ph[14] + separator + ph[15]


# return the distance from center, use l[5] to get node number
# the distance is calculated in one dimension (normal to crystal), which depends on which node
def get_dist_from_center(ph):  # check
    node = ph[5]
    # print node
    print(ph[13],ph[14],ph[15])
    if node == '0' or node == '1':
        # node 0, +ve x, node 2 -ve x
        dist = abs(float(ph[13]))

    elif node == '2' or node == '3':
        # node 1, +ve y, node 3 -ve y
        dist = abs(float(ph[14]))

    else:

        print('Node number does not make sense', node)

    # return the absolute distance from center
    print(100*'---')
    print('dist,:', dist)
    return dist

# same as above, but takes len(ph)=8
def get_process_dist_from_center(ph):  # check
    node = ph[7]
    if node == 0 or node == 2:
        # node 0, +ve x, node 2 -ve x
        dist = abs(float(ph[3]))
    elif node == 1 or node == 3:
        # node 1, +ve y, node 3 -ve y
        dist = abs(float(ph[4]))
    else:
        print('Node number does not make sense', node)
    # return the absolute distance from center
    return dist


# get the thetaLOR for gate simulation data Hits file
# depA is from one annihilation photon, depB and depC are from the other
def calc_thetaLOR_3D_GATE(depA, depB, depC):
    thetaLOR = calc_thetaLOR_3D(float(depA[13]), float(depA[14]), float(depA[15]), float(depB[13]), float(depB[14]),
                                float(depB[15]), float(depC[13]), float(depC[14]), float(depC[15]))
    return thetaLOR


# using the x,y,z positions of three energy deposition points, find theta_LOR again nana check
# point0 is in one node, point1 and point2 are on the other node (point2 is last energy deposition)
# cos(theta_LOR) = dotproduct(A,B)/(norm(A)*norm(B))
# data from GATE
# def calc_thetaLOR(x0,y0,z0,x1,y1,z1,x2,y2,z2):
def calc_thetaLOR(point0, point1, point2):
    # first get the x, y, z points
    x0 = point0[3]
    y0 = point0[4]
    z0 = point0[5]
    x1 = point1[3]
    y1 = point1[4]
    z1 = point1[5]
    x2 = point2[3]
    y2 = point2[4]
    z2 = point2[5]
    # get the vectors A and B
    vecA = [x1 - x0, y1 - y0, z1 - z0]
    vecB = [x2 - x1, y2 - y1, z2 - z1]
    normA = math.sqrt(vecA[0] * vecA[0] + vecA[1] * vecA[1] + vecA[2] * vecA[2])
    normB = math.sqrt(vecB[0] * vecB[0] + vecB[1] * vecB[1] + vecB[2] * vecB[2])
    if normA == 0 or normB == 0:
        return -1000000
    try:
        theta_LOR = math.acos((vecA[0] * vecB[0] + vecA[1] * vecB[1] + vecA[2] * vecB[2]) / (normA * normB))
    except:
        theta_LOR = -1000003
    return theta_LOR * 180. / 3.14159


# return 0 if all from the same positron check again
# if dtype (data type) == embed, that means that the full line is embedded in group[x][8]
def check_same_Positron(group1, group2, dtype='', ID=False):
    if dtype == 'embed':
        pos = group1[0][8][1]
    else:
        pos = group1[0][1]  # pick first one to compare against

    return_val = 0  # =0 if all from same positron eventID
    photID1 = []
    for l in group1:
        if dtype == 'embed':
            l = l[8]
        photID1.append(l[17])
        if l[1] != pos:
            return_val = 1
            break
    photID2 = []
    for l in group2:
        if dtype == 'embed':
            l = l[8]
        photID2.append(l[17])
        if l[1] != pos:
            return_val = 1
            break
    # the photID should be all the same within the same group, but differ from
    # the other group
    iphotID = 0
    # check within grp1
    for i in photID1:
        for j in photID1:
            if i != j:
                iphotID = 1
                break
    # check within grp2
    for i in photID2:
        for j in photID2:
            if i != j:
                iphotID = 1
                break
    # check between grp1 & grp2
    for i in photID1:
        for j in photID2:
            if i == j:
                iphotID = 1
                break

    if ID == False:
        return return_val
    else:
        if return_val == 1:
            iphotID = -1  # if they are different eventID, doesn't matter what the photon # is
        return return_val, iphotID


# return 1 if there was scatter in the phantom, return 0 if otherwise
# l[20] = # of Compton in phantom
# l[21] = # of Rayleigh in phantom
# if dtype (data type) == embed, that means that the full line is embedded in group[x][8]
def check_phantom_scatter(group1, group2, dtype=''):
    # check both groups (from both nodes)
    for l in group1:
        if dtype == 'embed':
            # print l[8]
            l = l[8]
        if int(l[20]) > 0 or int(l[21]) > 0:
            return 1  # scatter
    for l in group2:
        if dtype == 'embed':
            l = l[8]
        if int(l[20]) > 0 or int(l[21]) > 0:
            return 1  # scatter
    return 0  # no scatter


# change this slightly to provide a value indicative of where the scatter occured
def check_phantom_scatter_mod(group1, group2, dtype=''):
    # check both groups (from both nodes)
    returnval = 0
    for l in group1:
        if dtype == 'embed':
            # print l[8]
            l = l[8]
        if int(l[20]) > 0:  # FLAG COMPTON ONLY! # or int(l[21]) > 0:
            # return 1 #scatter
            returnval += 1  # scatter
            break
    for l in group2:
        if dtype == 'embed':
            l = l[8]
        if int(l[20]) > 0:  # FLAG COMPTON ONLY! # or int(l[21]) > 0:
            # return 1 #scatter
            returnval += 2  # scatter
            break
    return returnval  # no scatter


# return False if this group has 2 nodes (good case), return True otherwise (bad case)
# also return the indices of N1 and N2
# def check_for_nodes2(group):
#	N1_index = []
#	N2_index = []
#	for i,line in enumerate(group):
#		if line[7] == 1:
#			N1_index.append(i)
#		elif line[7] == 2:
#			N2_index.append(i)
#	if len(N1_index)>0 and len(N2_index)>0:
#		return False, N1_index, N2_index
#	else:
#		return True, N1_index, N2_index

# same as above but for 4 node system check again nana
# for False case, return the two separated groups
# for True case, don't return indeces of nodes
# maxPerNode = maximum number of interactions per node that are above the
# threshold
# retall = True if you want to return all interactions, even those below threshold, =False if you only want to return interactions above Et
# moreinfo = true if you want xyz info of first detected event (which may have
# been removed because threshold)
def check_for_nodes4(group, group_real, Et=0, maxPerNode=2, retall=True, moreinfo=False, failtype={},
                     pickhighest2=False):
    # check what type of format it is
    # if len(group[0]) > 8:
    if len(group[0]) > 9:  # changed 07-10-2016 to allow for adding the full line to group
        Ncol = 5
        Ecol = 11
    else:
        Ncol = 7
        Ecol = 2
    N_indices = [[], [], [], []]
    N_indices_above_Et = [[], [], [], []]
    E_above_Et = [0, 0, 0, 0]
    N_indices_dropped = [[], [], [], []]  # keep track of the dropped ones too
    # go over all interactions in the group
    for i, line in enumerate(group):
        # n = Node
        n = int(line[Ncol])  # get_node(float(line[13]),float(line[14]))
        # add this index to the appropriate node
        N_indices[n].append(i)
        # get the energy
        if Ecol == 2:
            Eval = line[Ecol]
        else:
            Eval = float(line[11]) * 1000.
        if Eval > Et:
            N_indices_above_Et[n].append(i)
            E_above_Et[n] += Eval
        else:
            N_indices_dropped[n].append(i)

    # check if there are only events on two of the nodes
    num_nodes = []

    # only look at interactions above energy threshold
    for i, val in enumerate(N_indices_above_Et):
        # if len(N_indices_above_Et[i]) > 0:
        # if len(N_indices_above_Et[i]) > 0 and len(N_indices_above_Et[i]) <= maxPerNode:
        # changed Jul-09-2016 to prevent data when 3+ nodes are triggered and have more than maxPerNode
        if len(N_indices_above_Et[i]) > 0:
            if len(N_indices_above_Et[i]) <= maxPerNode:
                num_nodes.append(i)
            else:  # this is when there is at least one node that has more than maxPerNode triggered within the time window and above Et

                # this code hasn't gone through rigourous testing
                if pickhighest2 == True:
                    if Ecol != 2:
                        # doesnt support this yet
                        exit()
                    # only keep the 2 highest energy interactions
                    thekeepers = [N_indices_above_Et[i][0], N_indices_above_Et[i][1]]
                    for jj, jjval in enumerate(N_indices_above_Et[i]):
                        # skip the first 2 because these are initially assumed to be the maxE
                        if jj == 0 or jj == 1:
                            continue
                        replace = 0
                        if group[thekeepers[0]][Ecol] > group[thekeepers[1]][Ecol]:
                            replace = 1
                        # if this one is biggest than the smallest keeper
                        if group[jj][Ecol] > group[thekeepers[replace]][Ecol]:
                            # note that this event is now being removed
                            N_indices_dropped[i].append(int(thekeepers[replace]))
                            thekeepers[replace] = jjval
                        else:
                            # if this one is small, need to remove it
                            N_indices_dropped[i].append(jjval)
                    # only keep the remaining interactions
                    N_indices_above_Et[i] = thekeepers
                    num_nodes.append(i)

                else:
                    failtype['more_than_maxPerNode'] += 1
                    if len(N_indices_above_Et[i]) == 3:
                        failtype['more_than_maxPerNode-3interactions'] += 1
                    if moreinfo == True:
                        return True, -1, -1, -1, -1, -1, -1, -1, -1
                    else:
                        return True, -1, -1, -1, -1
    # if there are more than 2 nodes triggered, try to find the best match to
    # reach 511keV, put the low energy nodes to the higher energy one(s) and
    # delete the low energy nodes to maintain 2 nodes
    if maxPerNode > 2 and len(num_nodes) > 2:

        # find the indices from smallest to largest energy
        index_order = [i[0] for i in sorted(enumerate(E_above_Et), key=lambda x: x[1])]

        # only keep indices of nodes that are considered (these nodes should have E>Et)
        E_indices = []
        for i in index_order:
            if i in num_nodes:
                E_indices.append(i)

        if len(num_nodes) == 3:
            # if there are 3, add the two smallest together
            if Ecol == 2:
                group[N_indices_above_Et[E_indices[1]][0]][Ecol] += E_above_Et[E_indices[0]]
            else:
                group[N_indices_above_Et[E_indices[1]][0]][Ecol] = str(
                    E_above_Et[E_indices[0]] / 1000. + float(group[N_indices_above_Et[E_indices[1]][0]][Ecol]) / 1000.)
            new_num_nodes = []
            for i in num_nodes:
                if i == E_indices[1] or i == E_indices[2]:  # the 2 highest
                    new_num_nodes.append(i)

        elif len(num_nodes) == 4:
            # if there are 4, add the smallest together and try to keep < 511
            if Ecol == 2:
                group[N_indices_above_Et[E_indices[3]][0]][Ecol] += E_above_Et[E_indices[0]]
                group[N_indices_above_Et[E_indices[2]][0]][Ecol] += E_above_Et[E_indices[1]]
            else:
                group[N_indices_above_Et[E_indices[3]][0]][Ecol] = str(
                    E_above_Et[E_indices[0]] / 1000. + float(group[N_indices_above_Et[E_indices[3]][0]][Ecol]) / 1000.)
                group[N_indices_above_Et[E_indices[2]][0]][Ecol] = str(
                    E_above_Et[E_indices[1]] / 1000. + float(group[N_indices_above_Et[E_indices[2]][0]][Ecol]) / 1000.)
            new_num_nodes = []
            for i in num_nodes:
                if i == E_indices[3] or i == E_indices[2]:  # the 2 highest
                    new_num_nodes.append(i)

        num_nodes = new_num_nodes
    # print len(num_nodes)

    if len(num_nodes) != 2:
        if len(num_nodes) < 2:
            failtype['less_than_2Node'] += 1
        else:
            failtype['more_than_2Node'] += 1
        if moreinfo == True:
            return True, -1, -1, -1, -1, -1, -1, -1, -1
        else:
            return True, -1, -1, -1, -1
    else:

        # find out (1) what the xyz was of the first event, and if the first
        # events were dropped
        if moreinfo == True:
            # they will be in the correct order due to exact timing info
            xyz1 = return_xyz(group[N_indices[num_nodes[0]][0]][8])
            xyz2 = return_xyz(group[N_indices[num_nodes[1]][0]][8])
            dropped_bef1 = 0
            dropped_bef2 = 0
            # if there was a dropped line before the last one taken into account, then record that
            if len(N_indices_dropped[num_nodes[0]]) > 0:
                for j in N_indices_dropped[num_nodes[0]]:
                    if j < N_indices_above_Et[num_nodes[0]][-1]:
                        dropped_bef1 = 1
            if len(N_indices_dropped[num_nodes[1]]) > 0:
                for j in N_indices_dropped[num_nodes[1]]:
                    if j < N_indices_above_Et[num_nodes[1]][-1]:
                        dropped_bef2 = 1
        N1 = []
        N2 = []
        N1_real = []
        N2_real = []
        # OLD WAY--->add all lines, even those below thresh
        if retall == True:
            for i, val in enumerate(N_indices[num_nodes[0]]):
                N1.append(group[val])
                N1_real.append(group_real[val])
            for i, val in enumerate(N_indices[num_nodes[1]]):
                N2.append(group[val])
                N2_real.append(group_real[val])
        # only add lines above threshold
        else:
            for i, val in enumerate(N_indices_above_Et[num_nodes[0]]):
                N1.append(group[val])
                N1_real.append(group_real[val])
            for i, val in enumerate(N_indices_above_Et[num_nodes[1]]):
                N2.append(group[val])
                N2_real.append(group_real[val])
        # print len(N1[0]),len(N2[0]),len(N1_real[0]),len(N2_real[0])
        # print len(N1),len(N2),len(N1_real),len(N2_real)
        if moreinfo == True:
            return False, N1, N2, N1_real, N2_real, xyz1, dropped_bef1, xyz2, dropped_bef2
        else:
            return False, N1, N2, N1_real, N2_real


# return the node number based on the x,y,z position
# def get_node4(x,y):
#	if x < 0: #node 1
#		return 1
#	else: #node 2
#		return 2

# check if the events are within the energy limits
# ADDED 04-13-2017
# sumAll = True if you want to sum the energy from all potential events, not just the first 2
# This is generally desired for counting events that fall in the energy window.
# This is also acceptable if maxPerNode is being used to gate (i.e. determine) how many events should be considered
def check_for_Elimits(N1_index, N2_index, Eminflag, EminCompton, Elimlo, Elimhi,
                      etype={'lessthanEwin': 0, 'morethanEwin': 0, 'lessthanECom': 0}, sumAll=True):  # check N1 first
    if len(N1_index) == 1:  # PE event
        if N1_index[0][2] > Elimhi or N1_index[0][2] < Elimlo:
            if N1_index[0][2] > Elimhi:
                etype['morethanEwin'] += 1
            else:
                etype['lessthanEwin'] += 1
            return 1  # not within limits
    else:  # compton
        if sumAll == True:
            Etotal = 0
            for i in N1_index:
                Etotal += i[2]
        else:  # just add energy from first 2 events
            Etotal = N1_index[0][2] + N1_index[1][2]
        if Etotal > Elimhi or Etotal < Elimlo:
            if Etotal > Elimhi:
                etype['morethanEwin'] += 1
            else:
                etype['lessthanEwin'] += 1
            return 1
        if Eminflag == True:  # check for minimum Compton
            if N1_index[0][2] < EminCompton or N1_index[1][2] < EminCompton:
                etype['lessthanECom'] += 1
                return 1

    # check N2
    if len(N2_index) == 1:  # PE event
        if N2_index[0][2] > Elimhi or N2_index[0][2] < Elimlo:
            if N2_index[0][2] > Elimhi:
                etype['morethanEwin'] += 1
            else:
                etype['lessthanEwin'] += 1
            return 1  # not within limits
    else:  # compton
        if sumAll == True:
            Etotal = 0
            for i in N2_index:
                Etotal += i[2]
        else:  # just add energy from first 2 events
            Etotal = N2_index[0][2] + N2_index[1][2]
        if Etotal > Elimhi or Etotal < Elimlo:
            if Etotal > Elimhi:
                etype['morethanEwin'] += 1
            else:
                etype['lessthanEwin'] += 1
            return 1
        if Eminflag == True:  # check for minimum Compton
            if N2_index[0][2] < EminCompton or N2_index[1][2] < EminCompton:
                etype['lessthanECom'] += 1
                return 1
    # if everything looks good, return 0
    return 0


'''
find the x,y,z that are closest to the center of the system and choose those
points for the first compton interaction
'''


def find_closest_GATE(ph2, ph3, itype):
    if itype == 'CP':
        xyz3 = return_xyz(ph3[0])  # get the PE first point
        dist1 = get_dist_from_center(ph2[0])
        dist2 = get_dist_from_center(ph2[1])
        if dist1 < dist2:  # pick one closest to center
            xyz2 = return_xyz(ph2[0])
        else:
            xyz2 = return_xyz(ph2[1])
    elif itype == 'PC':
        xyz2 = return_xyz(ph2[0])  # get the PE first point
        dist1 = get_dist_from_center(ph3[0])
        dist2 = get_dist_from_center(ph3[1])
        if dist1 < dist2:  # pick one closest to center
            xyz3 = return_xyz(ph3[0])
        else:
            xyz3 = return_xyz(ph3[1])
    elif itype == 'CC':
        dist1 = get_dist_from_center(ph2[0])
        dist2 = get_dist_from_center(ph2[1])
        dist3 = get_dist_from_center(ph3[0])
        dist4 = get_dist_from_center(ph3[1])
        if dist1 < dist2:  # pick one closest to center
            xyz2 = return_xyz(ph2[0])
        else:
            xyz2 = return_xyz(ph2[1])
        if dist3 < dist4:  # pick one closest to center
            xyz3 = return_xyz(ph3[0])
        else:
            xyz3 = return_xyz(ph3[1])

    return [xyz2, xyz3]


'''
find the x,y,z that are closest to the center of the system and choose those
points for the first compton interaction
Return the new reordered group
'''


def process_Compton_closest(ph2, ph3, itype):
    # NOTE: regardless of CP or PC the order of group is always PCC
    if itype == 'CP':
        # xyz3 = return_xyz(ph3[0]) #get the PE first point
        dist1 = get_process_dist_from_center(ph2[0])
        dist2 = get_process_dist_from_center(ph2[1])
        if dist1 < dist2:  # pick one closest to center
            # xyz2 = return_xyz(ph2[0])
            # min_grp = [ph2[0], ph2[1], ph3[0]]
            min_grp = [ph3[0], ph2[0], ph2[1]]
        else:
            # xyz2 = return_xyz(ph2[1])
            # min_grp = [ph2[1], ph2[0], ph3[0]]
            min_grp = [ph3[0], ph2[1], ph2[0]]
    elif itype == 'PC':
        # xyz2 = return_xyz(ph2[0]) #get the PE first point
        dist1 = get_process_dist_from_center(ph3[0])
        dist2 = get_process_dist_from_center(ph3[1])
        if dist1 < dist2:  # pick one closest to center
            # xyz3 = return_xyz(ph3[0])
            min_grp = [ph2[0], ph3[0], ph3[1]]
        else:
            # xyz3 = return_xyz(ph3[1])
            min_grp = [ph2[0], ph3[1], ph3[0]]
    elif itype == 'CC':
        dist1 = get_process_dist_from_center(ph2[0])
        dist2 = get_process_dist_from_center(ph2[1])
        dist3 = get_process_dist_from_center(ph3[0])
        dist4 = get_process_dist_from_center(ph3[1])
        min_grp = [[], [], [], []]
        if dist1 < dist2:  # pick one closest to center
            # xyz2 = return_xyz(ph2[0])
            min_grp[0] = ph2[0]
            min_grp[1] = ph2[1]
        else:
            # xyz2 = return_xyz(ph2[1])
            min_grp[0] = ph2[1]
            min_grp[1] = ph2[0]
        if dist3 < dist4:  # pick one closest to center
            # xyz3 = return_xyz(ph3[0])
            min_grp[2] = ph3[0]
            min_grp[3] = ph3[1]
        else:
            # xyz3 = return_xyz(ph3[1])
            min_grp[2] = ph3[1]
            min_grp[3] = ph3[0]

    return min_grp


'''
choose the first Compton interaction randomly. Ignore ambiguity (which would bring the % correct from 50% down to 25%)
Return the new reordered group.
correct_order = [0,0] #0 means correct ordering (based off exactly known
        # timing, and thus order in the group, (ordering will match returned order) 
'''


def process_Compton_random(ph2, ph3, itype, correct_order):
    # NOTE: regardless of CP or PC the order of group is always PCC
    if itype == 'CP':
        firstint = np.random.randint(0, 2)
        if firstint != 0:
            correct_order = [1, 0]
        min_grp = [ph3[0], ph2[firstint], ph2[(firstint + 1) % 2]]
    elif itype == 'PC':
        firstint = np.random.randint(0, 2)
        if firstint != 0:
            correct_order = [0, 1]
        min_grp = [ph2[0], ph3[firstint], ph3[(firstint + 1) % 2]]
    elif itype == 'CC':
        firstint1 = np.random.randint(0, 2)
        if firstint1 != 0:
            correct_order[0] = 1
        firstint2 = np.random.randint(0, 2)
        if firstint2 != 0:
            correct_order[1] = 1
        min_grp = [ph2[firstint1], ph2[(firstint1 + 1) % 2], ph3[firstint2], ph3[(firstint2 + 1) % 2]]
    return min_grp, correct_order


# get the energy-weighted x,y,z and return points while replacing the first interaction's
# (x,y,z) position based on the calculated value
def return_Eweight_xyz(point1, point2):
    x1 = float(point1[3]);
    y1 = float(point1[4]);
    z1 = float(point1[5])
    x2 = float(point2[3]);
    y2 = float(point2[4]);
    z2 = float(point2[5])
    E1 = float(point1[2]);
    E2 = float(point2[2])

    xnew = (x1 * E1 + x2 * E2) / (E1 + E2)
    ynew = (y1 * E1 + y2 * E2) / (E1 + E2)
    znew = (z1 * E1 + z2 * E2) / (E1 + E2)

    ph1 = point1
    ph1[3] = str(xnew)
    ph1[4] = str(ynew)
    ph1[5] = str(znew)
    ph2 = point2

    return ph1, ph2


'''
find the x,y,z based on energy weighting method
Return the new group with updated (x,y,z) position (only for the first
interactions)
'''


def process_Compton_Eweight(ph2, ph3, itype):
    # NOTE: regardless of CP or PC the order of group is always PCC
    if itype == 'CP':
        p1, p2 = return_Eweight_xyz(ph2[0], ph2[1])  # get the energy weight position
        # min_grp = [p1, p2, ph3[0]]
        min_grp = [ph3[0], p1, p2]
    elif itype == 'PC':
        p1, p2 = return_Eweight_xyz(ph3[0], ph3[1])  # get the energy weight position
        min_grp = [ph2[0], p1, p2]
    elif itype == 'CC':
        p1, p2 = return_Eweight_xyz(ph2[0], ph2[1])  # get the energy weight position
        p3, p4 = return_Eweight_xyz(ph3[0], ph3[1])  # get the energy weight position
        min_grp = [p1, p2, p3, p4]

    return min_grp


# get the theta difference
# ph2 = lines for one annihilation photon, ph3 = lines from the other
# itype == 'CP', 'PC', or 'CC',
def find_thetadiff_GATE(ph2, ph3, itype):
    min_theta_diff = 100000
    min_theta_diff2 = 100000  # only used for CC
    E1 = 511  # =mec2 #energy of initial photon
    xyz2 = '   '
    xyz3 = '   '

    # check CP events
    if itype == 'CP':
        xyz3 = return_xyz(ph3[0])  # get the PE first point
        # try the first way (this should always be true since it happens first in time)
        Edep2 = float(ph2[0][11]) * 1000.  # energy of the 1st compton in keV
        if Edep2 < Ecomplim * percabove:
            thetaE = calc_thetaE(E1, Edep2)
            # use xN2,yN2, xN1-1,yN1-1, xN1-2,yN1-2
            thetaLOR = calc_thetaLOR_3D_GATE(ph3[0], ph2[0], ph2[1])
            theta_diff = abs(thetaE - thetaLOR)
            min_theta_diff = theta_diff
            xyz2 = return_xyz(ph2[0])

        # try the other way
        Edep2 = float(ph2[1][11]) * 1000.
        if Edep2 < Ecomplim * percabove:
            thetaE = calc_thetaE(E1, Edep2)
            # use xN2,yN2, xN1-2,yN1-2, xN1-1,yN1-1
            thetaLOR = calc_thetaLOR_3D_GATE(ph3[0], ph2[1], ph2[0])
            theta_diff = abs(thetaE - thetaLOR)
            if theta_diff < min_theta_diff:
                min_theta_diff = theta_diff
                xyz2 = return_xyz(ph2[1])
        # return theta_diff in degrees and xyz positions
        return [min_theta_diff * 180. / 3.14159, 0], [xyz2, xyz3]

    elif itype == 'PC':
        xyz2 = return_xyz(ph2[0])  # get the PE first point
        # try the first way (this should always be true since it happens first in time)
        Edep3 = float(ph3[0][11]) * 1000.
        if Edep3 < Ecomplim * percabove:
            thetaE = calc_thetaE(E1, Edep3)
            # use xN2,yN2, xN1-1,yN1-1, xN1-2,yN1-2
            thetaLOR = calc_thetaLOR_3D_GATE(ph2[0], ph3[0], ph3[1])
            theta_diff = abs(thetaE - thetaLOR)
            min_theta_diff = theta_diff
            xyz3 = return_xyz(ph3[0])

        # trying the other way
        Edep3 = float(ph3[1][11]) * 1000.
        if Edep3 < Ecomplim * percabove:
            thetaE = calc_thetaE(E1, Edep3)
            # use xN2,yN2, xN1-1,yN1-1, xN1-2,yN1-2
            thetaLOR = calc_thetaLOR_3D_GATE(ph2[0], ph3[1], ph3[0])
            theta_diff = abs(thetaE - thetaLOR)
            if theta_diff < min_theta_diff:
                min_theta_diff = theta_diff
                xyz3 = return_xyz(ph3[1])
        # return theta_diff in degrees
        return [0, min_theta_diff * 180. / 3.14159], [xyz2, xyz3]

    elif itype == 'CC':
        sum_theta_diff_scenarios = [1e6, 1e6, 1e6, 1e6]
        sum_theta_diff_1 = [1e6, 1e6, 1e6, 1e6]
        sum_theta_diff_2 = [1e6, 1e6, 1e6, 1e6]
        xyz_scenarios = [['', ''], ['', ''], ['', ''], ['', '']]
        # try the first way, N2-1, N2-2  and N1-1, N1-2
        Edep3 = float(ph3[0][11]) * 1000.
        if Edep3 < Ecomplim * percabove:
            thetaE = calc_thetaE(E1, Edep3)
            Edep2 = float(ph2[0][11]) * 1000.
            if Edep2 < Ecomplim * percabove:
                thetaLOR = calc_thetaLOR_3D_GATE(ph2[0], ph3[0], ph3[1])
                theta_diff = abs(thetaE - thetaLOR)
                min_theta_diff = theta_diff
                # find the theta diff for the other photon interaction
                thetaE2 = calc_thetaE(E1, Edep2)
                thetaLOR2 = calc_thetaLOR_3D_GATE(ph3[0], ph2[0], ph2[1])
                theta_diff2 = abs(thetaE2 - thetaLOR2)
                min_theta_diff2 = theta_diff2
                xyz2 = return_xyz(ph2[0])
                xyz3 = return_xyz(ph3[0])
                sum_theta_diff_1[0] = theta_diff
                sum_theta_diff_2[0] = theta_diff2
                sum_theta_diff_scenarios[0] = theta_diff + theta_diff2
                xyz_scenarios[0] = [xyz2, xyz3]
            Edep2 = float(ph2[1][11]) * 1000.
            if Edep2 < Ecomplim * percabove:
                thetaLOR = calc_thetaLOR_3D_GATE(ph2[1], ph3[0], ph3[1])
                theta_diff = abs(thetaE - thetaLOR)
                if theta_diff < min_theta_diff:
                    min_theta_diff = theta_diff
                    xyz2 = return_xyz(ph2[1])
                    xyz3 = return_xyz(ph3[0])
                # find the theta diff for the other photon interaction
                thetaE2 = calc_thetaE(E1, Edep2)
                thetaLOR2 = calc_thetaLOR_3D_GATE(ph3[0], ph2[1], ph2[0])
                theta_diff2 = abs(thetaE2 - thetaLOR2)
                if theta_diff2 < min_theta_diff2:
                    min_theta_diff2 = theta_diff2
                    xyz2 = return_xyz(ph2[1])
                    xyz3 = return_xyz(ph3[0])
                xyz2 = return_xyz(ph2[1])
                xyz3 = return_xyz(ph3[0])
                sum_theta_diff_1[1] = theta_diff
                sum_theta_diff_2[1] = theta_diff2
                sum_theta_diff_scenarios[1] = theta_diff + theta_diff2
                xyz_scenarios[1] = [xyz2, xyz3]

        # trying the other way
        Edep3 = float(ph3[1][11]) * 1000.
        if Edep3 < Ecomplim * percabove:
            thetaE = calc_thetaE(E1, Edep3)
            Edep2 = float(ph2[0][11]) * 1000.
            if Edep2 < Ecomplim * percabove:
                # use xN2,yN2, xN1-1,yN1-1, xN1-2,yN1-2
                thetaLOR = calc_thetaLOR_3D_GATE(ph2[0], ph3[1], ph3[0])
                theta_diff = abs(thetaE - thetaLOR)
                if theta_diff < min_theta_diff:
                    min_theta_diff = theta_diff
                    xyz2 = return_xyz(ph2[0])
                    xyz3 = return_xyz(ph3[1])
                # find the theta diff for the other photon interaction
                thetaE2 = calc_thetaE(E1, Edep2)
                thetaLOR2 = calc_thetaLOR_3D_GATE(ph3[1], ph2[0], ph2[1])
                theta_diff2 = abs(thetaE2 - thetaLOR2)
                if theta_diff2 < min_theta_diff2:
                    min_theta_diff2 = theta_diff2
                    xyz2 = return_xyz(ph2[0])
                    xyz3 = return_xyz(ph3[1])
                xyz2 = return_xyz(ph2[0])
                xyz3 = return_xyz(ph3[1])
                sum_theta_diff_1[2] = theta_diff
                sum_theta_diff_2[2] = theta_diff2
                sum_theta_diff_scenarios[2] = theta_diff + theta_diff2
                xyz_scenarios[2] = [xyz2, xyz3]

            Edep2 = float(ph2[1][11]) * 1000.
            if Edep2 < Ecomplim * percabove:
                thetaLOR = calc_thetaLOR_3D_GATE(ph2[1], ph3[1], ph3[0])
                theta_diff = abs(thetaE - thetaLOR)
                if theta_diff < min_theta_diff:
                    min_theta_diff = theta_diff
                    xyz2 = return_xyz(ph2[1])
                    xyz3 = return_xyz(ph3[1])
                # find the theta diff for the other photon interaction
                thetaE2 = calc_thetaE(E1, Edep2)
                thetaLOR2 = calc_thetaLOR_3D_GATE(ph3[1], ph2[1], ph2[0])
                theta_diff2 = abs(thetaE2 - thetaLOR2)
                if theta_diff2 < min_theta_diff2:
                    min_theta_diff2 = theta_diff2
                    xyz2 = return_xyz(ph2[1])
                    xyz3 = return_xyz(ph3[1])
                xyz2 = return_xyz(ph2[1])
                xyz3 = return_xyz(ph3[1])
                sum_theta_diff_1[3] = theta_diff
                sum_theta_diff_2[3] = theta_diff2
                sum_theta_diff_scenarios[3] = theta_diff + theta_diff2
                xyz_scenarios[3] = [xyz2, xyz3]
        # find the lowest sum of theta difference
        ind = np.argmin(sum_theta_diff_scenarios)
        # if it's still the original value, then no combinations made sense
        if sum_theta_diff_scenarios[ind] == 1e6:
            print('No good choice for CC')
            min_theta_diff2 = -10000
            min_theta_diff = -10000
            min_pos = ['-1e6 -1e6 -1e6', '-1e6 -1e6 -1e6']
        else:
            min_theta_diff2 = sum_theta_diff_2[ind]
            min_theta_diff = sum_theta_diff_1[ind]
            min_pos = xyz_scenarios[ind]
        # return theta_diff in degrees, as well as first xyz of both first node points
        return [min_theta_diff2 * 180. / 3.14159, min_theta_diff * 180. / 3.14159], min_pos  # [xyz2,xyz3]
    else:
        print('Wrong type entered: ' + itype)
        return [-10000, -10000], ['', '']


# output format:
# xyz1 E1 xyz2 E2 false_line thetadiff1 thetadiff2 type samePositron Rayleigh PhantomScatter
# (where 1 = first annihilation photon, 2 = 2nd annihilation photon (on different node))
def process_P_C_GATE(N1_process, cut_list1, E1, N2_process, cut_list2, E2, NEC_printers, Ray, xyz1st_1, xyz1st_2,
                     choose_closest=False):
    # if it was not C-P,P-C,P-P,C-C don't process
    type1 = C_or_P(N1_process)
    type2 = C_or_P(N2_process)
    if type1 != 'N' and type2 != 'N':
        itype = type1 + type2
        # check for scatter within phantom (=1 if scatter, =0 if no)
        S = check_phantom_scatter(N1_process, N2_process)
        # get the first interaction x,y,z positions for both nodes
        # xyz1 = return_xyz(N1_process[0])
        # xyz2 = return_xyz(N2_process[0])
        # get the difference in xyz from the first interaction in detector
        # xyzdiff1 = str(float(xyz1st_1.split()[0])-float(xyz1.split()[0]))+' '+str(float(xyz1st_1.split()[1])-float(xyz1.split()[1]))+' '+str(float(xyz1st_1.split()[2])-float(xyz1.split()[2]))
        # xyzdiff2 = str(float(xyz1st_2.split()[0])-float(xyz2.split()[0]))+' '+str(float(xyz1st_2.split()[1])-float(xyz2.split()[1]))+' '+str(float(xyz1st_2.split()[2])-float(xyz2.split()[2]))

        # check if all events are from the same positron
        sameP = check_same_Positron(N1_process, N2_process)

        # now print depending on interaction type (P-P, C-P, P-C, C-C)
        # P-P
        if itype == 'PP':
            # check if the first interaction was lost
            if 1 in cut_list1 or 1 in cut_list2:
                # count it as a bad line
                false_line = 1
            else:
                false_line = 0
            thetadiff = [0, 0]  # since there is no thetadiff for P-P, set to zero
            # get the first interaction x,y,z positions for both nodes
            xyz1 = return_xyz(N1_process[0])
            xyz2 = return_xyz(N2_process[0])
        # print to file
        # NEC_printers.write(xyz1+' '+str(E1)+' '+xyz2+' '+str(E2)+' '+str(false_line)+' 0 0 PP '+str(sameP)+' '+str(Ray)+' '+str(S)+'\n')

        # P-C
        elif itype == 'PC':
            # check if the first interaction was lost
            if 1 in cut_list1 or 1 in cut_list2:  # or 2 in cut_list2:
                false_line = 1
            elif 2 in cut_list2:
                false_line = 2
            else:
                false_line = 0
            if choose_closest == False:
                thetadiff, [xyz1, xyz2] = find_thetadiff_GATE(N1_process, N2_process, itype)
            else:
                thetadiff = [-1, -1]  # there is no thetadiff when choosing closest
                [xyz1, xyz2] = find_closest_GATE(N1_process, N2_process, itype)
        # print to file
        # NEC_printers.write(xyz1+' '+str(E1)+' '+xyz2+' '+str(E2)+' '+str(false_line)+' '+str(thetadiff[0])+' '+str(thetadiff[1])+' PC '+str(sameP)+' '+str(Ray)+' '+str(S)+'\n')

        # C-P
        elif itype == 'CP':
            # check if the first interaction was lost
            if 1 in cut_list1 or 1 in cut_list2:
                false_line = 1
            elif 2 in cut_list1:
                false_line = 2
            else:
                false_line = 0
            if choose_closest == False:
                thetadiff, [xyz1, xyz2] = find_thetadiff_GATE(N1_process, N2_process, itype)
            else:
                thetadiff = [-1, -1]  # there is no thetadiff when choosing closest
                [xyz1, xyz2] = find_closest_GATE(N1_process, N2_process, itype)
        # get_dist_from_center(N1_process[0])
        # print to file
        # NEC_printers.write(xyz1+' '+str(E1)+' '+xyz2+' '+str(E2)+' '+str(false_line)+' '+str(thetadiff[0])+' '+str(thetadiff[1])+' CP '+str(sameP)+' '+str(Ray)+' '+str(S)+'\n')

        # C-C
        elif itype == 'CC':
            # check if the first interaction was lost
            if 1 in cut_list1 or 1 in cut_list2:
                # count it as a bad line
                false_line = 1
            elif 2 in cut_list1 or 2 in cut_list2:
                false_line = 2
            else:
                false_line = 0
            # get the thetadifference (for N1 and N2)
            if choose_closest == False:
                thetadiff, [xyz1, xyz2] = find_thetadiff_GATE(N1_process, N2_process, itype)
            else:
                thetadiff = [-1, -1]  # there is no thetadiff when choosing closest
                [xyz1, xyz2] = find_closest_GATE(N1_process, N2_process, itype)

        # get the difference in xyz from the first interaction in detector
        xyzdiff1 = str(float(xyz1st_1.split()[0]) - float(xyz1.split()[0])) + ' ' + str(
            float(xyz1st_1.split()[1]) - float(xyz1.split()[1])) + ' ' + str(
            float(xyz1st_1.split()[2]) - float(xyz1.split()[2]))
        xyzdiff2 = str(float(xyz1st_2.split()[0]) - float(xyz2.split()[0])) + ' ' + str(
            float(xyz1st_2.split()[1]) - float(xyz2.split()[1])) + ' ' + str(
            float(xyz1st_2.split()[2]) - float(xyz2.split()[2]))
        # Now print to file
        NEC_printers.write(xyz1 + ' ' + str(E1) + ' ' + xyz2 + ' ' + str(E2) + ' ' + str(false_line) + ' ' + str(
            thetadiff[0]) + ' ' + str(thetadiff[1]) + ' ' + itype + ' ' + str(sameP) + ' ' + str(Ray) + ' ' + str(
            S) + ' ' + N1_process[0][10] + ' ' + xyzdiff1 + ' ' + xyzdiff2 + '\n')


# N_indices = N1 and N2 indices
def process_Compton_kinetics(group, N1, N2, itype, Ecomplim, percabove):
    # now process only the Compton ones, to get theta
    E1 = 511  # =mec2 #energy of initial photon
    thetas = {}
    thetas['E_CC1'] = []  # thetaE for CC for node 1
    thetas['E_CC2'] = []  # thetaE for CC for node 2
    thetas['LOR_CC1'] = []
    thetas['LOR_CC2'] = []
    thetas['diff_CC1'] = []  # absolute theta difference for CC for node 1
    thetas['diff_CC2'] = []
    thetas['E_PC'] = []  # thetaE for PC on the node with compton
    thetas['LOR_PC'] = []
    thetas['diff_PC'] = []
    all_theta_diff = []
    final_grp = []
    correct_order = [0, 0]  # 0 means correct ordering (based off exactly known
    # timing, and thus order in the group, (ordering will match returned order)
    Norder = [1, 2]  # in this function the ordering of the nodes can be switched
    # depending on the event type. this needs to be kept track of
    # The algorithm seems to be N1,N2 except for CP, which is N2,N1

    min_theta_diff = 100000  # the smallest theta from the different combinations
    # N1 = N_indices[0] #N1 and N2 indices
    # N2 = N_indices[1]
    min_thetaE = -100000  # thetaE for min_theta_diff
    min_thetaLOR = -10000
    min_thetaE1 = -10000  # thetaE on node 1 for min_theta_diff
    min_thetaE2 = -10000
    min_thetaLOR1 = -10000
    min_thetaLOR2 = -10000
    min_thetadiff1 = -10000
    min_thetadiff2 = -10000
    # check CP events
    if itype == 'CP' or itype == 'XP':
        # check the different combinations of events (to get minimum theta difference) (it should always be the order in which they were in the file since they happen in order of time)
        min_grp = [N2[0], N1[0], N1[1]]  # default order, in case if's fail
        Norder = [2, 1]
        if itype == 'XP':
            min_theta_diff = 0
            min_thetaE = 0
            min_thetaLOR = 0

        if N1[0][2] < Ecomplim * percabove and itype != 'XP':  # check energy
            # if the energy is big, but within limit, set to maximum
            if N1[0][2] > Ecomplim:
                E2 = E1 - Ecomplim
            else:
                E2 = E1 - N1[0][2]
            thetaE = (180. / 3.14159) * math.acos(1.0 - mec2 * (1.0 / E2 - 1.0 / E1))
            # use xN2,yN2, xN1-1,yN1-1, xN1-2,yN1-2
            thetaLOR = calc_thetaLOR(N2[0], N1[0], N1[1])
            theta_diff = abs(thetaE - thetaLOR)  # get the difference of theta
            min_theta_diff = theta_diff
            min_grp = [N2[0], N1[0], N1[1]]
            correct_order = [0, 0]
            min_thetaE = thetaE
            min_thetaLOR = thetaLOR

        # try the other way (different order of events)
        if N1[1][2] < Ecomplim * percabove and itype != 'XP':  # check energy
            if N1[1][2] > Ecomplim:
                E2 = E1 - Ecomplim
            else:
                E2 = E1 - N1[1][2]

            thetaE = (180. / 3.14159) * math.acos(1.0 - mec2 * (1.0 / E2 - 1.0 / E1))
            # use xN2,yN2, xN1-2,yN1-2, xN1-1,yN1-1
            thetaLOR = calc_thetaLOR(N2[0], N1[1], N1[0])
            theta_diff = abs(thetaE - thetaLOR)
            # check if this theta difference was smaller
            if theta_diff < min_theta_diff:
                min_theta_diff = theta_diff
                min_grp = [N2[0], N1[1], N1[0]]
                correct_order = [0, 1]
                min_thetaE = thetaE
                min_thetaLOR = thetaLOR

        all_theta_diff.append(min_theta_diff)
        final_grp.append(min_grp)  # add this to the final number of groups
        # save the thetaE, thetaLOR and theta_diff for min_theta
        thetas['E_PC'].append(min_thetaE)
        thetas['LOR_PC'].append(min_thetaLOR)
        thetas['diff_PC'].append(min_theta_diff)

    # check PC events
    elif (len(N1) == 1 and len(N2) == 2) or itype == 'PX':
        min_grp = [N1[0], N2[0], N2[1]]  # default
        Norder = [1, 2]
        if itype == 'PX':
            min_theta_diff = 0
            min_thetaE = 0
            min_thetaLOR = 0

        # try the first order, check for energy
        if N2[0][2] < Ecomplim * percabove and itype != 'PX':
            if N2[0][2] > Ecomplim:
                E2 = E1 - Ecomplim
            else:
                E2 = E1 - N2[0][2]
            thetaE = (180. / 3.14159) * math.acos(1.0 - mec2 * (1.0 / E2 - 1.0 / E1))
            # use xN2,yN2, xN1-1,yN1-1, xN1-2,yN1-2
            thetaLOR = calc_thetaLOR(N1[0], N2[0], N2[1])
            theta_diff = abs(thetaE - thetaLOR)
            min_theta_diff = theta_diff
            min_grp = [N1[0], N2[0], N2[1]]
            correct_order = [0, 0]
            min_thetaE = thetaE
            min_thetaLOR = thetaLOR

        # trying the other way
        if N2[1][2] < Ecomplim * percabove and itype != 'PX':
            if N2[1][2] > Ecomplim:
                E2 = E1 - Ecomplim
            else:
                E2 = E1 - N2[1][2]
            thetaE = (180. / 3.14159) * math.acos(1.0 - mec2 * (1.0 / E2 - 1.0 / E1))
            # use xN2,yN2, xN1-1,yN1-1, xN1-2,yN1-2
            thetaLOR = calc_thetaLOR(N1[0], N2[1], N2[0])
            theta_diff = abs(thetaE - thetaLOR)
            if theta_diff < min_theta_diff:
                min_theta_diff = theta_diff
                min_grp = [N1[0], N2[1], N2[0]]
                correct_order = [0, 1]
                min_thetaE = thetaE
                min_thetaLOR = thetaLOR
        all_theta_diff.append(min_theta_diff)
        final_grp.append(min_grp)
        # save the thetaE, thetaLOR and theta_diff for min_theta
        thetas['E_PC'].append(min_thetaE)
        thetas['LOR_PC'].append(min_thetaLOR)
        thetas['diff_PC'].append(min_theta_diff)

    # check CC events
    elif (len(N1) == 2 and len(N2) == 2):  # or itype == 'XX' or itype == 'CX' or itype == 'XC':
        min_grp = [N1[0], N1[1], N2[0], N2[1]]  # default
        Norder = [1, 2]
        if itype[0] == 'X':
            theta_diff1 = 0
            thetaE1 = 0
            thetaLOR1 = 0
        if itype[1] == 'X':
            theta_diff2 = 0
            thetaE2 = 0
            thetaLOR2 = 0

        # try the first way, N2-1, N2-2  and N1-1, N1-2
        if N2[0][2] < Ecomplim * percabove:
            if N1[0][2] < Ecomplim * percabove:
                # find theta_diff for N1-1, N2-1 and N2-2
                if N2[0][2] > Ecomplim:
                    E2 = E1 - Ecomplim
                else:
                    E2 = E1 - N2[0][2]
                thetaE2 = (180. / 3.14159) * math.acos(1.0 - mec2 * (1.0 / E2 - 1.0 / E1))
                thetaLOR2 = calc_thetaLOR(N1[0], N2[0], N2[1])
                theta_diff2 = abs(thetaE2 - thetaLOR2)

                # find theta_diff for N2-1, N1-1 and N1-2
                if N1[0][2] > Ecomplim:
                    E2 = E1 - Ecomplim
                else:
                    E2 = E1 - N1[0][2]
                thetaE1 = (180. / 3.14159) * math.acos(1.0 - mec2 * (1.0 / E2 - 1.0 / E1))
                thetaLOR1 = calc_thetaLOR(N2[0], N1[0], N1[1])
                theta_diff1 = abs(thetaE1 - thetaLOR1)

                if itype[0] == 'X':
                    theta_diff1 = 0
                    thetaE1 = 0
                    thetaLOR1 = 0
                if itype[1] == 'X':
                    theta_diff2 = 0
                    thetaE2 = 0
                    thetaLOR2 = 0

                min_theta_diff = theta_diff2 + theta_diff1
                min_grp = [N1[0], N1[1], N2[0], N2[1]]
                correct_order = [0, 0]
                min_thetaE1 = thetaE1
                min_thetaLOR1 = thetaLOR1
                min_thetadiff1 = theta_diff1
                min_thetaE2 = thetaE2
                min_thetaLOR2 = thetaLOR2
                min_thetadiff2 = theta_diff2
            if N1[1][2] < Ecomplim * percabove:
                # find theta_diff for N1-2, N2-1 and N2-2
                if N2[0][2] > Ecomplim:
                    E2 = E1 - Ecomplim
                else:
                    E2 = E1 - N2[0][2]
                thetaE2 = (180. / 3.14159) * math.acos(1.0 - mec2 * (1.0 / E2 - 1.0 / E1))
                thetaLOR2 = calc_thetaLOR(N1[1], N2[0], N2[1])
                theta_diff2 = abs(thetaE2 - thetaLOR2)

                # find theta_diff for N2-1, N1-2 and N1-1
                if N1[1][2] > Ecomplim:
                    E2 = E1 - Ecomplim
                else:
                    E2 = E1 - N1[1][2]
                thetaE1 = (180. / 3.14159) * math.acos(1.0 - mec2 * (1.0 / E2 - 1.0 / E1))
                thetaLOR1 = calc_thetaLOR(N2[0], N1[1], N1[0])
                theta_diff1 = abs(thetaE1 - thetaLOR1)

                if itype[0] == 'X':
                    theta_diff1 = 0
                    thetaE1 = 0
                    thetaLOR1 = 0
                if itype[1] == 'X':
                    theta_diff2 = 0
                    thetaE2 = 0
                    thetaLOR2 = 0

                if (theta_diff1 + theta_diff2) < min_theta_diff:
                    min_theta_diff = theta_diff1 + theta_diff2
                    min_grp = [N1[1], N1[0], N2[0], N2[1]]
                    correct_order = [1, 0]
                    min_thetaE1 = thetaE1
                    min_thetaLOR1 = thetaLOR1
                    min_thetadiff1 = theta_diff1
                    min_thetaE2 = thetaE2
                    min_thetaLOR2 = thetaLOR2
                    min_thetadiff2 = theta_diff2

        # trying the other way
        if N2[1][2] < Ecomplim * percabove:
            if N2[0][2] < Ecomplim * percabove:
                # find theta_diff for N1-1, N2-2 and N2-1
                if N2[1][2] > Ecomplim:
                    E2 = E1 - Ecomplim
                else:
                    E2 = E1 - N2[1][2]
                thetaE2 = (180. / 3.14159) * math.acos(1.0 - mec2 * (1.0 / E2 - 1.0 / E1))
                # use xN2,yN2, xN1-1,yN1-1, xN1-2,yN1-2
                thetaLOR2 = calc_thetaLOR(N1[0], N2[1], N2[0])
                theta_diff2 = abs(thetaE2 - thetaLOR2)

                # find theta_diff for N2-2, N1-1 and N1-2
                if N1[0][2] > Ecomplim:
                    E2 = E1 - Ecomplim
                else:
                    E2 = E1 - N1[0][2]
                thetaE1 = (180. / 3.14159) * math.acos(1.0 - mec2 * (1.0 / E2 - 1.0 / E1))
                thetaLOR1 = calc_thetaLOR(N2[1], N1[0], N1[1])
                # add to existing theta_diff
                theta_diff1 = abs(thetaE1 - thetaLOR1)

                if itype[0] == 'X':
                    theta_diff1 = 0
                    thetaE1 = 0
                    thetaLOR1 = 0
                if itype[1] == 'X':
                    theta_diff2 = 0
                    thetaE2 = 0
                    thetaLOR2 = 0

                if (theta_diff1 + theta_diff2) < min_theta_diff:
                    min_theta_diff = theta_diff1 + theta_diff2
                    min_grp = [N1[0], N1[1], N2[1], N2[0]]
                    correct_order = [0, 1]
                    min_thetaE1 = thetaE1
                    min_thetaLOR1 = thetaLOR1
                    min_thetadiff1 = theta_diff1
                    min_thetaE2 = thetaE2
                    min_thetaLOR2 = thetaLOR2
                    min_thetadiff2 = theta_diff2

            if N2[1][2] < Ecomplim * percabove:
                # find theta_diff for N1-2, N2-2 and N2-1
                if N2[1][2] > Ecomplim:
                    E2 = E1 - Ecomplim
                else:
                    E2 = E1 - N2[1][2]
                thetaE2 = (180. / 3.14159) * math.acos(1.0 - mec2 * (1.0 / E2 - 1.0 / E1))
                thetaLOR2 = calc_thetaLOR(N1[1], N2[1], N2[0])
                theta_diff2 = abs(thetaE2 - thetaLOR2)

                # find theta_diff for N2-2, N1-2 and N1-1
                if N1[1][2] > Ecomplim:
                    E2 = E1 - Ecomplim
                else:
                    E2 = E1 - N1[1][2]
                thetaE1 = (180. / 3.14159) * math.acos(1.0 - mec2 * (1.0 / E2 - 1.0 / E1))
                thetaLOR1 = calc_thetaLOR(N2[1], N1[1], N1[0])
                # add to existing theta_diff
                theta_diff1 = abs(thetaE1 - thetaLOR1)

                if itype[0] == 'X':
                    theta_diff1 = 0
                    thetaE1 = 0
                    thetaLOR1 = 0
                if itype[1] == 'X':
                    theta_diff2 = 0
                    thetaE2 = 0
                    thetaLOR2 = 0

                if (theta_diff1 + theta_diff2) < min_theta_diff:
                    min_theta_diff = theta_diff1 + theta_diff2
                    min_grp = [N1[1], N1[0], N2[1], N2[0]]
                    correct_order = [1, 1]
                    min_thetaE1 = thetaE1
                    min_thetaLOR1 = thetaLOR1
                    min_thetadiff1 = theta_diff1
                    min_thetaE2 = thetaE2
                    min_thetaLOR2 = thetaLOR2
                    min_thetadiff2 = theta_diff2
        all_theta_diff.append(min_theta_diff)
        final_grp.append(min_grp)
        # save the thetaE, thetaLOR and theta_diff for both nodes for min_theta
        thetas['E_CC1'].append(min_thetaE1)
        thetas['LOR_CC1'].append(min_thetaLOR1)
        thetas['diff_CC1'].append(min_thetadiff1)
        thetas['E_CC2'].append(min_thetaE2)
        thetas['LOR_CC2'].append(min_thetaLOR2)
        thetas['diff_CC2'].append(min_thetadiff2)

    return min_grp, all_theta_diff, thetas, correct_order, Norder


# find if group is PP, CP, PC, or CC
def find_type(N1, N2):
    l_N1 = len(N1)
    l_N2 = len(N2)
    if (l_N1 == 1 and l_N2 == 1):
        itype = 'PP'
    elif (l_N1 == 2 and l_N2 == 1):
        itype = 'CP'
    elif (l_N1 == 1 and l_N2 == 2):
        itype = 'PC'
    elif (l_N1 == 2 and l_N2 == 2):
        itype = 'CC'
    else:
        itype = 'NOT A VALID TYPE'
    return itype


# find if group is P, 2A/2C in diff crystal, 1A/1C, 2A/2C in same crystal, 2A/1C, 2C/1A
# returns: ['type1','type2'], where type1, type2 = P, 2A/2C, etc
def find_subtype(N1, N2, actable):
    data = [N1, N2]
    types = []
    # go over each node's events
    for i in data:
        if len(i) == 1:
            types.append('PP')
        else:
            # find out if it was in the same crystal, etc
            # need to check for crystal
            no_mo_cr0 = '_'.join(i[0][8][5:8])
            no_mo_cr1 = '_'.join(i[1][8][5:8])
            # if they are not in the same crystal
            if no_mo_cr0 != no_mo_cr1:
                thistype = '2A/2C_diffC'
            else:  # if they are in same crystal
                # same AC
                if i[0][8][8] == i[1][8][8]:
                    thistype = '1A/1C'
                else:  # check the anodes and cathodes (within same crystal)
                    ac0 = actable[i[0][8][8]]
                    ac1 = actable[i[1][8][8]]
                    # different cases
                    if ac0[0] == ac1[0] and ac0[1] != ac1[1]:
                        thistype = '1A/2C'
                    elif ac0[0] != ac1[0] and ac0[1] == ac1[1]:
                        thistype = '2A/1C'
                    elif ac0[0] != ac1[0] and ac0[1] != ac1[1]:
                        thistype = '2A/2C_sameC'
                    else:
                        print('this case should never exist')
                        print(i[0])
                        print(i[1])
                        exit()
            types.append(thistype)
    return types


# blur the energy based on the desired type of blurring:
# -simple: blur everything based on the same %FWHM, i.e. the FWHM in % is not energy dependent, the initial way it was done
# -statlimited: (quantum noise), also called inverse square law
# -electroniclimited (electronic noise, flat response in terms of absolute FWHM and linear in terms of %)
def energy_blur(energy, Eblur_perc, blurtype='statlimited'):
    if energy <= 0:
        return 0

    # F = 1. #fano factor, assume=1

    # scale Eblur percent based on the blurring type:
    if blurtype == 'simple':
        FWHM = energy * Eblur_perc / 100.
    elif blurtype == 'statlimited':
        # based on Knoll book, 4th edition, ch4 p.117
        # Rpoisson0 = 2.35/sqrt(N0) = FWHM0 / H0
        # Rstatistical = 2.35 sqrt(F/N)
        # use the Eblur specified at 511keV to determine R1
        # Rpoisson1 = Rpoisson0 x sqrt(E0 / E1), see also GATE inverse square law for energy blurring
        R1 = Eblur_perc * math.sqrt(511. / energy)
        FWHM = R1 * energy / 100.
    elif blurtype == 'electroniclimited':
        # ignore the photon statistical noise
        # the FWHM (in absolute terms) for 511 will be the same for any energy
        FWHM = (Eblur_perc * 511.) / (100.)
    else:
        print('Choose a valid Eblur type', blurtype)
        exit()

    # get a new energy using gaussian distribution (old energy as mean)
    # divide by 100 because Eblur_perc is in percent
    # eSigma = (Eblur_perc*energy)/(235.5) # = /(100*2.355), to save time it is precomputed
    eSigma = (FWHM) / (2.355)
    # replace the energy value with the new gaussian distributed one
    energy = np.random.normal(energy, eSigma)
    if energy < 0:
        energy = 0.
    return energy


# set the energy blurring seed (to get reproducible results)
def set_energy_blur_seed(seed):
    np.random.seed(seed)


# print data to file
# N1 eventID1 sum(E1) thetaE1 thetaLOR1 thetadiff1 x1 y1 z1 N2 eventID2 sum(E2) thetaE2 thetaLOR2 thetadiff2 x2 y2 z2
def print_groups_lst(group, itype, w_all, w_PP, w_PCC, w_CC, w_CP, all_theta_diff, thetas, Ray, N1xyz, N1droppedbef,
                     N2xyz, N2droppedbef, correct_order, Norder, group_real='', prtgrp=True):
    i = 0  # legacy, depricated

    # print the groups
    if prtgrp == True:
        if group_real == '':
            for line in group:
                w_all.write(line[0] + ' ' + str(line[1]) + ' ' + str(line[2]) + ' ' + str(line[3]) + ' ' + str(
                    line[4]) + ' ' + str(line[5]) + ' ' + line[6] + ' ' + str(line[7]) + ' ' + str(Ray) + '\n')
        else:
            for jj, line in enumerate(group):
                # need to first replace energy with energy saved in group (because that one is energy blurred)
                # line[8] is l (the parsed line)
                line[8][11] = str(line[2])
                if float(line[8][11]) > 100:  # 0.1: #100kev threshold
                    line[8][-1] = line[8][-1].replace('\n', '')
                    w_all.write(' '.join(line[8]) + ',')
            w_all.write('\n')

    # Added to output file 09-11-2016:
    # itype,Ray,S,sameP,correct_order,correct_order1,correct_order2,Droppedbef,Droppedbef1,Droppedbef2,1st_xyz1,1st_xyz2
    # if S=1, there was scatter (Rayleigh or Compton) with phantom
    # S = check_phantom_scatter(group[0],group[1],dtype='embed')
    # if sameP=1, there was data with different eventID
    # sameP = check_same_Positron(group[0],group[1],dtype='embed')
    # correct_order,Norder
    if correct_order[0] == 1 or correct_order[1] == 1:
        Ncororder = 1
    else:
        Ncororder = 0
    # N1xyz,N1droppedbef,N2xyz,N2droppedbef
    if N1droppedbef == 1 or N2droppedbef == 1:
        Ndrop = 1
    else:
        Ndrop = 0
    # NOTE: instead of looking at Norder you can also look at the node number in
    # group to see which is bigger
    # need to watch out for order switch in ckinetic
    if Norder[0] > Norder[1]:
        Nxyz = [N2xyz, N1xyz]
        Ndroppedbef = [N2droppedbef, N1droppedbef]
    else:
        Nxyz = [N1xyz, N2xyz]
        Ndroppedbef = [N1droppedbef, N2droppedbef]

    # print the PP lst file
    if itype == 'PP':  # len(group) == 2: #if 1x N1 and 1x N2, PP
        # if S=1, there was scatter (Rayleigh or Compton) with phantom
        # S = check_phantom_scatter([group[0]],[group[1]],dtype='embed')
        S = check_phantom_scatter_mod([group[0]], [group[1]], dtype='embed')
        # if sameP=1, there was data with different eventID
        sameP, samePID = check_same_Positron([group[0]], [group[1]], dtype='embed', ID=True)

        # print: N1 eventID1 E1 0 0 0 x1 y1 z1 N2 eventID2 E2 0 0 0 x2 y2 z2
        w_PP.write(str(group[0][7]) + ' ' + str(group[0][0]) + ' ' + str(group[0][2]) + ' 0 0 0 ' + str(
            group[0][3]) + ' ' + str(group[0][4]) + ' ' + str(group[0][5]) + ' ')
        w_PP.write(str(group[1][7]) + ' ' + str(group[1][0]) + ' ' + str(group[1][2]) + ' 0 0 0 ' + str(
            group[1][3]) + ' ' + str(group[1][4]) + ' ' + str(group[1][5]) + ' ')
        # added 09-11-2016
        # itype,Ray,S,sameP,samePID,correct_order,correct_order1,correct_order2,Droppedbef,Droppedbef1,Droppedbef2,1st_xyz1,1st_xyz2
        w_PP.write(str(itype) + ' ' + str(Ray) + ' ' + str(S) + ' ' + str(sameP) + ' ' + str(samePID) + ' ' + str(
            Ncororder) + ' ' + str(correct_order[0]) + ' ' + str(correct_order[1]) + ' ' + str(Ndrop) + ' ' + str(
            Ndroppedbef[0]) + ' ' + str(Ndroppedbef[1]) + ' ' + str(Nxyz[0]) + ' ' + str(Nxyz[1]) + '\n')

    else:
        # print the results and add theta_diff at the end of each line
        # before printing, reorder it based on node (so it's easier to read file)
        # group.sort(key=lambda x: x[7]) #N1 first, then N2
        if prtgrp == True:
            for line in group:
                w_PCC.write(line[0] + ' ' + str(line[1]) + ' ' + str(line[2]) + ' ' + str(line[3]) + ' ' + str(
                    line[4]) + ' ' + str(line[5]) + ' ' + line[6] + ' ' + str(line[7]) + ' ' + str(
                    all_theta_diff[i]) + '\n')
            # w.write(line[0]+' '+str(line[1])+' '+str(line[2])+' '+str(line[3])+' '+str(line[4])+' '+str(line[5])+' '+line[6]+' '+str(line[7])+' '+str(all_theta_diff[i]*180./3.14159)+'\n')

        # print the CP and CC lst files
        CP_counter = 0
        CC_counter = 0
        if len(group) == 3:  # if 2x N1 and 1x N2 OR 1x N1 and 2x N1, CP
            # format for final_grp is always PCC (1 P point, then 2 C points)

            # if S=1, there was scatter (Rayleigh or Compton) with phantom
            # S = check_phantom_scatter([group[0]],[group[1],group[2]],dtype='embed')
            S = check_phantom_scatter_mod([group[0]], [group[1], group[2]], dtype='embed')
            # if sameP=1, there was data with different eventID
            sameP, samePID = check_same_Positron([group[0]], [group[1], group[2]], dtype='embed', ID=True)

            # check for node 1 is P, node 2 is C
            #		if itype == 'PC':
            # if str(group[0][7]) == '1':
            # print: N1 eventID1 E1 0 0 0 x1 y1 z1 N2 eventID2 sum(E2) thetaE thetaLOR thetadiff x2 y2 z2
            w_CP.write(str(group[0][7]) + ' ' + str(group[0][0]) + ' ' + str(group[0][2]) + ' 0 0 0 ' + str(
                group[0][3]) + ' ' + str(group[0][4]) + ' ' + str(group[0][5]) + ' ')
            # node 2 is the C
            w_CP.write(str(group[1][7]) + ' ' + str(group[1][0]) + ' ' + str(group[1][2] + group[2][2]) + ' ' + str(
                thetas['E_PC'][CP_counter]) + ' ' + str(thetas['LOR_PC'][CP_counter]) + ' ' + str(
                thetas['diff_PC'][CP_counter]) + ' ' + str(group[1][3]) + ' ' + str(group[1][4]) + ' ' + str(
                group[1][5]) + ' ')
            # node 2 is P, node 1 is C
            #		else:
            # print: N1 eventID1 sum(E1) thetaE thetaLOR thatadiff x1 y1 z1 N2 eventID2 E2 thetaE thetaLOR thetadiff x2 y2 z2
            #			w_CP.write(str(group[1][7])+' '+str(group[1][0])+' '+str(group[1][2]+group[2][2])+' '+str(thetas['E_PC'][CP_counter])+' '+str(thetas['LOR_PC'][CP_counter])+' '+str(thetas['diff_PC'][CP_counter])+' '+str(group[1][3])+' '+str(group[1][4])+' '+str(group[1][5])+' ')
            # node 2 is the C
            #			w_CP.write(str(group[0][7])+' '+str(group[0][0])+' '+str(group[0][2])+' 0 0 0 '+str(group[0][3])+' '+str(group[0][4])+' '+str(group[0][5])+'\n')

            w_CP.write(str(itype) + ' ' + str(Ray) + ' ' + str(S) + ' ' + str(sameP) + ' ' + str(samePID) + ' ' + str(
                Ncororder) + ' ' + str(correct_order[0]) + ' ' + str(correct_order[1]) + ' ' + str(Ndrop) + ' ' + str(
                Ndroppedbef[0]) + ' ' + str(Ndroppedbef[1]) + ' ' + str(Nxyz[0]) + ' ' + str(Nxyz[1]) + '\n')

            CP_counter += 1  # update counter
        # otherwise, it was a CC
        else:
            # if S=1, there was scatter (Rayleigh or Compton) with phantom
            # S = check_phantom_scatter([group[0],group[1]],[group[2],group[3]],dtype='embed')
            S = check_phantom_scatter_mod([group[0], group[1]], [group[2], group[3]], dtype='embed')
            # if sameP=1, there was data with different eventID
            sameP, samePID = check_same_Positron([group[0], group[1]], [group[2], group[3]], dtype='embed', ID=True)

            # check for node 1
            # if str(group[0][7]) == '1':
            # print: N1 eventID1 sum(E1) thetaE1 thetaLOR1 thetadiff1 x1 y1 z1 N2 eventID2 sum(E2) thetaE2 thetaLOR2 thetadiff2 x2 y2 z2
            w_CC.write(str(group[0][7]) + ' ' + str(group[0][0]) + ' ' + str(group[0][2] + group[1][2]) + ' ' + str(
                thetas['E_CC1'][CC_counter]) + ' ' + str(thetas['LOR_CC1'][CC_counter]) + ' ' + str(
                thetas['diff_CC1'][CC_counter]) + ' ' + str(group[0][3]) + ' ' + str(group[0][4]) + ' ' + str(
                group[0][5]) + ' ')
            # node 2 is the C
            w_CC.write(str(group[2][7]) + ' ' + str(group[2][0]) + ' ' + str(group[2][2] + group[3][2]) + ' ' + str(
                thetas['E_CC2'][CC_counter]) + ' ' + str(thetas['LOR_CC2'][CC_counter]) + ' ' + str(
                thetas['diff_CC2'][CC_counter]) + ' ' + str(group[2][3]) + ' ' + str(group[2][4]) + ' ' + str(
                group[2][5]) + ' ')
            # w_CC.write(str(group[2][7])+' '+str(group[2][0])+' '+str(group[2][2]+group[3][2])+' '+str(thetas['E_PC'][CC_counter])+' '+str(thetas['LOR_PC'][CC_counter])+' '+str(thetas['diff_PC'][CC_counter])+' '+str(group[2][3])+' '+str(group[2][4])+' '+str(group[2][5])+'\n')
            # node 2 is first in group
            #			else:
            # print: N1 eventID1 sum(E1) thetaE1 thetaLOR1 thetadiff1 x1 y1 z1 N2 eventID2 sum(E2) thetaE2 thetaLOR2 thetadiff2 x2 y2 z2
            #				w_CC.write(str(group[2][7])+' '+str(group[2][0])+' '+str(group[2][2]+group[3][2])+' '+str(thetas['E_CC1'][CC_counter])+' '+str(thetas['LOR_CC1'][CC_counter])+' '+str(thetas['diff_CC1'][CC_counter])+' '+str(group[2][3])+' '+str(group[2][4])+' '+str(group[2][5])+' ')
            # node 2 is the C
            #				w_CC.write(str(group[0][7])+' '+str(group[0][0])+' '+str(group[0][2]+group[1][2])+' '+str(thetas['E_CC2'][CC_counter])+' '+str(thetas['LOR_CC2'][CC_counter])+' '+str(thetas['diff_CC2'][CC_counter])+' '+str(group[0][3])+' '+str(group[0][4])+' '+str(group[0][5])+'\n')

            w_CC.write(str(itype) + ' ' + str(Ray) + ' ' + str(S) + ' ' + str(sameP) + ' ' + str(samePID) + ' ' + str(
                Ncororder) + ' ' + str(correct_order[0]) + ' ' + str(correct_order[1]) + ' ' + str(Ndrop) + ' ' + str(
                Ndroppedbef[0]) + ' ' + str(Ndroppedbef[1]) + ' ' + str(Nxyz[0]) + ' ' + str(Nxyz[1]) + '\n')
            CC_counter += 1  # update counter





