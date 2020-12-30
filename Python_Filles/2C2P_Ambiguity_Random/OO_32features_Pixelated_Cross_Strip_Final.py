from collections import defaultdict
import vg
import math as m
import numpy as np
import random
import pandas as pd

np.seterr(divide='ignore', invalid='ignore')


# -------------------------------------------------------------------------------------------------------------
# Target Sequence: [C1P1, C2P2] OR [C1P1, P2C2] OR [P1C1, C2P2] OR [P1C1, P2C2] sorted based on time
# Features: After blurring Energy, we gain our features such as theta_p, theta_e, energies ...
# # Don't forget to blur energy and shuffle family:)
# -------------------------------------------------------------------------------------------------------------


def icos(a):
    if a > 1.0:
        a = 1.0
    elif a < -1.0:
        a = -1.
    inv_ = m.acos(a)
    return m.degrees(inv_)

    # -------------------------------------------------------------------------------------------------------------
    # ----------------------------- Filter families based on Energy Window ----------------------------------------
    # -------------------------------------------------------------------------------------------------------------


def process_family(family):
    dict_ = defaultdict(list)
    for i, item in enumerate(family):
        dict_[item[-8]].append(i)  # ID of particle [-8]

    for key in dict_:
        items = dict_[key]  # items = [0, 1] [2, 3] ke 0 ye satre kamele
        energy = 0.0
        for item in items:
            energy += float(family[item][11])
        if energy < 421 or energy > 621:  # if energy < 421 or energy > 601:
            return False
        # print(items)
    return True

    # -------------------------------------------------------------------------------------------------------------
    # ---------------------------------------- Calculate Theta_P and Theta_E and Features for ML-------------------
    # -------------------------------------------------------------------------------------------------------------


def extract_theta(family, a, b, c):
    a_vector_1 = [float(family[a][13]) - float(family[c][13]),
                  float(family[a][14]) - float(family[c][14]),
                  float(family[a][15]) - float(family[c][15])]
    b_vector_1 = [float(family[b][13]) - float(family[a][13]),
                  float(family[b][14]) - float(family[a][14]),
                  float(family[b][15]) - float(family[a][15])]  # j i k -> j i l
    a_vector_1 = np.array(a_vector_1)
    b_vector_1 = np.array(b_vector_1)
    theta_p = vg.angle(a_vector_1, b_vector_1)
    theta_e = icos(1. - 511. * (1 / (float(family[b][11])) - 1 / (float(family[a][11]) + float(family[b][11]))))
    return theta_p, theta_e  # by this function, we gain theta_p_1 .. theta_p_8 and theta_e_1 .. theta_e_8


def ambiguity1_extract_theta(family, a, b, c):
    if float(family[a][14]) < 1e-2 or float(family[a][13]) < 1e-2:
        family[a][14] = 1e-2
        family[a][13] = 1e-2
    a_vector_9 = [float(family[a][13]) - float(family[c][13]),
                  float(family[b][14]) - float(family[c][14]),
                  float('{:.4f}'.format(float(family[b][14]) / float(family[a][13]))) - float(family[c][15])]
    b_vector_9 = [float(family[a][14]) - float(family[a][13]),
                  float(family[b][13]) - float(family[b][14]),
                  float('{:.4f}'.format(float(family[b][13]) / float(family[a][14]))) - float(
                      '{:.4f}'.format(float(family[b][14]) / float(family[a][13])))]
    a_vector_9 = np.array(a_vector_9)
    b_vector_9 = np.array(b_vector_9)
    theta_p_9 = vg.angle(a_vector_9, b_vector_9)
    theta_e_9 = icos(1. - 511. * (1 / (float(family[b][11])) - 1 / (float(family[a][11]) + float(family[b][11]))))

    return theta_p_9, theta_e_9


def ambiguity2_extract_theta(family, a, b, c):
    # -------------------------------provide Second Ambiguity from First Ambiguity-------------------------------------
    if float(family[a][14]) < 1e-2 or float(family[a][13]) < 1e-2:
        family[a][14] = 1e-2
        family[a][13] = 1e-2

    a_vector_10 = [float(family[a][14]) - float(family[c][13]),
                   float(family[b][13]) - float(family[c][14]),
                   float('{:.4f}'.format(float(family[b][13]) / float(family[a][14]))) - float(family[c][15])]
    b_vector_10 = [-float(family[a][14]) + float(family[a][13]),
                   -float(family[b][13]) + float(family[b][14]),
                   -float('{:.4f}'.format(float(family[b][13]) / float(family[a][14]))) + float(family[b][13])]
    a_vector_10 = np.array(a_vector_10)
    b_vector_10 = np.array(b_vector_10)
    theta_p_10 = vg.angle(a_vector_10, b_vector_10)
    theta_e_10 = icos(1. - 511. * (1 / (float(family[a][11])) - 1 / (float(family[a][11]) + float(family[b][11]))))
    return theta_p_10, theta_e_10


def calculate(family):  # col10: time stamp nana C1: 10866 C2: 5609 C3: 5555 C4: 2738
    # global non_comp, compton, pe
    # random.shuffle(family)
    return_pack = {'Event_ID': [], 'ID_Flag': [], 'X': [], 'Y': [], 'Z': [],
                   'theta_p_1': [], 'theta_e_1': [], 'theta_p_2': [], 'theta_e_2': [],
                   'theta_p_3': [], 'theta_e_3': [], 'theta_p_4': [], 'theta_e_4': [],
                   'energy_c1': [], 'energy_p1': [], 'energy_c2': [], 'energy_p2': [],
                   'event1x': [], 'event1y': [], 'event1z': [],
                   'event2x': [], 'event2y': [], 'event2z': [],
                   'event3x': [], 'event3y': [], 'event3z': [],
                   'event4x': [], 'event4y': [], 'event4z': [],
                   'time1': [], 'time2': [], 'time3': [], 'time4': [],
                   'DDA1': [], 'DDA2': [], 'DDA3': [], 'DDA4': [],
                   'target_seq': [],
                   'theta_p_T': [], 'theta_e_T': [],
                   'theta_p_F': [], 'theta_e_F': [],
                   'c1': 0, 'c2': 0, 'c3': 0, 'c4': 0,
                   'rf_counter': 0, 'tf_counter': 0, 'valid_family': False}
    # -------------------------------------------------------------------------------------------------------------
    # ---------------------------------Check for family validity whether we have 1C2P combination -----------------
    # -------------------------------------------------------------------------------------------------------------
    if len(family) != 4:
        return return_pack

    counter_2 = 0
    counter_3 = 0
    for i in range(len(family)):  # count the number of rows for column -8 to be 2 or 3
        if family[i][-8] == '2':
            counter_2 += 1
        elif family[i][-8] == '3':
            counter_3 += 1
        """return empty pack if the row is neither Compton nor Photon"""
        if family[i][-3] not in ['Compton', 'PhotoElectric']:
            return return_pack

    if counter_2 != 2 or counter_3 != 2:  # Check if family has 2P2C photon IDs
        return return_pack

    # -------------------------------------------------------------------------------------------------------------
    # ----------------------------------Check if all event ids are identical to recognize Random Coincidences -
    # -------------------------------------------------------------------------------------------------------------
    event_id = []
    for row in family:
        event_id.append(int(row[1]))

    if event_id[1:] == event_id[:-1]:
        return_pack['ID_Flag'].append(1)
        # return_pack['tf_counter'] += 1
    else:
        return_pack['ID_Flag'].append(0)
        # return_pack['rf_counter'] += 1

    """ Blur Energy to find theta_p and theta_e after blurring not from Ground Truth"""
    mu1, sigma1 = 0.0, 17.35881104
    for i in range(len(family)):
        val = float(family[i][11])
        val += np.random.normal(mu1, sigma1)
        family[i][11] = str(val)

    # -------------------------------------------------------------------------------------------------------------
    # Prepare target sequence: [C1P1, C2P2] OR [C1P1, P2C2] OR [P1C1, C2P2] OR [P1C1, P2C2] sorted by time! Smaller time
    # comes first]
    # -------------------------------------------------------------------------------------------------------------

    dict_label = {'time': [], 'panel': [], 'row': []}
    for i, row in enumerate(family):
        dict_label['time'].append(row[10])
        dict_label['panel'].append(row[-8])
        dict_label['row'].append(i)

    df = pd.DataFrame(dict_label)
    df = df.sort_values(by=['time'], ignore_index=False)

    first_panel = df['panel'][0]
    target_seq = []
    target_seq.extend(list(df[df['panel'] == first_panel]['row']))
    target_seq.extend(list(df[df['panel'] != first_panel]['row']))
    return_pack['target_seq'] = target_seq

    # -------------------------------------------------------------------------------------------------------------
    # ---------------------------------------- Calculate Theta_P and Theta_E --------------------------------------
    # -------------------------------------------------------------------------------------------------------------

    i = target_seq[0]
    j = target_seq[1]
    k = target_seq[2]
    l = target_seq[3]

    # print(i, j, k, l)
    ################ Calculate 16 theta_p and theta_e ##########################################

    theta_p_1, theta_e_1 = extract_theta(family, i, j, k)
    theta_p_2, theta_e_2 = extract_theta(family, j, i, k)
    theta_p_3, theta_e_3 = extract_theta(family, k, l, j)
    theta_p_4, theta_e_4 = extract_theta(family, l, k, j)
    theta_p_5, theta_e_5 = extract_theta(family, i, j, l)
    theta_p_6, theta_e_6 = extract_theta(family, j, i, l)
    theta_p_7, theta_e_7 = extract_theta(family, k, l, i)
    theta_p_8, theta_e_8 = extract_theta(family, l, k, i)

    theta_p_9, theta_e_9 = ambiguity1_extract_theta(family, i, j, k)  # provide First Ambiguity
    theta_p_10, theta_e_10 = ambiguity2_extract_theta(family, i, j, k)  # provide Second Ambiguity
    theta_p_11, theta_e_11 = ambiguity1_extract_theta(family, i, j, l)  # provide third Ambiguity
    theta_p_12, theta_e_12 = ambiguity2_extract_theta(family, i, j, l)  # provide fourth Ambiguity

    theta_p_13, theta_e_13 = ambiguity1_extract_theta(family, k, l, j)  # provide fifth Ambiguity
    theta_p_14, theta_e_14 = ambiguity2_extract_theta(family, k, l, j)  # provide sixth Ambiguity
    theta_p_15, theta_e_15 = ambiguity1_extract_theta(family, k, l, i)  # provide seventh Ambiguity
    theta_p_16, theta_e_16 = ambiguity2_extract_theta(family, k, l, i)  # provide eighth Ambiguity

    ######################################################################################################################

    ###################################################################################################################
    # ------------------------
    # theta_p_1 hamishe awal doroste chon ba time order shode va midunim awal ki khorde!
    return_pack['theta_p_T'].extend([theta_p_1, theta_p_3])  # theta_p_1, theta_p_3 are true based on TIME! Doroste!
    return_pack['theta_e_T'].extend([theta_e_1, theta_e_3])
    return_pack['theta_p_F'].extend([theta_p_2, theta_p_4, theta_p_5, theta_p_6, theta_p_7, theta_p_8,
                                     theta_p_9, theta_p_10, theta_p_11, theta_p_12, theta_p_13, theta_p_14,
                                     theta_p_15, theta_p_16])
    return_pack['theta_e_F'].extend([theta_e_2, theta_e_4, theta_e_5, theta_e_6, theta_e_7, theta_e_8,
                                     theta_e_9, theta_e_10, theta_e_11, theta_e_12, theta_e_13, theta_e_14,
                                     theta_e_15, theta_e_16])

    DDA1 = abs(return_pack['theta_p_T'][0] - return_pack['theta_e_T'][0])
    return_pack['DDA1'] = DDA1
    DDA2 = abs(return_pack['theta_p_F'][0] - return_pack['theta_e_F'][0])
    return_pack['DDA2'] = DDA2
    DDA3 = abs(return_pack['theta_p_T'][1] - return_pack['theta_e_T'][1])
    return_pack['DDA3'] = DDA3
    DDA4 = abs(return_pack['theta_p_F'][1] - return_pack['theta_e_F'][1])
    return_pack['DDA4'] = DDA4

    # DDA5 = abs(return_pack['theta_p_F'][2] - return_pack['theta_e_F'][2])
    # return_pack['DDA5'] = DDA5
    # DDA6 = abs(return_pack['theta_p_F'][3] - return_pack['theta_e_F'][3])
    # return_pack['DDA6'] = DDA6
    # DDA7 = abs(return_pack['theta_p_F'][4] - return_pack['theta_e_F'][4])
    # return_pack['DDA7'] = DDA7
    # DDA8 = abs(return_pack['theta_p_F'][5] - return_pack['theta_e_F'][5])
    # return_pack['DDA8'] = DDA8
    if np.isnan(theta_p_1) or np.isnan(theta_p_2) or np.isnan(theta_p_3) or np.isnan(theta_p_4) \
            or np.isnan(theta_p_5) or np.isnan(theta_p_6) or np.isnan(theta_p_7) or np.isnan(theta_p_8):
        return return_pack
    # print('DDA1:', DDA1, 'DDA2:', DDA2, 'DDA3:', DDA3, 'DDA4:', DDA4, 'DDA5:', DDA5, 'DDA6:', DDA6, 'DDA7:', DDA7, 'DDA8:', DDA8)

    return_pack['valid_family'] = True

    return return_pack


def main():
    with open("50BigBinnedNoBlurred4.csv", 'r') as f:  # 3 radif data 50BigBinnedNoBlurred4
        with open("oo_test_Output_16.csv", 'w') as g:  # az 3 radif + kudum Correct kudum Wrong!
            # lines = f.readlines()
            family = []
            invalid_family_counter = 0
            tf_counter = 0
            rf_counter = 0
            C1 = 0
            C2 = 0
            C3 = 0
            C4 = 0
            for line in f:
                out = line.rstrip("\r\n")
                if out == "":
                    process = process_family(family)
                    # print('process', process)
                    if process:
                        return_pack = calculate(family)
                        # print('return_pack:', return_pack)
                        if return_pack['valid_family']:
                            if return_pack['DDA1'] <= return_pack['DDA2'] and return_pack['DDA3'] <= return_pack[
                                'DDA4']:
                                C1 += 1  # C1: 10866
                            elif return_pack['DDA1'] <= return_pack['DDA2'] and return_pack['DDA3'] >= return_pack[
                                'DDA4']:
                                C2 += 1  # C2: 5609
                            elif return_pack['DDA1'] >= return_pack['DDA2'] and return_pack['DDA3'] <= return_pack[
                                'DDA4']:
                                C3 += 1  # C3: 5555
                            elif return_pack['DDA1'] >= return_pack['DDA2'] and return_pack['DDA3'] >= return_pack[
                                'DDA4']:
                                C4 += 1  # C4: 2738
                            # We put true thetas in the 0 index! From time we knew which sequence is true!
                            x_ = [[return_pack['theta_p_T'][0], return_pack['theta_e_T'][0],
                                   return_pack['theta_p_T'][1], return_pack['theta_e_T'][1]],
                                  [return_pack['theta_p_F'][0], return_pack['theta_e_F'][0],
                                   return_pack['theta_p_F'][1], return_pack['theta_e_F'][1]],
                                  [return_pack['theta_p_F'][2], return_pack['theta_e_F'][2],
                                   return_pack['theta_p_F'][3], return_pack['theta_e_F'][3]],
                                  [return_pack['theta_p_F'][4], return_pack['theta_e_F'][4],
                                   return_pack['theta_p_F'][5], return_pack['theta_e_F'][5]],
                                  [return_pack['theta_p_F'][6], return_pack['theta_e_F'][6],
                                   return_pack['theta_p_F'][7], return_pack['theta_e_F'][7]],
                                  [return_pack['theta_p_F'][8], return_pack['theta_e_F'][8],
                                   return_pack['theta_p_F'][9], return_pack['theta_e_F'][9]],
                                  [return_pack['theta_p_F'][10], return_pack['theta_e_F'][10],
                                   return_pack['theta_p_F'][11], return_pack['theta_e_F'][11]],
                                  [return_pack['theta_p_F'][12], return_pack['theta_e_F'][12],
                                   return_pack['theta_p_F'][13], return_pack['theta_e_F'][13]]
                                  ]

                            indices = [0, 1, 2, 3, 4, 5, 6, 7]  # list(range(len(x_)))
                            random.shuffle(indices)
                            if return_pack['ID_Flag'][0] == 0:
                                label = max(indices) + 1
                            else:
                                label = indices.index(0)

                            # x = shuffle (temp)
                            x = []
                            for i in indices:
                                # print(x_[i])
                                # print('**' * 10)
                                x.extend(x_[i])
                            # print(x[7])

                            g.write(str(return_pack['ID_Flag'][0]) + "\t" +
                                    str(x[0]) + "\t" + str(x[1]) + "\t" + str(x[2]) + "\t" + str(x[3]) + "\t" +
                                    str(x[4]) + "\t" + str(x[5]) + "\t" + str(x[6]) + "\t" + str(x[7]) + "\t" +
                                    str(x[8]) + "\t" + str(x[9]) + "\t" + str(x[10]) + "\t" + str(x[11]) + "\t" +
                                    str(x[12]) + "\t" + str(x[13]) + "\t" + str(x[14]) + "\t" + str(x[15]) + "\t" +
                                    str(x[16]) + "\t" + str(x[17]) + "\t" + str(x[18]) + "\t" + str(x[19]) + "\t" +
                                    str(x[20]) + "\t" + str(x[21]) + "\t" + str(x[22]) + "\t" + str(x[23]) + "\t" +
                                    str(x[24]) + "\t" + str(x[25]) + "\t" + str(x[26]) + "\t" + str(x[27]) + "\t" +
                                    str(x[28]) + "\t" + str(x[29]) + "\t" + str(x[30]) + "\t" + str(x[31]) + "\t" +
                                    str(label) + "\n")


                        else:
                            invalid_family_counter += 1
                    family = []
                    continue
                else:
                    out = out.strip()
                    items = out.split("\t")
                    if items[22] != "RayleighScattering":
                        family.append(items)
            print('C1:', C1, 'C2:', C2, 'C3:', C3, 'C4:', C4)


if __name__ == "__main__":
    main()
