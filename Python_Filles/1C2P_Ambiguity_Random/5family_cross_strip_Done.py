from collections import defaultdict
import vg
import math as m
import numpy as np
import random
import pandas as pd

np.seterr(divide='ignore', invalid='ignore')


# -------------------------------------------------------------------------------------------------------------
# Target Label: az ruye time label mizanam! uni  ke zudtar khorde Theta_p, theta_e uno awale list mizarim!
# Bad bara ML, shuffle mikonim(0, 1, 2, 3)
# Features: After blurring Energy, we gain our features such as theta_p, theta_e, energies ...
# Blur Energy! Shuffle and Families!
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
        items = dict_[key]  # items = [0, 1] ke 0 ye satre kamele
        energy = 0.0
        for item in items:
            energy += float(family[item][11])
        if energy < 421 or energy > 621:  # if energy < 421 or energy > 601:
            return False
    return True

    # -------------------------------------------------------------------------------------------------------------
    # ---------------------------------------- Calculate Theta_P and Theta_E and Features for ML-------------------
    # -------------------------------------------------------------------------------------------------------------


def calculate(family):
    # global non_comp, compton, pe # This is not true!
    random.shuffle(family)
    return_pack = {'Event_ID': [], 'ID_Flag': [], 'X': [], 'Y': [], 'Z': [],
                   'theta_p_1': [], 'theta_e_1': [], 'theta_p_2': [], 'theta_e_2': [],
                   'energy_non': [], 'energy_comp': [], 'energy_pe': [],
                   'event1x': [], 'event1y': [], 'event1z': [],
                   'event2x': [], 'event2y': [], 'event2z': [],
                   'event3x': [], 'event3y': [], 'event3z': [],
                   'time_non': [], 'time_comp': [], 'time_pe': [],
                   'DDA1': [], 'DDA2': [],
                   'target_seq_T': [], 'target_seq_F': [], 'panel1': [], 'panel2': [], 'panel3': [],
                   'label_T': None, 'label_F': None,
                   'energy_T': [], 'energy_F': [],
                   'time_T': [], 'time_F': [],
                   'theta_p_T': [], 'theta_p_F': [],
                   'theta_e_T': [], 'theta_e_F': [],
                   'rf_counter': 0, 'tf_counter': 0, 'valid_family': False}
    # -------------------------------------------------------------------------------------------------------------
    # ---------------------------------Check for family validity whether we have 1C2P combination -----------------
    # -------------------------------------------------------------------------------------------------------------

    if len(family) != 3:
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
    """ Check if family has 2P1C photon IDs """
    if not ((counter_2 == 1 and counter_3 == 2) or (counter_2 == 2 and counter_3 == 1)):
        # if not (counter_2 == 1 and counter_3 == 2):
        return return_pack

    # -------------------------------------------------------------------------------------------------------------
    # ---------------------------------------Check if all event ids are identical ---------------------------------
    # -------------------------------------------------------------------------------------------------------------
    event_id = []
    for row in family:
        event_id.append(int(row[1]))
        return_pack['Event_ID'] = event_id[0]

    if event_id[1:] == event_id[:-1]:
        return_pack['ID_Flag'].append(1)
        # return_pack['tf_counter'] += 1
    else:
        return_pack['ID_Flag'].append(0)
        # return return_pack
        # return_pack['rf_counter'] += 1

    dict_ = defaultdict(list)
    for i, item in enumerate(family):
        dict_[item[-8]].append(i)

    comp_recov = None
    for k in dict_:  # key='2' ya '3'
        items = dict_[k]  # [0] ya [1,2] ya [0,1,2] ya [0] ya [2]maslan
        if len(items) == 2:  # age tu items 2 ta item darim pas comp_recov = value(hala ya 2 ya 3)
            # print(items)
            comp_recov = k  # comp_recov = '2' OR '3'
        else:  # age tu items 2 ta nadarim va faghat yek item darim pas in item awal non_comp has!:)
            non_comp = items[0]
        # print('dict_.keys():', dict_.keys())
        # print('dict_.values():', dict_.values())

    """ Blur Energy to find theta_p and theta_e after blurring not from Ground Truth"""
    # mu1, sigma1 = 0.0, 17.35881104
    # for i in range(len(family)):
    #     val = float(family[i][11])
    #     val += np.random.normal(mu1, sigma1)
    #     family[i][11] = str(val)

    if comp_recov is not None:
        # -------------------------------------------------------------------------------------------------------------
        # We label data-set: [Non_Comp index, {compton photo OR photo compton} sorted by time! Smaller time comes first]
        # -------------------------------------------------------------------------------------------------------------
        for i, item in enumerate(dict_[comp_recov]):
            if family[item][-3] == "Compton":  # age awali Compton has pas item ro beriz tu Compton!
                compton = item
                # print('Compton:', compton)
            else:
                pe = item  # age awali compton nabashe mishe photoelectric! pas awali mishe photoelectric!

        a_vector_1 = [float(family[compton][13]) - float(family[non_comp][13]),
                      float(family[compton][14]) - float(family[non_comp][14]),
                      float(family[compton][15]) - float(family[non_comp][15])]
        b_vector_1 = [float(family[pe][13]) - float(family[compton][13]),
                      float(family[pe][14]) - float(family[compton][14]),
                      float(family[pe][15]) - float(family[compton][15])]
        a_vector_1 = np.array(a_vector_1)
        b_vector_1 = np.array(b_vector_1)
        theta_p_1 = vg.angle(a_vector_1, b_vector_1)
        theta_e_1 = icos(1. - 511. * (1 / (float(family[pe][11])) - 1 / (float(family[compton][11])
                                                                         + float(family[pe][11]))))

        a_vector_2 = [float(family[pe][13]) - float(family[non_comp][13]),
                      float(family[pe][14]) - float(family[non_comp][14]),
                      float(family[pe][15]) - float(family[non_comp][15])]
        b_vector_2 = [-float(family[pe][13]) + float(family[compton][13]),
                      -float(family[pe][14]) + float(family[compton][14]),
                      -float(family[pe][15]) + float(family[compton][15])]
        a_vector_2 = np.array(a_vector_2)
        b_vector_2 = np.array(b_vector_2)
        theta_p_2 = vg.angle(a_vector_2, b_vector_2)
        theta_e_2 = icos(1. - 511. * (1 / (float(family[compton][11])) - 1 / (float(family[compton][11]) +
                                                                              float(family[pe][11]))))

        # -------------------------- Provide 2 more possible interaction for Cross-Strip
        p1 = [float(family[compton][13]), float(family[pe][14]),
              float('{:.4f}'.format(float(family[pe][14]) / float(family[compton][13])))]
        p2 = [float(family[compton][14]), float(family[pe][13]),
              float('{:.4f}'.format(float(family[pe][13]) / float(family[compton][14])))]

        a_vector_3 = [p1[0] - float(family[non_comp][13]),
                      p1[1] - float(family[non_comp][14]),
                      p1[2] - float(family[non_comp][15])]

        b_vector_3 = [p2[0] - p1[0],
                      p2[1] - p1[1],
                      p2[2] - p1[2]]
        a_vector_3 = np.array(a_vector_3)
        b_vector_3 = np.array(b_vector_3)
        theta_p_3 = vg.angle(a_vector_3, b_vector_3)
        theta_e_3 = icos(1. - 511. * (1 / (float(family[pe][11])) - 1 / (float(family[compton][11])
                                                                         + float(family[pe][11]))))

        a_vector_4 = [p2[0] - float(family[non_comp][13]),
                      p2[1] - float(family[non_comp][14]),
                      p2[2] - float(family[non_comp][15])]
        b_vector_4 = [-p2[0] + p1[0],
                      -p2[1] + p1[1],
                      -p2[2] + p2[1]]
        a_vector_4 = np.array(a_vector_4)
        b_vector_4 = np.array(b_vector_4)
        theta_p_4 = vg.angle(a_vector_4, b_vector_4)
        theta_e_4 = icos(1. - 511. * (1 / (float(family[compton][11])) - 1 / (float(family[compton][11]) +
                                                                              float(family[pe][11]))))

        # -------------------------- Done: Provide 2 more possible interaction for Cross-Strip

        # return_pack['theta_p_1'].append(theta_p_1)

        # Simplest way to find target sequence
        if float(family[pe][10]) < float(family[compton][10]):
            target_seq_T = [non_comp, pe, compton]
            return_pack['energy_T'].append(float(family[pe][11]))
            return_pack['energy_T'].append(float(family[compton][11]))
            return_pack['target_seq_T'] = target_seq_T
            return_pack['label_T'] = 1
            return_pack['theta_p_T'].append(theta_p_2)
            return_pack['theta_e_T'].append(theta_e_2)
            return_pack['time_T'].append(float(family[pe][10]))
            return_pack['time_T'].append(float(family[compton][10]))

            target_seq_F = [non_comp, compton, pe]
            return_pack['energy_F'].append(float(family[compton][11]))
            return_pack['energy_F'].append(float(family[pe][11]))
            return_pack['target_seq_T'] = target_seq_F
            return_pack['label_F'] = 0
            return_pack['theta_p_F'].extend([theta_p_1, theta_p_3, theta_p_4])
            return_pack['theta_e_F'].extend([theta_e_1, theta_e_3, theta_e_4])
            return_pack['time_F'].append(float(family[compton][10]))
            return_pack['time_F'].append(float(family[pe][10]))
        else:
            target_seq_T = [non_comp, compton, pe]
            return_pack['energy_T'].append(float(family[compton][11]))
            return_pack['energy_T'].append(float(family[pe][11]))
            return_pack['target_seq_T'] = target_seq_T
            return_pack['label_T'] = 1
            return_pack['theta_p_T'].append(theta_p_1)
            return_pack['theta_e_T'].append(theta_e_1)
            return_pack['time_T'].append(float(family[compton][10]))
            return_pack['time_T'].append(float(family[pe][10]))

            target_seq_F = [non_comp, pe, compton]
            return_pack['energy_F'].append(float(family[pe][11]))
            return_pack['energy_F'].append(float(family[compton][11]))
            return_pack['target_seq_F'] = target_seq_F
            return_pack['label_F'] = 0
            return_pack['theta_p_F'].extend([theta_p_2, theta_p_3, theta_p_4])
            return_pack['theta_e_F'].extend([theta_e_2, theta_e_3, theta_e_4])
            return_pack['time_F'].append(float(family[pe][10]))
            return_pack['time_F'].append(float(family[compton][10]))

        # -------------------------------------------------------------------------------------------------------------
        # -------------------------Play with DDA if needed-------------------------------------------------------------
        # -------------------------------------------------------------------------------------------------------------

        DDA1 = abs(theta_p_1 - theta_e_1)
        DDA2 = abs(theta_p_2 - theta_e_2)
        DDA3 = abs(theta_p_3 - theta_e_3)
        DDA4 = abs(theta_p_4 - theta_e_4)

        return_pack['DDA1'].append(DDA1)
        return_pack['DDA2'].append(DDA2)

        if DDA1 < DDA2:
            # print("First")
            pred_seq = [int(non_comp), int(compton), int(pe)]
            return_pack['pred_seq'] = pred_seq

            # print('pred_seq: ', pred_seq, 'DDA1:', DDA1)
        else:
            pred_seq = [int(non_comp), int(pe), int(compton)]
            return_pack['pred_seq'] = pred_seq
            # print('pred_seq: ', pred_seq, 'DDA2: ', DDA2)

        if np.isnan(theta_p_1) or np.isnan(theta_p_2) or np.isnan(theta_p_3) or np.isnan(theta_p_4):
            # print(theta_p_2)
            return return_pack

        return_pack['valid_family'] = True

        return return_pack
    else:
        return_pack['valid_family'] = False
        return return_pack


def main():  # TestShrink50BigBinnedNoBlurred
    with open("TestShrink50BigBinnedNoBlurred.csv",
              'r') as f:  # 3 row data 50BigBinnedNoBlurred3 Pixelated_Data: Sh50GPixNoBlurredTimeNorm
        with open("testfile_output.csv", 'w') as g:
            family = []
            invalid_family_counter = 0
            tf_counter = 0
            rf_counter = 0

            for line in f:
                out = line.rstrip("\r\n")
                if out == "":
                    process = process_family(family)
                    # print('process', process)
                    if process:
                        return_pack = calculate(family)
                        # print('return_pack:', return_pack)
                        if return_pack['valid_family']:
                            # We put true thetas in the 0 index! From time we knew which sequence is true!
                            x_ = [[return_pack['theta_p_T'][0], return_pack['theta_e_T'][0]],
                                  [return_pack['theta_p_F'][0], return_pack['theta_e_F'][0]],
                                  [return_pack['theta_p_F'][1], return_pack['theta_e_F'][1]],
                                  [return_pack['theta_p_F'][2], return_pack['theta_e_F'][2]]]

                            indices = [0, 1, 2, 3]  # list(range(len(x_)))
                            random.shuffle(indices)
                            if return_pack['ID_Flag'][0] == 0:
                                label = max(indices) + 1
                            else:
                                label = indices.index(0)

                            # x = shuffle (temp)
                            x = []
                            for i in indices:
                                x.extend(x_[i])

                            g.write(str(return_pack['ID_Flag'][0]) + "\t" +
                                    str(x[0]) + "\t" + str(x[1]) + "\t" + str(x[2]) + "\t" + str(x[3]) + "\t" +
                                    str(x[4]) + "\t" + str(x[5]) + "\t" + str(x[6]) + "\t" + str(x[7]) + "\t" +
                                    str(label) + "\n")

                        else:
                            invalid_family_counter += 1
                    family = []
                    continue
                else:
                    out = out.strip()
                    items = out.split("\t")
                    if items[-3] != "RayleighScattering":
                        family.append(items)


if __name__ == "__main__":
    main()
