'''
Function for creating histogram representations

Literature References:
    - DOI: Need correct one

Disclaimers:
    - This only works for mdl/sdf type files
    - This is an attempt at the recreation from literature and may not be
      implemented as exactly as it is in the literature source


TODO:
1. Extract feature data
2. Make histograms for each feature
3. Find min/max of histograms
4. Make feature vector component from histogram
'''
import copy
import glob
import numpy as np
from .utils.molecule import Molecule
from .utils.calcs import length
from .utils.calcs import angle
from .utils.calcs import torsion


def hist_maker(dataset):
    '''
    Parameters
    ---------
    dataset: path
        path to all molecules in the dataset

    Returns
    -------
    bond_info: dict
        dict of all bonds and list of corresponding length values in dataset
    angle_info: dict
        dict of all angles and list of corresponding angle values in dataset
    torsion_info: dict
        dict of all torsions and list of corresponding torsion values in dataset
    '''
    # iterate through all of the molecules in the dataset
    #   and get the sizes of the largest bags
    bond_info = {}
    angle_info = {}
    torsion_info = {}
    for mol_file in glob.iglob("{}/*".format(dataset)):
        current_molecule = Molecule(mol_file)
        if current_molecule.ftype != 'sdf':
            raise NotImplementedError(
                'file type \'{}\'  is unsupported. Accepted formats: sdf.'.format(current_molecule.ftype))

        # grab bonds/nonbonds
        for i in range(current_molecule.n_atom):
            for j in range(i, current_molecule.n_atom):
                atomi = current_molecule.sym[i]
                atomj = current_molecule.sym[j]
                zi = current_molecule.at_num[i]
                zj = current_molecule.at_num[j]
                if i != j:
                    if zj > zi:
                        atomi, atomj = atomj, atomi
                    bond = "{}{}".format(atomi, atomj)
                    rij = length(current_molecule, i, j)
                    if bond not in bond_info:
                        bond_info[bond] = [rij]
                    else:
                        bond_info[bond].append(rij)

        # grab angles
        angles = []
        angval = []
        for i in range(current_molecule.n_connect):
            # This is a convoluted way of grabing angles but was one of the
            # fastest. The connectivity is read through and all possible
            # connections are made based on current_molecule.connect.
            # current_molecule.connect then gets translated into
            # current_molecule.sym to make bags based off of atom symbols
            connect = []
            for j in range(current_molecule.n_connect):
                if i in current_molecule.connect[j]:
                    if i == current_molecule.connect[j][0]:
                        connect.append(int(current_molecule.connect[j][1]))
                    elif i == current_molecule.connect[j][1]:
                        connect.append(int(current_molecule.connect[j][0]))
            if len(connect) > 1:
                for k in range(len(connect)):
                    for l in range(k + 1, len(connect)):
                        k_c = connect[k] - 1
                        i_c = i - 1
                        l_c = connect[l] - 1
                        a = current_molecule.sym[k_c]
                        b = current_molecule.sym[i_c]
                        c = current_molecule.sym[l_c]
                        if c < a:
                            # swap for lexographic order
                            a, c = c, a
                        abc = a + b + c
                        ang_theta = angle(current_molecule, k_c, i_c, l_c)
                        angles.append(abc)
                        angval.append(ang_theta)

        for i in range(len(angles)):
            if angles[i] not in angle_info:
                angle_info[angles[i]] = [angval[i]]
            else:
                angle_info[angles[i]].append(angval[i])

        # grab torsions
        # This generates all torsions based on current_molecule.connect
        # not on the current_molecule.sym (atom type)
        tors = []
        for i in range(current_molecule.n_connect):
            # Iterate through the list of connected files and store
            # them as b and c for an abcd torsion
            b = int(current_molecule.connect[i][0])
            c = int(current_molecule.connect[i][1])
            for j in range(current_molecule.n_connect):
                # Join connected values on b of bc to make abc .
                # Below is done twice, swapping which to join on
                # to make sure and get all possibilities
                if int(current_molecule.connect[j][0]) == b:
                    a = int(current_molecule.connect[j][1])
                    # Join connected values on c of abc to make abcd.
                    # Below is done twice, swapping which to join on
                    # to make sure and get all possibilities
                    for k in range(current_molecule.n_connect):
                        if int(current_molecule.connect[k][0]) == c:
                            d = int(current_molecule.connect[k][1])
                            abcd = [a, b, c, d]
                            if len(abcd) == len(set(abcd)):
                                tors.append(abcd)
                    for k in range(current_molecule.n_connect):
                        if int(current_molecule.connect[k][1]) == c:
                            d = int(current_molecule.connect[k][0])
                            abcd = [a, b, c, d]
                            if len(abcd) == len(set(abcd)):
                                tors.append(abcd)
                elif int(current_molecule.connect[j][1]) == b:
                    a = int(current_molecule.connect[j][0])
                    for k in range(current_molecule.n_connect):
                        if int(current_molecule.connect[k][0]) == c:
                            d = int(current_molecule.connect[k][1])
                            abcd = [a, b, c, d]
                            if len(abcd) == len(set(abcd)):
                                tors.append(abcd)
                    for k in range(current_molecule.n_connect):
                        if int(current_molecule.connect[k][1]) == c:
                            d = int(current_molecule.connect[k][0])
                            abcd = [a, b, c, d]
                            if len(abcd) == len(set(abcd)):
                                tors.append(abcd)

        torsions = []
        torval = []
        # This translates all of the torsions from current_molecule.connect
        # to their symbol in order to make bags based upon the symbol
        for i in range(len(tors)):
            a = tors[i][0] - 1
            b = tors[i][1] - 1
            c = tors[i][2] - 1
            d = tors[i][3] - 1
            a_sym = current_molecule.sym[a]
            b_sym = current_molecule.sym[b]
            c_sym = current_molecule.sym[c]
            d_sym = current_molecule.sym[d]
            if d_sym < a_sym:
                # swap for lexographic order
                a_sym, b_sym, c_sym, d_sym = d_sym, c_sym, b_sym, a_sym
            abcd = a_sym + b_sym + c_sym + d_sym
            tor_theta = torsion(current_molecule, a, b, c, d)
            # print(a, b, c, d, a_sym, b_sym, c_sym, d_sym, tor_theta)
            torsions.append(abcd)
            torval.append(tor_theta)
        for i in range(len(torsions)):
            if torsions[i] not in torsion_info:
                torsion_info[torsions[i]] = [torval[i]]
            else:
                torsion_info[torsions[i]].append(torval[i])

    return bond_info, angle_info, torsion_info
