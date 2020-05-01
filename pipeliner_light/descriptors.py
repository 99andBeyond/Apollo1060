import os

from rdkit.Chem import Descriptors, MolToSmiles, AllChem

current_path = os.path.dirname(os.path.abspath(__file__))
descriptors_list = []
with open(os.path.join(current_path, 'files', 'descriptors_list.txt'), "r") as f:
    for line in f:
        descriptors_list.append(line.strip())

descriptors_dict = dict(Descriptors.descList)


def featurize_molecule(mol, features):
    features_list = []
    for feature in features:
        features_list.extend(feat_dict[feature['type']](mol, feature))
    return features_list


def ecfp(molecule, options):
    return [x for x in AllChem.GetMorganFingerprintAsBitVect(
        molecule, options['radius'], options['length'])]


def rdkit_headers():
    headers = [x[0] for x in Descriptors.descList]
    return headers


def fingerprint_headers(options):
    return ['{}{}_{}'.format(options['type'], options['radius'], x) for x in range(options['length'])]


def rdkit_descriptors(molecule, options=None):
    descriptors = []
    for desc_name in descriptors_list:
        try:
            desc = descriptors_dict[desc_name]
            bin_value = desc(molecule)
        except (ValueError, TypeError, ZeroDivisionError) as exception:
            print(
                'Calculation of the Descriptor {} failed for a molecule {} due to {}'.format(
                    str(desc_name), str(MolToSmiles(molecule)), str(exception))
            )
            bin_value = 'NaN'

        descriptors.append(bin_value)

    return descriptors


feat_dict = {"ECFP": ecfp, "DESCS": rdkit_descriptors}
