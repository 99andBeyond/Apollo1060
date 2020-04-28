import pandas as pd
from tqdm import tqdm

from pipeliner_light.pipelines import ClassicPipe
from pipeliner_light.smol import SMol

if __name__ == "__main__":

    predictions_list = []

    ccr5_pipe = ClassicPipe.load('Models/hiv_ccr5')
    int_pipe = ClassicPipe.load('Models/hiv_int')
    rt_pipe = ClassicPipe.load('Models/hiv_rt')

    with open('Data/ChEMBL_500_sample.txt', 'r') as inp:
        smiles_list = inp.readlines()

    for sml_str in tqdm(smiles_list):
        smiles = sml_str.strip()
        smol = SMol(smiles)  # standardization
        smol.featurize(ccr5_pipe.features)  # same intital features set before per-model selection
        predicted_ccr5_pic_50 = ccr5_pipe.predict_vector(smol.features_values)
        predicted_int_pic_50 = int_pipe.predict_vector(smol.features_values)
        predicted_rt_pipe_pic_50 = rt_pipe.predict_vector(smol.features_values)
        predictions_list.append([smiles,
                                 predicted_ccr5_pic_50,
                                 predicted_int_pic_50,
                                 predicted_rt_pipe_pic_50])

    df = pd.DataFrame(predictions_list, columns=['SMILES', 'CCR5_pIC50', 'INT_pIC50', 'RT_pIC50'])
    df.to_csv('ChEMBL_500_sample_predicted_example.csv', index=False)
