import numpy as np
from tqdm import tqdm


def transform_text_for_ml(list_of_strings, vocabulary_ids, max_length):
    transformed_data = []

    for string in tqdm(list_of_strings):
        list_of_char = list(string)
        list_of_char_id = [vocabulary_ids[char] for char in list_of_char]

        nb_char = len(list_of_char_id)

        # padding for fixed input length
        if nb_char < max_length:
            for i in range(max_length - nb_char):
                list_of_char_id.append(vocabulary_ids['P'])
        transformed_data.append(list_of_char_id)

    ml_input_training = np.array(transformed_data)

    return ml_input_training
