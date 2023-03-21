import pandas as pd
import os
from configuration import param


def load_dataset(path):
    """
    Load the dataframe using Pandas read_csv

    Parameter
    ---------
    path : str
        The path of the dataset
    
    Return
    -------
    dataframe
    """
    dataframe = pd.read_csv(path, names=['Timestamp', 'CAN_ID', 'DLC', 'DATA[0]', 'DATA[1]', 'DATA[2]',
                                         'DATA[3]', 'DATA[4]', 'DATA[5]', 'DATA[6]', 'DATA[7]', 'Flag'], header=None)
    return dataframe


def clear_dataset(dataframe):
    """
    - Clear the CAN ID "0316" became "316".
    - Clear the rows where the packets are not legth 8

    Parameter
    ---------
    dictionary_dataframe : pandas.DataFrame
        The dictionary_dataframe that needs to be processed
    
    Return
    ------
    dictionary_dataframe
    """
    dataframe['CAN_ID'] = dataframe['CAN_ID'].str[1:]
    dataframe = dataframe[dataframe['Flag'].isnull() == False]
    dataframe.reset_index(drop=True, inplace=True)
    return dataframe


def convert_data_features(dataframe):
    """
    Converts the dictionary_data features from hexadecimal to decimal format

    Parameter
    ---------
    dataframe : pandas.DataFrame
        The dataframe that needs to be processed
    
    Return
    ------
    dataframe
    """
    for i in range(0, 8):
        dataframe['DATA[' + str(i) + ']'] = dataframe['DATA[' + str(i) + ']'].apply(int, base=16)
    return dataframe


dictionary_frequency_normal = dict()
dictionary_frequency_attack = dict()


def calculate_frequency(row):
    """
    Calculate the frequency of the CAN_ID with the Timestamp.
    Parameters
    ----------
    row each rows of the dataset processed

    Returns
    -------
    freq : the frequency calculated
    bin_st[0] to bin_st[11] the binary form of CAN_ID
    """
    global dictionary_frequency_normal
    global dictionary_frequency_attack

    if row.Flag == "R":
        if row.CAN_ID not in dictionary_frequency_normal:
            dictionary_frequency_normal[row.CAN_ID] = row.Timestamp
            freq = 0
        else:
            # calculate the frequency and convert it in milliseconds
            freq = (row.Timestamp - dictionary_frequency_normal[row.CAN_ID]) * 1000
            dictionary_frequency_normal[row.CAN_ID] = row.Timestamp
    else:
        if row.CAN_ID not in dictionary_frequency_attack:
            dictionary_frequency_attack[row.CAN_ID] = row.Timestamp
            freq = 0
        else:
            freq = (row.Timestamp - dictionary_frequency_attack[row.CAN_ID]) * 1000
            dictionary_frequency_attack[row.CAN_ID] = row.Timestamp
    # convert the CAN ID into binary form
    # bin_st = bin(int(row.CAN_ID, 16))[2:].zfill(12)
    bin_st = bin(int('1' + row.CAN_ID, 16))[3:]

    return freq, bin_st[0], bin_st[1], bin_st[2], bin_st[3], bin_st[4], bin_st[5], bin_st[6], bin_st[7], bin_st[8], \
           bin_st[9], bin_st[10], bin_st[11]


dictionary_data = dict()


def feature_transformation(row):
    """
    Calculates the Delta and the Frequency attributes.

    Parameter
    ---------
    row : str
        Take the row of the dataframe
    
    Return
    ------
    
    """
    global dictionary_data

    def transform_data_values(dictionary, row_processed):
        """
        Internal function to perform the assigned to the DATA values
        Parameters
        ----------
        dictionary : dictionary can be normal or attack. It depends on the type of dataset that is processed
        row_processed : The row processed

        Returns
        -------
        Dictionary processed
        """
        for data_packet in range(0, 8):
            dictionary[row_processed.CAN_ID]['DATA[' + str(data_packet) + ']'] = row['DATA[' + str(data_packet) + ']']
        return dictionary

    
    if row.CAN_ID not in dictionary_data:
        dictionary_data[row.CAN_ID] = {}
        dictionary_data = transform_data_values(dictionary_data, row)
        total_delta = 0
        delta_dif = [0, 0, 0, 0, 0, 0, 0, 0]  # delta values refer to DATA[0]-DATA[7]

    else:
        delta_dif = []
        for i in range(0, 8):
            result = abs(dictionary_data[row.CAN_ID]['DATA[' + str(i) + ']'] - row['DATA[' + str(i) + ']'])
            delta_dif.append(result)
        total_delta = sum(delta_dif)
        dictionary_data = transform_data_values(dictionary_data, row)

    dif0, dif1, dif2, dif3, dif4, dif5, dif6, dif7 = delta_dif

    return total_delta, dif0, dif1, dif2, dif3, dif4, dif5, dif6, dif7


def init_preprocessing():
    """
    Initialize the preprocessing
    Return
    ------
    concatenated_dataset the dataset concatenated

    """
    dataset_list = ['DoS_dataset', 'RPM_dataset', 'Fuzzy_dataset', 'gear_dataset']
    concatenated_dataset = None
    i = 1

    for dataset_processing in dataset_list:
        global dictionary_data
        global dictionary_frequency_attack
        global dictionary_frequency_normal
        dictionary_frequency_normal.clear()
        dictionary_frequency_attack.clear()
        dictionary_data.clear()

        print(f"[+] Processing the dataset: {dataset_processing}")

        # T: 1 DoS T: 2 RPM T: 3 Fuzzy T: 4 Gear
        dataset_path = os.path.join(param.get('path_dir_dataset'), dataset_processing + '.csv')
        dataset_processing = load_dataset(dataset_path)
        dataset_processing = clear_dataset(dataset_processing)
        dataset_processing = convert_data_features(dataset_processing)

        print("[+] The data are converted.")

        dataset_processing[['FREQUENCY', 'CAN_ID_0', 'CAN_ID_1', 'CAN_ID_2', 'CAN_ID_3', 'CAN_ID_4',
                            'CAN_ID_5', 'CAN_ID_6', 'CAN_ID_7', 'CAN_ID_8', 'CAN_ID_9', 'CAN_ID_10',
                            'CAN_ID_11']] = dataset_processing.apply(
            calculate_frequency, axis=1, result_type="expand")
        dataset_processing.reset_index(drop=True, inplace=True)
        print("[+] The frequency has been calculated.")

        dataset_processing[
            ['DELTA_TOT', 'DELTA_DATA[0]', 'DELTA_DATA[1]', 'DELTA_DATA[2]', 'DELTA_DATA[3]', 'DELTA_DATA[4]',
                'DELTA_DATA[5]', 'DELTA_DATA[6]', 'DELTA_DATA[7]']] = dataset_processing.apply(
            feature_transformation, axis=1, result_type="expand")
        dataset_processing.reset_index(drop=True, inplace=True)

        dataset_processing = dataset_processing[
            ['Timestamp', 'FREQUENCY', 'DELTA_TOT', 'DELTA_DATA[0]', 'DELTA_DATA[1]', 'DELTA_DATA[2]',
                'DELTA_DATA[3]', 'DELTA_DATA[4]', 'DELTA_DATA[5]', 'DELTA_DATA[6]', 'DELTA_DATA[7]', 'CAN_ID',
                'CAN_ID_0', 'CAN_ID_1', 'CAN_ID_2', 'CAN_ID_3', 'CAN_ID_4', 'CAN_ID_5', 'CAN_ID_6', 'CAN_ID_7',
                'CAN_ID_8', 'CAN_ID_9', 'CAN_ID_10', 'CAN_ID_11', 'DLC', 'DATA[0]', 'DATA[1]', 'DATA[2]',
                'DATA[3]', 'DATA[4]', 'DATA[5]', 'DATA[6]', 'DATA[7]', 'Flag']]

        dataset_processing['Flag'] = dataset_processing['Flag'].map({'R': 0, 'T': i})

        index = dataset_processing.index[dataset_processing['Flag'] == 0].tolist()
        index = index[0:1900000]
        dataset_processing.drop(dataset_processing.index[index], inplace=True)
        dataset_processing.reset_index(drop=True, inplace=True)
        
        print("[+] Dataset is processed")
        if concatenated_dataset is None:
            concatenated_dataset = pd.concat([dataset_processing], axis=0)
        else:
            concatenated_dataset = pd.concat([concatenated_dataset, dataset_processing], axis=0)
        i += 1

    concatenated_dataset = concatenated_dataset.drop('Timestamp', axis=1)
    concatenated_dataset = concatenated_dataset.drop('CAN_ID', axis=1)
    concatenated_dataset = concatenated_dataset.drop('DLC', axis=1)

    if param.get('save'):
        concatenated_dataset.to_csv(param.get('path_to_save_dataset'), index=False)

    return concatenated_dataset
