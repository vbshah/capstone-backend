import pandas as pd
import os
import numpy as np
from multiprocessing import Pool, freeze_support


def allowed_file(filename, allowd_types):
    print('filename', filename.rsplit('.', 1)[1].lower())
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowd_types


def get_file(filename, path, delimiter='\t'):
    """
    Returns whole file
    :param filename: file-key
    :param path: path to the file
    :param delimiter: how you want to split the file
    :return: dataframe of the file
    """
    extension = filename.rsplit('.', 1)[1].lower()
    if extension == 'csv':
        return pd.read_csv(os.path.join(path, filename))


def calculate_meta(data):
    code = '# generating information about CSV file\n'
    output = dict()
    output['columns'] = list(data.columns)
    data_types = dict()
    mean = dict()
    median = dict()
    mode = dict()
    for column in data.columns:
        data_type = str(data[column].dtype)
        if data_type == 'object':
            mode[column] = data[column].mode()[0]
            overall_type = 'categorical'
        else:
            mean[column] = data[column].mean()
            median[column] = data[column].median()
            overall_type = 'numerical'
        data_types[column] = data_type + '/' + overall_type

    output['dataTypes'] = data_types
    shape = data.shape
    output['shape'] = {'rows': shape[0], 'columns': data.shape[1]}
    output['missing'] = na_count(data)
    output['mean'] = mean
    output['median'] = median
    output['mode'] = mode
    print('missing', output['missing'])

    # now generating code for what we have done
    code = '# generating information about CSV file\n'
    code += "output = data.describe().to_dict()  # getting description of file and into Python dictionary\n"
    code += "for column in output.keys():\n"
    code += "\tdata_type = str(data[column].dtype)\n"
    code += "\tif data_type == 'object':\n"
    code += "\t\toverall_type = 'categorical'\n"
    code += "\telse:\n"
    code += "\t\toverall_type = 'numerical\n"
    code += "\toutput[column]['data_type'] = overall_type\n"
    code += "\toutput[column]['missing'] = data[column].isna().sum()\n"
    code += "print('overall stats', output)\n\n"
    return output, code


def na_count(data):
    print('type of data', type(data))
    missing_list = dict()
    cols = list(data.columns)
    for col in cols:
        print('counting for col', col)
        count = 0
        temp = data[col]
        for i, v in temp.items():
            if v in ('-', '?', 'na', 'n/a', 'NA', 'N/A'):
                count += 1
        print('column data', data[col].isnull())
        count += len(np.where(data[col].isnull())[0])
        missing_list[col] = count

    return missing_list

