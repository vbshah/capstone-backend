import pickle
import uuid
import os
import joblib


def generate_report(report):
    output = []
    cols = ['precision', 'recall', 'f1-score', 'support']
    for key in report:
        row = dict()
        if isinstance(report[key], dict):
            for col in cols:
                row[col] = round(report[key][col], 4)
        else:
            for col in cols:
                row[col] = '-'
            row['f1-score'] = round(report[key], 4)
        output.append(row)
    return output


def generate_rmse_table(rmse_score):
    output = []
    for key in rmse_score:
        output.append({'round': key, 'value': rmse_score[key]})
    return output


def save_model(model, features, model_name, path, encoder_object = {}):
    model_file_name = str(uuid.uuid1()) + '_' + model_name + '.sav'
    meta_object = {'features': features, 'encoder_object': encoder_object}  # In file, we will store both features and model's encoder
    feature_file_name = model_file_name + '_meta.sav'
    joblib.dump(model, os.path.join(path, model_file_name))
    pickle.dump(meta_object, open(os.path.join(path, feature_file_name), 'wb'))
    print('where model is stored', os.path.join(path, model_file_name))
    return model_file_name, feature_file_name
