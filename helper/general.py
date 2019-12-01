import pandas as pd
import pymongo


def add_code(file_key, file_collection, new_line):
    """
    Add code to given filekey in mongodb
    :param file_key: file-key in mongoDB
    :param file_collection: file-collection in mongoDB
    :param new_line: new code
    :return: None
    """
    new_val = {'$push': {'code': new_line+'\n'}}
    query = {'filename': file_key}
    file_collection.update_one(query, new_val)
    updated_file = file_collection.find(query)
    print('new code', list(updated_file)[0]['code'])


