import uuid
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np



def generate_histogram(column, column_name, path, dtype):
    """
    Generate histogram for given data, save it in directory and return the path
    :param column: Pandas Series data, pd.Series
    :param column_name: name of the column, String
    :param path: Path where files will be stored
    :return: name of the file
    """
    print('Name of your column', column_name)
    print('dtype', dtype)
    print('column data', column[:10])
    plot = None
    plt.figure()
    if dtype == 'object':
        plot = column.value_counts().plot(kind='bar').get_figure()
        code = "plot = column.value_counts().plot(kind='bar').get_figure()"
    else:
        plot = column.plot.hist().get_figure()
        code = "plot = column.plot.hist().get_figure()"
    chart_name = str(uuid.uuid1())
    plot.suptitle('histogram of '+column_name)
    filename = chart_name+'_'+column_name+'_histogram.png'
    plot.savefig(path+'/'+filename)
    return filename, code


def generate_boxplot(column, column_name, path):
    """
    Generate histogram for given data, save it in directory and return the path
    :param column: Pandas Series data, pd.Series
    :param column_name: name of the column, String
    :param path: Path where files will be stored
    :return: name of the file
    """
    plt.figure()
    code = "# code to generate box plot\n"
    print('Name of your column', column_name)
    plot = column.plot.box().get_figure()
    code += "plot = column.plot.box().get_figure()\n"
    chart_name = str(uuid.uuid1())
    filename = chart_name+column_name+'_boxplot.png'
    plot.savefig(path+'/'+filename)
    return filename, code


def generate_correlation(data, column_names, path):
    print('column names', column_names)
    for col in column_names:
        data[col] = data[col].astype(float, errors='ignore')
    # generating code
    code = "# generating gorrelation graph and table\n"
    code += "for column in columns:\n"
    code += "\tdata[column] = data[column].astype(float, errors='ignore')\n"

    correlation_df = data[column_names].corr()
    code += "correlation_df = data[column_names].corr()\n"
    correlation = correlation_df.to_dict()
    code += "correlation = correlation_df.to_dict()\n"
    code += "print('your correlation table', correlation)\n"
    plt.figure()
    code += "plt.figure()\n"
    sns.heatmap(correlation_df, xticklabels=correlation_df.columns, yticklabels=correlation_df.columns, vmin=-1, vmax=1)
    code += "sns.heatmap(correlation_df, xticklabels=correlation_df.columns, yticklabels=correlation_df.columns, vmin=-1, vmax=1)\n"
    plt.tight_layout()
    chart_name = str(uuid.uuid1())
    filename = chart_name+'-'.join(column_names)+'_correlation.png'
    plt.savefig(path+'/'+filename)
    output = []
    column_names.insert(0, '-')
    print('correlation', correlation)
    for col in column_names[1:]:
        row = {'columns': col}
        for col2 in column_names[1:]:
            row[col2] = correlation[col2][col]
        output.append(row)
    return column_names, output, filename, code


def save_plot(plot, path, plot_type=''):
    chart_name = str(uuid.uuid1())
    filename = chart_name + '_' + plot_type + '.png'
    plot.savefig(path+'/'+filename)
    return filename

def generate_atag(url):
    return "<a href='"+url+'"/>'


def apply_pca(data, columns, path):
    print('columns for PCA', columns)
    data = data[(data.values != '?').all(axis=1)]
    data.dropna(inplace=True)
    data = data[columns]
    numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    # convert columns to numeric values if it's possible
    for col in data.columns:
        data[col] = data[col].astype(float, errors='ignore')

    # convert non-numeric columns
    for col in data.columns:
        if col not in numeric_columns:
            le = LabelEncoder()
            le.fit(data[col])
            data[col] = le.transform(data[col])
            print('col', col, 'encoded classes', data[col].unique())

    # convert columns to numeric values if it's possible
    for col in data.columns:
        data[col] = data[col].astype(float, errors='ignore')

    pca = PCA().fit(data)
    plt.figure()
    # plt.hold(False)
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    plt.tight_layout()
    pca_plot = save_plot(plt, path, '_pca_plot_')
    return pca_plot

