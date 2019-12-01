from sklearn.preprocessing import StandardScaler


# we dont need to return anything, Dataframe is passed as reference
def apply_std_scaling(data, columns):
    scalar = StandardScaler()
    code = "scalar = StandardScaler()\n"
    data[columns] = scalar.fit_transform(data[columns])
    code += "data[columns] = scalar.fit_transform(data[columns])\n"
    return code
