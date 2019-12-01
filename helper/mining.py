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
