import numpy


def load_irs_table(path_table):
    table_dict = dict()

    with open(path_table, 'r') as f:
        for line in f.readlines():
            toks = line.replace('\n', '').split(',')
            wn = toks[0].split('-')
            fg = toks[-1]

            if fg not in table_dict.keys():
                table_dict[fg] = list()

            if len(wn) == 2:
                table_dict[fg].append([int(wn[1]), int(wn[0]), toks[1]])
            else:
                table_dict[fg].append([int(wn[0]), int(wn[0]), toks[1]])

    return table_dict


def cluster_wn(wn):
    peaks = list()
    peak = [wn[0]]

    for i in range(1, wn.shape[0]):
        if wn[i] - wn[i - 1] < 100:
            peak.append(wn[i])
        else:
            if len(peak) == 0:
                peak = [wn[i - 1]]
            peaks.append(peak)
            peak = [wn[i]]
    peaks.append(peak + [wn[-1]])

    for i in range(0, len(peaks)):
        peaks[i] = [numpy.min(peaks[i]), numpy.max(peaks[i])]

    return peaks


def is_explicit_case(peak_wn, table, shift_coeff):
    valid = [False for _ in range(0, len(table))]

    for i in range(0, len(table)):
        for wn in peak_wn:
            if table[i][0] - shift_coeff <= wn <= table[i][1] + shift_coeff:
                valid[i] = True
                break

    if numpy.sum(valid) == len(table):
        return True
    else:
        return False
