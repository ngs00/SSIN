def extract_intro(text):
    intro = text.split('---------------------------------------------------------------------------')[1]
    intro = intro.split('|')[0]

    return intro


def extract_table(text):
    table = list()

    tok = text.split('|---')
    header = tok[0].split('|')[1:-1]
    data = None
    for t in tok[1:]:
        if t.find('\n') > -1:
            data = t
    if data is None:
        return None

    data_row = data.split('\n')[1:]

    table.append([h.strip() for h in header])
    for row in data_row:
        cols = [c for c in row.split('|') if c != '']
        if len(cols) <= 1:
            continue
        table.append([c.strip() for c in cols])

    return table


def extract_conc(text):
    return text.split('|')[-1]