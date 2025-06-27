import datetime
import numpy
import os
import matplotlib.pyplot as plt
from fpdf import FPDF
from fpdf.fonts import FontFace


def make_llm_prompt(irs, impt_peaks, target_fg):
    str_peaks = ''

    for i in range(0, len(impt_peaks)):
        str_peaks += '[{}, {}]'.format(impt_peaks[i][0], impt_peaks[i][1])
        if i < len(impt_peaks) - 1:
            str_peaks += ', '

    prompt = 'The following JSON data is the IR spectrum. {}.'.format(irs.to_json())
    prompt += ' The molecule associated with this IR spectrum contains {}.'.format(target_fg)
    prompt += ' Analyze the IR spectrum based on the {} functional group'.format(target_fg)
    prompt += ' and the absorption peaks in the ranges of {}.'.format(str_peaks)
    prompt += ' Draw a table that summarizes the relevance to {} for the given absorption peaks'.format(target_fg)
    prompt += ' and the reasons of your analysis.'
    prompt += ' Write the relevance in the table only using Yes or No.'

    return prompt


def write_analysis_report(file_name, llm_output, irs):
    raw_text = llm_output.replace('\u2261', '#')

    table_content = extract_table(raw_text)
    if table_content is None:
        raise Exception('Table was not generated.')

    invalid = False
    peak_positions = list()
    try:
        for r in table_content[1:]:
            peak_range = r[0]
            peak_range = peak_range.replace('[', '').replace(']', '')
            peak_range = peak_range.replace('(', '').replace(')', '')
            peak_range = peak_range.replace('cm⁻¹', '')
            peak_range = peak_range.replace('~', '')
            peak_range = peak_range.replace('\u2013', '-')

            if peak_range.find('-') > -1:
                peak_range = peak_range.split('-')
                peak_positions.append([int(peak_range[0]), int(peak_range[1])])
            elif peak_range.find(',') > -1:
                peak_range = peak_range.split(',')
            else:
                invalid = True
                break
            peak_positions.append([int(peak_range[0]), int(peak_range[1])])
    except ValueError:
        invalid = True

    if invalid:
        raise Exception('Parsing error.')

    draw_irs(irs, peak_positions)

    pdf = FPDF(format='A4')
    pdf.add_font('noto-sans', fname='res/NotoSans-Light.ttf', uni=True)
    pdf.set_font('noto-sans', size=14)
    pdf.add_page()
    pdf.set_font('noto-sans', size=14)
    pdf.cell(w=0, h=20, text='IR Spectrum Analysis Report', new_x="LMARGIN", new_y="NEXT")
    pdf.set_font('noto-sans', size=8)
    pdf.cell(w=0, h=4, text='Generator: SSIN-Phi4', new_x="LMARGIN", new_y="NEXT")
    pdf.cell(w=0, h=4, text='Date: {}'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')), new_x="LMARGIN",
             new_y="NEXT")
    pdf.set_draw_color(0, 130, 153)
    pdf.line(x1=11, x2=200, y1=39, y2=39)
    pdf.cell(w=0, h=4, text='', new_x="LMARGIN", new_y="NEXT")

    pdf.set_font(size=8)
    pdf.cell(w=0, h=4, text='Input IR Spectrum', new_x="LMARGIN", new_y="NEXT")
    pdf.image('irs_img_for_writing_analysis_report.png', x=0.1 * pdf.epw, w=0.9 * pdf.epw)
    pdf.multi_cell(w=0, text=raw_text.split('|')[0], new_x='LMARGIN', new_y='NEXT')

    pdf.set_draw_color(0, 0, 0)
    headings_style = FontFace(emphasis='', fill_color=(166, 166, 166))
    with pdf.table(headings_style=headings_style) as table:
        row = table.row()
        for c in table_content[0]:
            row.cell(c)

        for r in table_content[1:]:
            row = table.row()
            for c in r:
                row.cell(c)

    pdf.multi_cell(w=0, text=raw_text.split('|')[-1], new_x='LMARGIN', new_y='NEXT')
    pdf.output(file_name)

    os.remove('irs_img_for_writing_analysis_report.png')


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


def draw_irs(irs, peak_positions):
    plt.figure(figsize=(12, 5))
    plt.rcParams.update({'font.size': 12})
    plt.grid(linestyle='--')
    plt.xlim([550, 3800])
    plt.xticks(numpy.arange(550, 3800 + 1, step=250))
    plt.gca().invert_xaxis()
    plt.xlabel('Wavenumber (cm-1)')
    plt.ylabel('Absorbance')
    plt.plot(irs.wavenumber, irs.absorbance_savgol, c='#747474')

    for pp in peak_positions:
        if pp[0] == pp[1]:
            continue
        ppa = numpy.array(0.5 * (numpy.arange(pp[0], pp[1], step=2) - 550), dtype=int)
        peak_max_idx = ppa[numpy.argmax(irs.absorbance_savgol[ppa])]
        plt.scatter(irs.wavenumber[peak_max_idx], irs.absorbance_savgol[peak_max_idx],
                    c='royalblue', zorder=100)

    plt.tight_layout()
    plt.savefig('irs_img_for_writing_analysis_report.png', bbox_inches='tight', dpi=300)
    plt.close()
