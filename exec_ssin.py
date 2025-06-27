import ollama
from util.data import load_ref_dataset, read_jdx_file
from method.model import *
from util.analysis_report import *


# Set the target functional group.
target_fg = 'alkyl halide'


# Load reference spectra of SSIN.
# Set ``path_jdx`` to the path of the JDX files in your machine.
print('Load a reference dataset.')
dataset_ref = load_ref_dataset(path_metadata='res/metadata_func_group_ref.xlsx',
                               path_jdx='../../data/chem_data/ir/nist/jdx',
                               target_substruct=target_fg)


# Load the trained SSIN model.
model = SSIN(dim_emb=128, len_spect=dataset_ref.len_spect, ref_db=dataset_ref).cuda()
model.load_state_dict(torch.load('save/model/{}/model_0.pt'.format(target_fg)))


# Load the input IR spectrum.
# Set ``file_name`` to the path of the input IR spectrum.
irs = read_jdx_file(file_name='../../data/chem_data/ir/nist/jdx/C50793_0.jdx', norm_y=True, wmin=550, wmax=3801)


# Execute SSIN to calculate a detection result and set of important peaks.
print('Execute the trained SSIN.')
preds, attns = predict_from_jdx(model, irs)
impt_peaks = identify_impt_peaks(irs, attns, target_fg)


# Execute LLM to analyze the set of important peaks in terms of the target functional group.
print('Execute LLM.')
llm_prompt = make_llm_prompt(irs, impt_peaks, target_fg)
answer = ollama.generate(model='phi4:latest', prompt=llm_prompt)


# Make a report based on the reasoning results of SSIN-LLM.
write_analysis_report('analysis_report.pdf', answer['response'], irs)
print('The IR spectrum analysis report has been successfully generated.')
