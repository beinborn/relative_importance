import argparse
import os
import pandas as pd
import numpy as np

# Extract relative fixation duration from the English part of the GECO corpus

def read_geco_file(reading_data_df, sentence_info_df, task):
    subjects = pd.unique(reading_data_df['PP_NR'].values)
    sentences = pd.unique(sentence_info_df['SENTENCE_ID'].values)

    flat_word_index = 0
    for subj in subjects:
        print(subj)
        subj_data_orig = reading_data_df.loc[reading_data_df['PP_NR'] == subj]
        df_subj = pd.DataFrame(columns=['sentence_id','word_id','word_id_orig','word','TRT','relFix'])
        for j, sent in enumerate(sentences):
            word_ids = sentence_info_df["WORD_ID"].loc[sentence_info_df['SENTENCE_ID'] == sent].values
            tokens = sentence_info_df["WORD"].loc[sentence_info_df['SENTENCE_ID'] == sent].values

            for k, (w, id) in enumerate(zip(tokens, word_ids)):
                trt = subj_data_orig['WORD_TOTAL_READING_TIME'].loc[subj_data_orig['WORD_ID'] == id].values
                # todo: take words with punct?
                #w2 = subj_data_orig['WORD'].loc[subj_data_orig['WORD_ID'] == id].values
                #print(w, id, trt)
                if trt.size > 0:
                    trt = 0 if trt == "." else trt
                else:
                    trt=0
                df_subj.loc[flat_word_index] = [j,k,id,str(w).lower(),float(trt),0]
                flat_word_index += 1

        i = 0
        max_sent = df_subj['sentence_id'].max()
        while i < max_sent:
            sent_data = df_subj.loc[df_subj['sentence_id'] == i]
            try:
                # min-max scale feature  values
                x = [float(s)/sum(sent_data['TRT'].values) for s in sent_data['TRT'].values]
                df_subj.loc[df_subj['sentence_id'] == i, 'relFix'] = x
            except ValueError:
                print(sent_data)
            i += 1
        # write CSV files for each subject
        df_subj.to_csv("data/"+task+"/"+subj+"-relfix-feats.csv")

    print("ALL DONE.")

def extract_features(task):

    # join results from all subjects
    sent_dict ={}
    relfix_dir = "data/" + task + "/relfix/"
    for file in sorted(os.listdir(relfix_dir)):
        print("Reading files for subj ", file)
        subj_data = pd.read_csv(dir+file, delimiter=',')
        max_sent = subj_data['sentence_id'].max()
        print(max_sent, " sentences")

        # join words in sentences
        i = 0
        while i < max_sent:
            sent_data = subj_data.loc[subj_data['sentence_id'] == i]
            if " ".join(map(str, list(sent_data['word']))):
                relfix_vals = list(sent_data['relFix'])
                if " ".join(map(str, list(sent_data['word']))) not in sent_dict:
                    sent_dict[" ".join(map(str, list(sent_data['word'])))] = [list(sent_data['word']), [relfix_vals]]
                else:
                    sent_dict[" ".join(map(str, list(sent_data['word'])))][1].append(relfix_vals)
            i += 1

    # average feature values for all subjects
    averaged_dict = {}
    for sent, features in sent_dict.items():
        avg_rel_fix = np.nanmean(np.array(features[1]), axis=0)
        if len(features[0]) > 1:
            averaged_dict[sent] = [features[0], avg_rel_fix]
    print(len(averaged_dict), " total sentences.")
    out_file_text = open("../results/" + task + "_sentences.txt", "w")
    out_file_relFix = open("../results/" + task + "_relfix_averages.txt", "w")
    for sent, feat in averaged_dict.items():
        print(sent,file=out_file_text)
        print(", ".join(map(str,feat[1])),file=out_file_relFix)


def main():
    # Make sure that this is available
    desc = "Extract relative fixation duration data from the GECO Corpus"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-l', '--language', help="English (en) or Dutch (nl)", default='en')
    args = parser.parse_args()
    lang = args.language
    if lang == 'en':
        reading_data_fn = "data/geco/MonolingualReadingData.xlsx"
        print("Reading file for GECO: ", reading_data_fn)
        reading_data_df = pd.read_excel(reading_data_fn, usecols="A,E,F,I,J,K,BB",
                                        na_filter=False, engine='openpyxl')
        sentence_info_fn = "data/geco/EnglishMaterial_corrected.csv"
        print("Reading file for GECO: ", sentence_info_fn)
        sentence_info_df = pd.read_csv(sentence_info_fn, na_filter=False)
        task =  "geco"
    elif lang == 'nl':
        reading_data_fn = "data/geco_nl/L1ReadingData.xlsx"
        print("Reading file for GECO: ", reading_data_fn)
        reading_data_df = pd.read_excel(reading_data_fn, na_filter=False,
                                        usecols="A,E,F,I,J,K,BB", engine='openpyxl')
        sentence_info_fn = "data/geco_nl/DutchMaterials.xlsx"
        print("Reading file for GECO: ", sentence_info_fn)
        sentence_info_df = pd.read_excel(sentence_info_fn, na_filter=False,
                                         engine='openpyxl')
        sentence_info_df.rename({'IA_ID':'WORD_ID'}, axis='columns', inplace=True)
        task =  "geco_nl"
    else:
        raise ValueError("Language '" + lang + "' is not supported.")
    read_geco_file(reading_data_df, sentence_info_df, task)
    extract_features(task)

if __name__ == "__main__":
    main()
