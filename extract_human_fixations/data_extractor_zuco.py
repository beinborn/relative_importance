import os
import h5py
import pandas as pd
import numpy as np

def fix_punctuation(words, feats):
    punct = ["..", "...", ".", ",", "?", "!", ";", ")", ":", "'", "(", '``', '`', '-']
    clean_words = []; clean_feats = []
    #print(words)
    for w, f in zip(words, feats):
        if w not in punct:
            #print(w)
            w = w.strip("'")
            w = w.strip("`")
            w = w.strip('"')
            w = w.strip('[')
            w = w.strip(']')
            clean_words.append(w)
            clean_feats.append(f)
    #print(clean_words)
    return clean_words, clean_feats

def read_zuco_files(dir, task):

    stopchars = (".", ",", "?", "!", ";", ")", ":", "'")
    beginchars = ("(", '``', '`')

    print("Reading files for task ", task)
    for file in sorted(os.listdir(dir)):
        if file.endswith("_"+task+".mat"):
            subj = file.split("_")[0][-3:]
            fpath = os.path.join(dir, file)
            print(file)
            print("Processing subject: ", subj)

            df_subj = pd.DataFrame(columns=['sentence_id','word_id','word','TRT','relFix'])

            flat_word_index = 0
            matlab_file = h5py.File(fpath)
            sentence_data = matlab_file["sentenceData/word"]
            for j, row in enumerate(sentence_data):
                # test if there is results available for this sentence
                try:
                    word_content_data = matlab_file[row[0]]["content"]
                    word_trt_data = matlab_file[row[0]]["TRT"]

                    for k, (w, f) in enumerate(zip(word_content_data, word_trt_data)):
                        word = u"".join(chr(c) for c in matlab_file[w[0]].value)
                        trt = matlab_file[f[0]].value[0]
                        #word = word+"<EOS>" if k == len(word_content_data)-1 else word
                        #print(word, trt)

                        #tokenize correctly for downstream processing
                        # more end of word: "..."
                        if word == "...":
                            df_subj.loc[flat_word_index] = [j,k+1,".",0.0,0]
                            flat_word_index += 1
                            df_subj.loc[flat_word_index] = [j,k+1,".",0.0,0]
                            flat_word_index += 1
                            df_subj.loc[flat_word_index] = [j,k+1,".",0.0,0]
                            flat_word_index += 1
                        if len(word)>3 and word.endswith("..."):
                            new_word = word[:-3]
                            punct = word[-1]
                            df_subj.loc[flat_word_index] = [j,k,new_word.lower(),float(trt),0]
                            flat_word_index += 1
                            df_subj.loc[flat_word_index] = [j,k+1,punct,0.0,0]
                            flat_word_index += 1
                            df_subj.loc[flat_word_index] = [j,k+1,punct,0.0,0]
                            flat_word_index += 1
                            df_subj.loc[flat_word_index] = [j,k+1,punct,0.0,0]
                            flat_word_index += 1
                        elif word.endswith(stopchars):
                            new_word = word[:-1]

                            punct = word[-1]
                            df_subj.loc[flat_word_index] = [j,k,new_word.lower(),float(trt),0]
                            flat_word_index += 1
                            df_subj.loc[flat_word_index] = [j,k+1,punct,0.0,0]
                            flat_word_index += 1
                        elif "'" in word:
                            tok1, punt, tok2 = word.partition["'"]
                            df_subj.loc[flat_word_index] = [j,k,tok1.lower(),float(trt),0]
                            flat_word_index += 1
                            df_subj.loc[flat_word_index] = [j,k+1,punct,0.0,0]
                            flat_word_index += 1
                            df_subj.loc[flat_word_index] = [j,k,tok2.lower(),float(trt),0]
                        elif "/" in word:
                            tok1, punt, tok2 = word.partition["'"]
                            df_subj.loc[flat_word_index] = [j,k,tok1.lower(),float(trt),0]
                            flat_word_index += 1
                            df_subj.loc[flat_word_index] = [j,k+1,punct,0.0,0]
                            flat_word_index += 1
                            df_subj.loc[flat_word_index] = [j,k,tok2.lower(),float(trt),0]
                        elif word.endswith(beginchars):
                            new_word = word[1:]
                            #print(new_word)
                            punct = word[0]
                            df_subj.loc[flat_word_index] = [j,k+1,punct,0.0,0]
                            flat_word_index += 1
                            df_subj.loc[flat_word_index] = [j,k,new_word.lower(),float(trt),0]
                            flat_word_index += 1
                        else:
                            df_subj.loc[flat_word_index] = [j,k,word.lower(),float(trt),0]
                            flat_word_index += 1
                except:
                    continue

            i = 0
            max_sent = df_subj['sentence_id'].max()
            while i < max_sent:
                sent_data = df_subj.loc[df_subj['sentence_id'] == i]
                try:
                    x = [float(s)/sum(sent_data['TRT'].values) for s in sent_data['TRT'].values]
                    df_subj.loc[df_subj['sentence_id'] == i, 'relFix'] = x
                except ValueError:
                    print(sent_data)
                i += 1
            df_subj.to_csv("zuco2_"+task+"/"+subj+"-relfix-feats.csv")
    print("ALL DONE.")

def extract_features(dirs):

    # join results from all subjects
    sent_dict ={}
    for dir in dirs:
        print("-----")
        print(dir)
        for file in sorted(os.listdir(dir)):
            print("Reading files for subj ", file)
            subj_data = pd.read_csv(dir+file, delimiter=',')
            max_sent = subj_data['sentence_id'].max()
            print(max_sent, " sentences")

            # join words in sentences
            i = 0
            while i < max_sent:
                sent_data = subj_data.loc[subj_data['sentence_id'] == i]
                #print(sent_data)

                if " ".join(list(sent_data['word'])):
                    relfix_vals = list(sent_data['relFix'])
                    if " ".join(list(sent_data['word'])) not in sent_dict:
                        sent_dict[" ".join(list(sent_data['word']))] = [list(sent_data['word']), [relfix_vals]]
                    else:
                        sent_dict[" ".join(list(sent_data['word']))][1].append(relfix_vals)
                i += 1

    # average feature values for all subjects
    averaged_dict = {}
    for sent, features in sent_dict.items():
        avg_rel_fix = np.nanmean(np.array(features[1]), axis=0)
        avg_ref_fix_min = np.nanmin(np.array(features[1]), axis=0)
        avg_ref_fix_max = np.nanmax(np.array(features[1]), axis=0)
        avg_ref_fix_std = np.nanstd(np.array(features[1]), axis=0)

        if len(features[0]) > 1:
            feat, avg_rel_fix = fix_punctuation(features[0], avg_rel_fix)
            #print(feat)
            sent = " ".join(feat)
            averaged_dict[sent] = [feat, avg_rel_fix]
            print(sent)
            print(features[1])
            #print(avg_rel_fix,avg_ref_fix_min,avg_ref_fix_max,avg_ref_fix_std)
    print(len(averaged_dict), " total sentences.")

    out_file_text = open("../results/zuco_sentences.txt", "w")
    out_file_relFix = open("../results/zuco_relfix_averages.txt", "w")
    for sent, feat in averaged_dict.items():
        print(sent,file=out_file_text)
        print(", ".join(map(str,feat[1])),file=out_file_relFix)


def main():
    dir = "data/zuco/"
    task = "NR"
    read_zuco_files(dir, task)
    extract_features(["zuco1_SR/", "zuco1_NR/", "zuco2_NR/"])

if __name__ == "__main__":
    main()
