import os
import numpy as np
import pandas as pd

import logging.config

from pathlib import Path

CONFIG = {
    "version": 1,
    "formatters": {
        "simple": {
            "format": "[%(asctime)s - %(name)s - %(levelname)s] %(message)s"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "simple",
            "level": "DEBUG",
            "stream": "ext://sys.stdout"
        }
    },
    "loggers": {
        "processing": {
            "handlers": ["console"],
            "level": "DEBUG"
        }
    }
}

logging.config.dictConfig(CONFIG)
LOGGER = logging.getLogger("processing")


def read_potsdam_file(filename):
    pass
    #csv_df = pandas.read_csv(filename)

def extract_features(dirs):
    pass

#def main():
    #filename = ''
    #read_potd_file(filename)
    #extract_features(["geco/"])

def get_mean_fix_dur(n_fix, total_fix_dur):
    return 0 if n_fix == 0 else total_fix_dur / n_fix

def get_fix_prob(n_fix):
    return int(n_fix > 0)


def get_n_refix(n_fix):
    return max([n_fix - 1, 0])


def get_reread_prob(n_refix):
    return int(n_refix > 0)

def get_n_fixations(filepath):
    nfx_file = pd.read_csv('extract_human_fixations/data/potsdam_fixations/' + filepath, sep="\t", header=0)
    return nfx_file

class PotsdamDataNormalizer:
    def __init__(self, dir, task):
        self.dir = dir
        self.task = task

        self.files = {}  # dict indexed by frag and subj, each entry is a 2-item list containing the relevant fpaths
        self.frags = []
        self.subjs = []
        self.full_subj = 'reader0'
        self.frags_words = {}  # dict indexed by frag, each entry is the list of words
        self.frags_features = {}  # dict indexed by frag, each entry is the list of numpy arrays containing features
        self.flat_words = []
        self.flat_features = []

    def read_files(self):
        LOGGER.info(f"Reading files for task {self.task}")
        for file in sorted(os.listdir(self.dir + "raw")):
            if not file.endswith(".txt"):
                continue
            frag = file.split("_")[1]
            subj = file.split("_")[0]
            fpath = os.path.join(self.dir + "raw", file)
            self.add_file(fpath, frag, subj)

    def add_file(self, fpath, frag, subj):
        if frag not in self.files:
            self.files[frag] = {subj: [fpath]}
        elif subj not in self.files[frag]:
            self.files[frag][subj] = [fpath]
        else:
            self.files[frag][subj].append(fpath)

    def read_frags(self):
        self.frags = [frag for frag in self.files]

    def read_subjs(self):
        self.subjs = [subj for subj in self.files[self.frags[0]]]

    def read_words(self):
        """
        Word list is extracted from the full subject.
        """
        LOGGER.info(f"Reading words for task {self.task}")
        for frag in self.frags:
            self.frags_words[frag] = []
            flt_file = pd.read_csv(self.files[frag][self.full_subj][0], sep="\t", header=0)
            for index, row in flt_file.iterrows():
                if index != len(flt_file)-1:
                    if flt_file.at[index+1, 'SentenceBegin'] == 1:
                        self.frags_words[frag].append(str(row['WORD'])+"<eos>")
                    else:
                        self.frags_words[frag].append(str(row['WORD']))
                else:
                    self.frags_words[frag].append(str(row['WORD']) + "<eos>")


    def calc_features(self):
        LOGGER.info(f"Start of features calculation for task {self.task}")

        for subj in self.subjs:
            csv_data = []
            sentence_idx_counter = 0
            for idx, frag in enumerate(self.frags):
                flt_file = pd.read_csv(self.files[frag][subj][0], sep="\t", header=0)
                tfd = flt_file["TFT"].tolist()
                words = self.frags_words[frag]

                max_sent = flt_file['SentenceIndex'].max()
                for word_idx in range(len(words)):
                    word_row = flt_file.iloc[word_idx]
                    word = word_row['WORD']
                    sentence_idx = word_row['SentenceIndex']
                    all_sentence_idx = sentence_idx_counter + sentence_idx
                    word_idx = word_row['WordIndexInSentence']
                    trt = word_row['FPRT'] + word_row['RRT']
                    tft = word_row['TFT']
                    tft_sum = flt_file[flt_file['SentenceIndex'] == sentence_idx]['TFT'].sum()
                    rel_tft = tft / tft_sum
                    row = [all_sentence_idx, word_idx, word, trt, rel_tft]
                    csv_data.append(row)

                sentence_idx_counter += max_sent

            Path(self.dir+"relfix").mkdir(parents=True, exist_ok=True)
            output_df = pd.DataFrame(data=csv_data, columns=['sentence_id', 'word_id', 'word', 'TRT', 'relFix'])
            output_df.to_csv(self.dir+"relfix/"+subj+'-relfix-feats.csv')


def extract_features(dirs):

    # join results from all subjects
    sent_dict ={}
    for dir in dirs:
        for file in sorted(os.listdir(dir)):
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
    out_file_text = open("results/potsdam_sentences.txt", "w")
    out_file_relFix = open("results/potsdam_relfix_averages.txt", "w")
    for sent, feat in averaged_dict.items():
        print(sent,file=out_file_text)
        print(", ".join(map(str,feat[1])),file=out_file_relFix)


if __name__ == "__main__":
    print(os.listdir())
    normalizer = PotsdamDataNormalizer('extract_human_fixations/data/potsdam/', 'potsdam')
    normalizer.read_files()
    normalizer.read_frags()
    normalizer.read_subjs()
    normalizer.read_words()
    normalizer.calc_features()
    extract_features(['potsdam/'])
