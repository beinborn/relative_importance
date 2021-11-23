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


def get_mean_fix_dur(n_fix, total_fix_dur):
    return 0 if n_fix == 0 else total_fix_dur / n_fix


def get_fix_prob(n_fix):
    return int(n_fix > 0)


def get_n_refix(n_fix):
    return max([n_fix - 1, 0])


def get_reread_prob(n_refix):
    return int(n_refix > 0)

class RussSentCorpDataNormalizer:
    def __init__(self, dir, task, print_every=3000):
        self.dir = dir
        self.task = task
        self.print_every = print_every

        self.full_subj = "1.edf"

        self.frags = []
        self.subjs = []
        self.frags_words = {}  # dict indexed by frag, each entry is the list of words
        self.frags_features = {}  # dict indexed by frag, each entry is the list of numpy arrays containing features
        self.flat_words = []
        self.flat_features = []

    def read_file(self):
        LOGGER.info(f"Reading file for task {self.task}")
        self.file = pd.read_csv(self.dir + "raw/data_103.csv", sep="\t", header=0)

    def read_frags(self):
        self.frags = sorted(self.file["item.id"].unique())
        print(self.frags)
        print(len(self.frags))

    def read_subjs(self):
        """
        Reads list of subjs and sorts it such that the full subject is in the first position.
        """
        LOGGER.info(f"Reading subjects for task {self.task}")
        self.subjs = sorted(self.file["DATA_FILE"].unique())

        print(self.subjs)
        print(len(self.subjs))

    def read_words(self):
        """
        Word list is extracted from the full subject.
        """

        LOGGER.info(f"Reading words for task {self.task}")

        for f in self.frags:
            self.frags_words[f] = []
            flt_file = self.file
            isfrag = flt_file["item.id"] == f
            issubj = flt_file["DATA_FILE"] == self.full_subj
            flt_file = flt_file[isfrag]
            flt_file = flt_file[issubj]
            flt_file["word.serial.no"] = flt_file["word.serial.no"].astype(str).astype(float)
            flt_file = flt_file.sort_values(by=["word.serial.no"])
            sent = [str(w) for w in flt_file["word.id"].tolist()]
            if sent:
                sent[-1] += "<eos>"
            if not self.frags_words[f]:
                self.frags_words[f] = sent
                #print(f, s, [str(w) for w in flt_file["word.id"].tolist()])

        print(len(self.frags_words))

    def calc_features(self):

        LOGGER.info(f"Start of features calculation for task {self.task}")

        for j, subj in enumerate(self.subjs):
            LOGGER.info(f"Processing subject {j + 1} out of {len(self.subjs)}")
            csv_data = []
            for i, frag in enumerate(self.frags):
                flt_file = self.file
                isfrag = flt_file["item.id"] == frag
                issubj = flt_file["DATA_FILE"] == subj
                flt_file = flt_file[isfrag]
                flt_file = flt_file[issubj]
                flt_file["word.serial.no"] = flt_file["word.serial.no"].astype(str).astype(float)
                flt_file = flt_file.sort_values(by=["word.serial.no"])
                if len(flt_file["word.id"].to_list()) == len(self.frags_words[frag]):
                    for k, w in enumerate(self.frags_words[frag]):
                        nfx = 0.0 if flt_file["IA_FIXATION_COUNT"].to_list()[k] == "NA" else float(flt_file["IA_FIXATION_COUNT"].to_list()[k])
                        ffd = 0.0 if flt_file["IA_FIRST_FIXATION_DURATION"].to_list()[k] == "NA" else float(flt_file["IA_FIRST_FIXATION_DURATION"].to_list()[k])
                        fpd = 0.0 if flt_file["IA_FIRST_RUN_DWELL_TIME"].to_list()[k] == "NA" else float(flt_file["IA_FIRST_RUN_DWELL_TIME"].to_list()[k])
                        tfd = 0.0 if flt_file["IA_DWELL_TIME"].to_list()[k] == "NA" else flt_file["IA_DWELL_TIME"].to_list()[k]
                        word = flt_file["word.id"].to_list()[k]
                        mfd = get_mean_fix_dur(nfx, tfd)
                        fxp = get_fix_prob(nfx)
                        nrfx = get_n_refix(nfx)
                        rrdp = get_reread_prob(nrfx)
                        tfd_sum = flt_file["IA_DWELL_TIME"].sum()
                        rel_tfd = tfd / tfd_sum

                        row = [frag, k, word, tfd, rel_tfd]
                        csv_data.append(row)

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
    out_file_text = open("results/russsent_sentences.txt", "w")
    out_file_relFix = open("results/russsent_relfix_averages.txt", "w")
    for sent, feat in averaged_dict.items():
        print(sent,file=out_file_text)
        print(", ".join(map(str,feat[1])),file=out_file_relFix)


if __name__ == "__main__":
    print(os.listdir())
    normalizer = RussSentCorpDataNormalizer('extract_human_fixations/data/russsent/', 'russsent')
    normalizer.read_file()
    normalizer.read_frags()
    normalizer.read_subjs()
    normalizer.read_words()
    normalizer.calc_features()
    extract_features(['extract_human_fixations/data/russsent/relfix/'])
