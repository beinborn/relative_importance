import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd
import seaborn as sns
from wordfreq import word_frequency
import spacy.tokens


def process_tokens(all_tokens):
    nlp = spacy.load('en_core_web_md')

    pos_tags = []
    frequencies = []

    for tokens in all_tokens:
        doc = spacy.tokens.doc.Doc(
            nlp.vocab, words=tokens)
        # run the pos tagger
        processed = nlp.tagger(doc)
        sentence_tags = [token.pos_ for token in processed]
        sentence_frequencies = [word_frequency(token.lemma_, 'en') for token in processed]
        pos_tags.append(sentence_tags)
        frequencies.append(sentence_frequencies)

    return pos_tags, frequencies


def visualize_frequencies(et_frequencies, human_saliency, lm_frequencies, machine_saliency, outfile):
    human_data = pd.DataFrame({"Frequency": et_frequencies, "Importance": human_saliency})
    model_data = pd.DataFrame({"Frequency": lm_frequencies, "Importance": machine_saliency})

    all_data = pd.concat([human_data.assign(dataset='Model'), model_data.assign(dataset='Human')])

    mypalette = sns.diverging_palette(150, 275, s=80, l=55, n=2)
    sns.stripplot(x='Frequency', y='Importance', data=all_data,
                  hue='dataset', dodge=True, palette=mypalette)

    print("Overall correlation: ")
    print("Human - Frequency")
    print(scipy.stats.spearmanr(et_frequencies, human_saliency)[0])
    print("Model - Frequency")
    print(scipy.stats.spearmanr(lm_frequencies, machine_saliency)[0])
    plt.legend(loc='upper right')

    plt.xlabel("Frequency")
    plt.ylabel("Relative Importance")
    plt.savefig(outfile, bbox_inches="tight")
    plt.close()


def visualize_lengths(et_tokens, human_saliency, lm_tokens, machine_saliency, outfile):
    et_lengths = [len(token) for token in et_tokens]
    lm_lengths = [len(token) for token in lm_tokens]
    human_data = pd.DataFrame({"Length": et_lengths, "Importance": human_saliency})
    model_data = pd.DataFrame({"Length": lm_lengths, "Importance": machine_saliency})

    all_data = pd.concat([human_data.assign(dataset='Model'), model_data.assign(dataset='Human')])
    mypalette = sns.diverging_palette(150, 275, s=80, l=55, n=2)
    sns.stripplot(x='Length', y='Importance', data=all_data,
                  hue='dataset', dodge=True, palette=mypalette)

    print("Overall correlation: ")
    print("Human - Length")
    print(scipy.stats.spearmanr(et_lengths, human_saliency)[0])
    print("Model - Length")
    print(scipy.stats.spearmanr(lm_lengths, machine_saliency)[0])
    plt.legend(loc='upper right')

    plt.xlabel("Length")
    plt.ylabel("Relative Importance")
    plt.savefig(outfile, bbox_inches="tight")
    plt.close()


def visualize_posdistribution(tag2importance, outfile):
    means = []
    stds = []


    function_word_tags = ["ADP", "AUX", "CCONJ", "DET", "NUM", "PART", "PRON", "SCONJ"]
    other_tags = ["PUNCT", "SYM", "X"]
    # Selection and order can be changed for the plot
    labels = ["ADJ", "ADV", "NOUN", "VERB", "INTJ", "ADP", "AUX", "CONJ", "DET", "PART", "PRON", "NUM"]
    for label in labels:
        if label not in other_tags:
            if label == "CONJ":
                values = tag2importance["CCONJ"] + tag2importance["SCONJ"]
            else:
                values = tag2importance[label]
            #print(label, len(values))
            mean = np.nanmean(values)
            std = np.nanstd(values)

            means.append(mean)
            stds.append(std)

            #print(f"Mean: {mean:.4f}, {std:.4f}")

    data = pd.DataFrame({"PosTag": labels, "Mean": means, "Std": stds})

    sns.set(font_scale=2)
    sns.catplot(x="PosTag", y="Mean", data=data, kind="bar", height=7, aspect=2)
    plt.ylim(0.0, 0.055)
    plt.xlabel("")
    plt.ylabel("Relative Importance")

    # Save the figure and show
    plt.savefig(outfile, bbox_inches="tight")
    plt.close()


def calculate_saliency_by_wordclass(pos_tags, saliencies):

    tag2importance = defaultdict(list)

    for i, tags in enumerate(pos_tags):

        # if i % 500 == 0:
        #     print(i, len(pos_tags))
        try:
            if not len(tags) == len(saliencies[i]):
                pass

            else:
                salience = saliencies[i]

                for k, tag in enumerate(tags):
                    # Does it make sense to normalize importance by length? Not sure
                    # The values get very small then
                    try:
                        tag2importance[tag].append(salience[k] / len(tags))
                    except:
                        print("Tokenisation ERROR!: ")
                        continue
        except TypeError:
            pass
    return tag2importance


def visualize_sentence(i, et_tokens, human_saliency, lm_saliency, outfile):

    if i == 153:
        # hardcoded for better looking plot
        tokens = ["Oh,", "Sherlock", "Holmes", "by", "all", "means."]
    else:
        if et_tokens[i] == len(human_saliency[i]) == len(lm_saliency[i]):
            tokens = et_tokens[i]
        else:
            print("Mismatched tokenisation for sentence", i)

    human_data = pd.DataFrame({"Tokens": tokens, "Importance": human_saliency[i]})
    model_data = pd.DataFrame({"Tokens": tokens, "Importance": lm_saliency[i]})

    all_data = pd.concat([human_data.assign(dataset='Model'), model_data.assign(dataset='Human')])
    fig, ax = plt.subplots(figsize=(8, 4))
    mypalette = sns.diverging_palette(150, 275, s=80, l=55, n=2)
    sns.set(font_scale=2)
    sns.lineplot(x="Tokens", y="Importance", data=all_data, hue="dataset", style="dataset", markers=False,
                 dashes=[(3, 3), (3, 3)], palette=mypalette)
    sns.scatterplot(x="Tokens", y="Importance", data=all_data, hue="dataset", size="Importance", markers=["o", "o"],
                    legend=False, sizes=(50, 300), palette=mypalette)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels)
    plt.savefig(outfile)
    plt.close()



def flatten(mylist):
    return [item for sublist in mylist for item in sublist]


def flatten_saliency(mylist):
    flattened_salience = []
    for salience in mylist:
        normalized_salience = [s / len(salience) for s in salience]
        flattened_salience.extend(normalized_salience)
    return flattened_salience
