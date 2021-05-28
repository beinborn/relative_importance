import sklearn
import numpy as np
import scipy.stats
import random


# Here we calculate length, frequency, and permutation baselines on the sentence level
# Note that the correlation functions yield a warning if one of the list is constant (stdev = 0)
# For example, the phrase "you did not" would yield the length vector [3,3,3]
# and then correlation cannot be calculated
def calculate_len_baseline(tokens, importance):
    spearman = []
    kendall = []
    mi_scores = []

    for i, sent in enumerate(tokens):
        lengths = [len(token) for token in sent]

        if len(lengths) > 1:
            mi_scores.append(sklearn.metrics.mutual_info_score(lengths, importance[i]))
            spearman.append(scipy.stats.spearmanr(lengths, importance[i])[0])
            kendall.append(scipy.stats.kendalltau(lengths, importance[i])[0])

    print("---------------")
    print("Length Baseline")
    print("Spearman Correlation: Mean: {:0.2f}, Stdev: {:0.2f}".format(np.nanmean(np.asarray(spearman)),
                                                                       np.nanstd(np.asarray(spearman))))
    # print("Kendall Tau: Mean: {:0.2f}, Stdev: {:0.2f}".format( np.nanmean(np.asarray(kendall)), np.nanstd(np.asarray(kendall))))
    # print("Mutual Information: Mean: {:0.2f}, Stdev: {:0.2f}".format( np.nanmean(np.asarray(mi_scores)), np.nanstd(np.asarray(mi_scores))))
    print("---------------")
    print()
    spearman_mean = np.nanmean(np.asarray(spearman))
    spearman_std = np.nanstd(np.asarray(spearman))
    return spearman_mean, spearman_std


def calculate_freq_baseline(frequencies, importance):
    spearman = []
    kendall = []
    mi_scores = []

    for i in range(len(frequencies)):
        if len(frequencies[i])>0:
            mi_scores.append(sklearn.metrics.mutual_info_score(frequencies[i], importance[i]))
            spearman.append(scipy.stats.spearmanr(frequencies[i], importance[i])[0])
            kendall.append(scipy.stats.kendalltau(frequencies[i], importance[i])[0])

    spearman_mean = np.nanmean(np.asarray(spearman))
    spearman_std = np.nanstd(np.asarray(spearman))
    print("---------------")
    print("Frequency Baseline")
    print("Spearman Correlation: Mean: {:0.2f}, Stdev: {:0.2f}".format(spearman_mean,
                                                                       spearman_std))
    # print("Kendall Tau: Mean: {:0.2f}, Stdev: {:0.2f}".format( np.nanmean(np.asarray(kendall)), np.nanstd(np.asarray(kendall))))
    # print("Mutual Information: Mean: {:0.2f}, Stdev: {:0.2f}".format( np.nanmean(np.asarray(mi_scores)), np.nanstd(np.asarray(mi_scores))))
    print("---------------")
    print()
    return spearman_mean, spearman_std


def calculate_permutation_baseline(human_importance, model_importance, num_permutations=100, seed=35):
    all_random_correlations = []
    for i in range(len(human_importance)):
        if not len(human_importance[i]) == len(model_importance[i]):
            pass
            #  print("Alignment Error: " + str(i))
        else:
            # Ignore sentences of length 1
            if len(human_importance[i])>1:
                random_correlations = []
                for k in range(num_permutations):
                    shuffled_importance = random.sample(list(model_importance[i]), len(model_importance[i]))
                    spearman = scipy.stats.spearmanr(shuffled_importance, human_importance[i])[0]
                    random_correlations.append(spearman)
                mean_sentence = np.nanmean(np.asarray(random_correlations))
                all_random_correlations.append(mean_sentence)

    spearman_mean = np.nanmean(np.asarray(all_random_correlations))
    spearman_std = np.nanstd(np.asarray(all_random_correlations))
    print("---------------")
    print("Permutation baseline: Mean: {:0.2f}, stdev: {:0.2f}".format(spearman_mean, spearman_std))
    print("---------------")
    print()
    return spearman_mean, spearman_std
