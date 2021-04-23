# This code has been dapted from bertviz: https://github.com/jessevig/bertviz/


def merge_subwords(tokens, summed_importance):
    adjusted_tokens = []
    adjusted_importance = []

    current_token = ""
    current_importance = 0

    # Tokenizer use different word piece separators. We simply check for both here
    word_piece_separators = ("##", "_")
    for i, token in enumerate(tokens):
        # We sum the importance of word pieces
        current_importance += summed_importance[i]

        # Identify word piece
        if token.startswith(word_piece_separators):
            #skip the hash tags
            current_token += token[2:]

        else:
            current_token += token


        # Is this the last token of the sentence?
        if i == len(tokens)-1:
            adjusted_tokens.append(current_token)
            adjusted_importance.append(current_importance)

        else:
        # Are we at the end of a word?
            if not tokens[i+1].startswith(word_piece_separators):
                # append merged token and importance
                adjusted_tokens.append(current_token)
                adjusted_importance.append(current_importance)

                # reset
                current_token = ""
                current_importance = 0
    return adjusted_tokens, adjusted_importance

# Word piece tokenization splits words separated by hyphens. Most eye-tracking corpora don't do this.
# This method sums the importance for tokens separated by hyphens.
def merge_hyphens(tokens, importance):
    adjusted_tokens = []
    adjusted_importance = []

    if "-" in tokens:
        # Get all indices of -
        indices = [i for i, x in enumerate(tokens) if x == "-"]
        i = 0
        while i <= len(tokens)-1:
            if i+1 in indices:
                combined_token = tokens[i] + tokens[i+1] + tokens[i+2]
                combined_heat = importance[i] + importance[i + 1] + importance[i + 2]
                i += 3
                adjusted_tokens.append(combined_token)
                adjusted_importance.append(combined_heat)
            else:
                adjusted_tokens.append(tokens[i])
                adjusted_importance.append(importance[i])
                i += 1

        return adjusted_tokens, adjusted_importance

    else:
        return tokens, importance


#print(merge_hyphens(["this", "co", "-", "exists", "peaceful", "##ly", "today"], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]))