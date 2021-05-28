# Aligning the tokenization of different language model tokenizers with the tokenization in the eye-tracking corpora is really tricky.
# We did our best to account for as many cases as possible.
# Some cases are so specific that they would need to be hard-coded.
# For example, the ZUCO corpus contains a few instances of "U.S" which is seen as a single token but separated by most tokenizers.
# We decided to simply ignore these very specific cases but encourage you to do better.

def merge_subwords(tokens, summed_importance):
    adjusted_tokens = []
    adjusted_importance = []

    current_token = ""
    current_importance = 0

    # Tokenizers use different word piece separators. We simply check for both here
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
        while i < len(tokens):
            if i+1 in indices and i+2<len(tokens):
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

# Word piece tokenization splits parentheses and currency symbols as separate tokens. This is not done in Zuco.

def merge_symbols(tokens, importance):

    initial_symbols = ["(", "$", "€", "\"","\'" ]
    end_symbols = [")", "%", "\"", "\'"]
    all_symbols = initial_symbols + end_symbols
    # First check if anything needs to be done
    if any(token in all_symbols for token in tokens):
        adjusted_tokens = []
        adjusted_importance = []
        i = 0
        while i <= len(tokens)-1:
            combined_token = tokens[i]
            combined_heat = importance[i]

            # Nothing to be done for the last token
            if i <=len(tokens)-2:

                # Glue the parentheses back to the token
                if tokens[i] in initial_symbols:
                    combined_token = combined_token + tokens[i+1]
                    combined_heat = combined_heat + importance[i+1]
                    i+=1


                if i < len(tokens)-1 and tokens[i + 1] in end_symbols:
                    combined_token = combined_token + tokens[i + 1]
                    combined_heat = combined_heat + importance[i + 1]
                    i += 1
            adjusted_tokens.append(combined_token)
            adjusted_importance.append(combined_heat)
            i += 1

        return adjusted_tokens, adjusted_importance

    else:
        return tokens, importance

def merge_albert_tokens(tokens, importance):
    adjusted_tokens = []
    adjusted_importance = []
    i = 0
    # We ignore the last token [SEP]
    while i < len(tokens) - 1:
        combined_token = tokens[i]
        combined_heat = importance[i]
        # Nothing to be done for the last token
        if i < (len(tokens) -2):
            while not tokens[i+1].startswith("▁"):
                combined_token = combined_token + tokens[i+1]
                combined_heat = combined_heat + importance[i + 1]
                i += 1
                if i == len(tokens) - 2:
                    break
        combined_token.replace("▁", "")
        adjusted_tokens.append(combined_token)
        adjusted_importance.append(combined_heat)
        i += 1
    # Add the last token
    adjusted_tokens.append(tokens[i])
    adjusted_importance.append(importance[i])
    return adjusted_tokens, adjusted_importance

# These are just some tests for the methods
#print(merge_hyphens(["this", "co", "-", "exists", "peaceful", "##ly", "today"], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]))
# print(merge_symbols(["Check", "this", "(", "February", ",", "1985", ")"], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]))
# print(merge_symbols(["Check", "this", "(", "1985", ")"], [0.1, 0.2, 0.3, 0.4, 0.5]))
# print(merge_symbols(["Check", "this", "(", ")", "okay"], [0.1, 0.2, 0.3, 0.4, 0.5]))
# print(merge_symbols(["It", "costs", "$", "200", "."], [0.1, 0.2, 0.3, 0.4, 0.5]))

# Check merging of subwords
# print(merge_albert_tokens(['[CLR]', '▁presents', '▁a', '▁good', '▁case', '▁while', '▁failing', '▁to', '▁provide', '▁a', '▁reason', '▁for', '▁us', '▁to', '▁care', '▁beyond', '▁the', '▁very', '▁basic', '▁', 'dict', 'um', 's', '▁of', '▁human', '▁dec', 'ency', '[SEP]'], [0.21363762069336079, 0.06875212381891577, 0.014697464273896824, 0.02376395684040837, 0.05140783800073661, 0.027206018198431502, 0.020411475649548733, 0.012897350417446742, 0.01881216717103111, 0.012974560274041994, 0.0335241384655605, 0.01461099640793805, 0.018708945235305297, 0.014533449537614894, 0.041455039447895116, 0.023473885808694913, 0.012837706137504347, 0.020037277734046212, 0.02037590565738258, 0.0164441674593348, 0.098617052940637, 0.03552736363748528, 0.016664912663178762, 0.0145626107970789, 0.022517101256231506, 0.044144579179509834, 0.031474810989978494, 0.05592948130680448]))
#
# # Check what happens if no merging
# print(merge_albert_tokens(['[CLR]', '▁presents', '▁a', '▁good', '▁case','[SEP]' ], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]))