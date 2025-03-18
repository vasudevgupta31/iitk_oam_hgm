import numpy as np



def int_to_smile(array, indices_token, pad_char):
    """
    Converts an array of integers into a list of SMILES strings.
    Removes padding characters.
    """
    return [''.join(indices_token[str(int(x))] for x in seq).replace(pad_char, '') for seq in array]


def one_hot_encode(token_lists, n_chars):
    """
    Converts token indices into one-hot encoding.
    """
    output = np.zeros((len(token_lists), len(token_lists[0]), n_chars), dtype=np.float32)
    rows, cols = np.indices((len(token_lists), len(token_lists[0])))
    output[rows, cols, token_lists] = 1
    return output


def get_token_proba(preds, temp):
    """
    Applies temperature-based sampling on model predictions.
    """
    preds = np.asarray(preds, dtype=np.float64)
    exp_preds = np.exp(preds / temp)  # Normalize in one step
    probas = exp_preds / np.sum(exp_preds)
    
    return probas, np.random.choice(len(probas), p=probas)  # Directly sample


def sample(model, temp, start_char, end_char, max_len, indices_token, token_indices):
    """
    Generates a SMILES sequence using the trained model.
    """
    n_chars = len(indices_token)
    seed_token = [token_indices[start_char]]
    generated = [indices_token[str(seed_token[0])]]

    while generated[-1] != end_char and len(generated) < max_len:
        x_seed = one_hot_encode(np.array([seed_token]), n_chars)  # Convert directly to NumPy
        logits = model.predict(x_seed, verbose=0)[0, -1]
        
        probas, next_char_ind = get_token_proba(logits, temp)
        generated.append(indices_token[str(next_char_ind)])
        seed_token.append(next_char_ind)
            
    return ''.join(generated)


def softmax(preds):
    """
    Computes softmax probabilities.
    """
    exp_preds = np.exp(preds - np.max(preds))  # Prevents overflow
    return exp_preds / np.sum(exp_preds)
