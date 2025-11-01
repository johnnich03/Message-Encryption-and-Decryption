import numpy as np
import random
from utils import *

# Empirical English letter frequencies (normalized)
ENGLISH_LETTER_FREQUENCIES = {
    'E': 12.70, 'T': 9.06, 'A': 8.17, 'O': 7.51, 'I': 6.97,
    'N': 6.75, 'S': 6.33, 'H': 6.09, 'R': 5.99, 'D': 4.25,
    'L': 4.03, 'C': 2.78, 'U': 2.76, 'M': 2.41, 'W': 2.36,
    'F': 2.23, 'G': 2.02, 'Y': 1.97, 'P': 1.93, 'B': 1.49,
    'V': 0.98, 'K': 0.77, 'J': 0.15, 'X': 0.15, 'Q': 0.10, 'Z': 0.07
}
# Normalize to sum to 1
TOTAL = sum(ENGLISH_LETTER_FREQUENCIES.values())
for k in ENGLISH_LETTER_FREQUENCIES:
    ENGLISH_LETTER_FREQUENCIES[k] /= TOTAL

def compute_log_probability(text, permutation_map, char_to_ix, frequency_statistics, transition_matrix):
    """
    Computes the log probability of a text under a given permutation map (switching the
    charcter c from permutation_map[c]), given the text statistics

    Note: This is quite slow, as it goes through the whole text to compute the probability,
    if you need to compute the probabilities frequently, see compute_log_probability_by_counts.

    Arguments:
    text: text, list of characters

    permutation_map[c]: gives the character to replace 'c' by

    char_to_ix: characters to index mapping

    frequency_statistics: frequency of character i is stored in frequency_statistics[i]

    transition_matrix: probability of j following i

    Returns:
    p: log likelihood of the given text
    """
    t = text
    p_map = permutation_map
    cix = char_to_ix
    fr = frequency_statistics
    tm = transition_matrix

    i0 = cix[p_map[t[0]]]
    p = np.log(fr[i0])
    i = 0
    while i < len(t)-1:
        subst = p_map[t[i+1]]
        i1 = cix[subst]
        p += np.log(tm[i0, i1])
        i0 = i1
        i += 1

    return p

def compute_transition_counts(text, char_to_ix):
    """
    Computes transition counts for a given text, useful to compute if you want to compute
    the probabilities again and again, using compute_log_probability_by_counts.

    Arguments:
    text: Text as a list of characters

    char_to_ix: character to index mapping

    Returns:
    transition_counts: transition_counts[i, j] gives number of times character j follows i
    """
    N = len(char_to_ix)
    transition_counts = np.zeros((N, N))
    c1 = text[0]
    i = 0
    while i < len(text)-1:
        c2 = text[i+1]
        transition_counts[char_to_ix[c1],char_to_ix[c2]] += 1
        c1 = c2
        i += 1

    return transition_counts

def compute_log_probability_by_counts(transition_counts, text, permutation_map, char_to_ix, frequency_statistics, transition_matrix):
    """
    Computes the log probability of a text under a given permutation map (switching the
    charcter c from permutation_map[c]), given the transition counts and the text

    Arguments:

    transition_counts: a matrix such that transition_counts[i, j] gives the counts of times j follows i,
                       see compute_transition_counts

    text: text to compute probability of, should be list of characters

    permutation_map[c]: gives the character to replace 'c' by

    char_to_ix: characters to index mapping

    frequency_statistics: frequency of character i is stored in frequency_statistics[i]

    transition_matrix: probability of j following i stored at [i, j] in this matrix

    Returns:

    p: log likelihood of the given text
    """
    c0 = char_to_ix[permutation_map[text[0]]]
    p = np.log(frequency_statistics[c0])

    p_map_indices = {}
    for c1, c2 in permutation_map.items():
        p_map_indices[char_to_ix[c1]] = char_to_ix[c2]

    indices = [value for (key, value) in sorted(p_map_indices.items())]

    p += np.sum(transition_counts*np.log(transition_matrix[indices,:][:, indices]))

    return p

def compute_difference(text_1, text_2):
    """
    Compute the number of times to text differ in character at same positions

    Arguments:

    text_1: first text list of characters
    text_2: second text, should have same length as text_1

    Returns
    cnt: number of times the texts differ in character at same positions
    """

    cnt = 0
    for x, y in zip(text_1, text_2):
        if y != x:
            cnt += 1

    return cnt

def get_state(text, transition_matrix, frequency_statistics, char_to_ix):
    """
    Generates a default state of given text statistics

    Arguments:
    pretty obvious

    Returns:
    state: A state that can be used along with,
           compute_probability_of_state, propose_a_move,
           and pretty_state for metropolis_hastings

    """
    transition_counts = compute_transition_counts(text, char_to_ix)
    p_map = generate_identity_p_map(char_to_ix.keys())

    state = {"text" : text, "transition_matrix" : transition_matrix,
             "frequency_statistics" : frequency_statistics, "char_to_ix" : char_to_ix,
            "permutation_map" : p_map, "transition_counts" : transition_counts}

    return state

def compute_probability_of_state(state):
    """
    Computes the probability of given state using compute_log_probability_by_counts
    """

    p = compute_log_probability_by_counts(state["transition_counts"], state["text"], state["permutation_map"],
                                          state["char_to_ix"], state["frequency_statistics"], state["transition_matrix"])

    return p

def sample_two_characters_weighted(char_to_ix):
    """
    Sample two different characters according to English letter frequency
    """
    chars = list(char_to_ix.keys())
    weights = np.array([ENGLISH_LETTER_FREQUENCIES.get(c.upper(), 0.001) for c in chars])
    weights = weights / weights.sum()  # normalize just in case

    chosen = np.random.choice(chars, size=2, replace=False, p=weights)
    return chosen[0], chosen[1]


def propose_a_move(state):
    """
    Proposes a new move for the given state using asymmetric proposal based on English frequencies
    """
    new_state = {key: value.copy() if isinstance(value, dict) else value for key, value in state.items()}
    p_map = state["permutation_map"].copy()

    # Sample two characters according to English frequency
    c1, c2 = sample_two_characters_weighted(state["char_to_ix"])

    # Find the keys that map to these characters
    k1, k2 = None, None
    for key, val in p_map.items():
        if val == c1:
            k1 = key
        if val == c2:
            k2 = key

    # Perform the swap
    p_map[k1], p_map[k2] = p_map[k2], p_map[k1]

    new_state["permutation_map"] = p_map
    return new_state


def proposal_probability(from_state, to_state):
    """
    Calculate the proposal probability q(x,y) for moving from from_state to to_state
    """
    # Get the character swaps that were made
    original_map = from_state["permutation_map"]
    new_map = to_state["permutation_map"]

    # Find which characters were swapped
    swapped = []
    for c in original_map:
        if original_map[c] != new_map[c]:
            swapped.append(c)

    # Should be exactly two characters swapped
    if len(swapped) != 2:
        return 0.0

    c1, c2 = swapped[0], swapped[1]

    # Calculate the probability of proposing this swap
    chars = list(from_state["char_to_ix"].keys())
    weights = np.array([ENGLISH_LETTER_FREQUENCIES.get(c.upper(), 0.001) for c in chars])
    weights = weights / weights.sum()

    # Probability of selecting c1 then c2 or c2 then c1
    ix1 = chars.index(c1)
    ix2 = chars.index(c2)
    p_select = weights[ix1] * weights[ix2] / (1 - weights[ix1]) + weights[ix2] * weights[ix1] / (1 - weights[ix2])

    return p_select

def pretty_state(state, full=True):
    """
    Returns the state in a pretty format
    """
    if not full:
        return pretty_string(scramble_text(state["text"][1:200], state["permutation_map"]), full)
    else:
        return pretty_string(scramble_text(state["text"], state["permutation_map"]), full)

# import numpy as np
# import random
# from utils import *
#
#
# def compute_log_probability(text, permutation_map, char_to_ix, frequency_statistics, transition_matrix):
#     """
#     Computes the log probability of a text under a given permutation map (switching the
#     charcter c from permutation_map[c]), given the text statistics
#
#     Note: This is quite slow, as it goes through the whole text to compute the probability,
#     if you need to compute the probabilities frequently, see compute_log_probability_by_counts.
#
#     Arguments:
#     text: text, list of characters
#
#     permutation_map[c]: gives the character to replace 'c' by
#
#     char_to_ix: characters to index mapping
#
#     frequency_statistics: frequency of character i is stored in frequency_statistics[i]
#
#     transition_matrix: probability of j following i
#
#     Returns:
#     p: log likelihood of the given text
#     """
#     t = text
#     p_map = permutation_map
#     cix = char_to_ix
#     fr = frequency_statistics
#     tm = transition_matrix
#
#     i0 = cix[p_map[t[0]]]
#     p = np.log(fr[i0])
#     i = 0
#     while i < len(t) - 1:
#         subst = p_map[t[i + 1]]
#         i1 = cix[subst]
#         p += np.log(tm[i0, i1])
#         i0 = i1
#         i += 1
#
#     return p
#
#
# def compute_transition_counts(text, char_to_ix):
#     """
#     Computes transition counts for a given text, useful to compute if you want to compute
#     the probabilities again and again, using compute_log_probability_by_counts.
#
#     Arguments:
#     text: Text as a list of characters
#
#     char_to_ix: character to index mapping
#
#     Returns:
#     transition_counts: transition_counts[i, j] gives number of times character j follows i
#     """
#     N = len(char_to_ix)
#     transition_counts = np.zeros((N, N))
#     c1 = text[0]
#     i = 0
#     while i < len(text) - 1:
#         c2 = text[i + 1]
#         transition_counts[char_to_ix[c1], char_to_ix[c2]] += 1
#         c1 = c2
#         i += 1
#
#     return transition_counts
#
#
# def compute_log_probability_by_counts(transition_counts, text, permutation_map, char_to_ix, frequency_statistics,
#                                       transition_matrix):
#     """
#     Computes the log probability of a text under a given permutation map (switching the
#     charcter c from permutation_map[c]), given the transition counts and the text
#
#     Arguments:
#
#     transition_counts: a matrix such that transition_counts[i, j] gives the counts of times j follows i,
#                        see compute_transition_counts
#
#     text: text to compute probability of, should be list of characters
#
#     permutation_map[c]: gives the character to replace 'c' by
#
#     char_to_ix: characters to index mapping
#
#     frequency_statistics: frequency of character i is stored in frequency_statistics[i]
#
#     transition_matrix: probability of j following i stored at [i, j] in this matrix
#
#     Returns:
#
#     p: log likelihood of the given text
#     """
#     c0 = char_to_ix[permutation_map[text[0]]]
#     p = np.log(frequency_statistics[c0])
#
#     p_map_indices = {}
#     for c1, c2 in permutation_map.items():
#         p_map_indices[char_to_ix[c1]] = char_to_ix[c2]
#
#     indices = [value for (key, value) in sorted(p_map_indices.items())]
#
#     p += np.sum(transition_counts * np.log(transition_matrix[indices, :][:, indices]))
#
#     return p
#
#
# def compute_difference(text_1, text_2):
#     """
#     Compute the number of times to text differ in character at same positions
#
#     Arguments:
#
#     text_1: first text list of characters
#     text_2: second text, should have same length as text_1
#
#     Returns
#     cnt: number of times the texts differ in character at same positions
#     """
#
#     cnt = 0
#     for x, y in zip(text_1, text_2):
#         if y != x:
#             cnt += 1
#
#     return cnt
#
#
# def get_state(text, transition_matrix, frequency_statistics, char_to_ix):
#     """
#     Generates a default state of given text statistics
#
#     Arguments:
#     pretty obvious
#
#     Returns:
#     state: A state that can be used along with,
#            compute_probability_of_state, propose_a_move,
#            and pretty_state for metropolis_hastings
#
#     """
#     transition_counts = compute_transition_counts(text, char_to_ix)
#     p_map = generate_identity_p_map(char_to_ix.keys())
#
#     state = {"text": text, "transition_matrix": transition_matrix,
#              "frequency_statistics": frequency_statistics, "char_to_ix": char_to_ix,
#              "permutation_map": p_map, "transition_counts": transition_counts}
#
#     return state
#
#
# def compute_probability_of_state(state):
#     """
#     Computes the probability of given state using compute_log_probability_by_counts
#     """
#
#     p = compute_log_probability_by_counts(state["transition_counts"], state["text"], state["permutation_map"],
#                                           state["char_to_ix"], state["frequency_statistics"],
#                                           state["transition_matrix"])
#
#     return p
#
#
# def propose_a_move(state):
#     """
#     Proposes a new move for the given state,
#     by moving one step (randomly swapping two characters)
#     """
#     new_state = {}
#     for key, value in state.items():
#         new_state[key] = value
#     new_state["permutation_map"] = move_one_step(state["permutation_map"])
#     return new_state
#
#
# def pretty_state(state, full=True):
#     """
#     Returns the state in a pretty format
#     """
#     if not full:
#         return pretty_string(scramble_text(state["text"][1:200], state["permutation_map"]), full)
#     else:
#         return pretty_string(scramble_text(state["text"], state["permutation_map"]), full)
