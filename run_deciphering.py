from metropolis_hastings import *
import shutil
from deciphering_utils import *

# !/usr/bin/python

import sys
from optparse import OptionParser


def main(argv):
    inputfile = None
    decodefile = None
    parser = OptionParser()

    parser.add_option("-i", "--input", dest="inputfile",
                      help="input file to train the code on")

    parser.add_option("-d", "--decode", dest="decode",
                      help="file that needs to be decoded")

    parser.add_option("-e", "--iters", dest="iterations",
                      help="number of iterations to run the algorithm for", default=5000)

    parser.add_option("-t", "--tolerance", dest="tolerance",
                      help="percentate acceptance tolerance, before we should stop", default=0.02)

    parser.add_option("-p", "--print_every", dest="print_every",
                      help="number of steps after which diagnostics should be printed", default=10000)

    (options, args) = parser.parse_args(argv)

    filename = options.inputfile
    decode = options.decode

    if filename is None:
        print("Input file is not specified. Type -h for help.")
        sys.exit(2)

    if decode is None:
        print("Decoding file is not specified. Type -h for help.")
        sys.exit(2)

    char_to_ix, ix_to_char, tr, fr = compute_statistics(filename)

    s = list(open(decode, 'r').read())
    scrambled_text = list(s)
    i = 0
    initial_state = get_state(scrambled_text, tr, fr, char_to_ix)

    # We'll store states, entropies, and track best guesses
    states = []
    entropies = []
    best_guesses = []  # Will store (state, entropy, iter_num, accept_prob)

    while i < 3:
        iters = options.iterations
        print_every = int(options.print_every)
        tolerance = options.tolerance

        # Track iteration count and acceptance within this run
        current_iter = 0
        accept_count = 0
        total_proposals = 0

        def wrapped_proposal_function(state):
            nonlocal current_iter, accept_count, total_proposals
            new_state = propose_a_move(state)
            total_proposals += 1
            return new_state

        def wrapped_log_density(state):
            return compute_probability_of_state(state)

        # Run metropolis-hastings
        state, lps, _ = metropolis_hastings(
            initial_state,
            proposal_function=wrapped_proposal_function,
            log_density=wrapped_log_density,
            iters=iters,
            print_every=print_every,
            tolerance=tolerance,
            pretty_state=pretty_state
        )

        # For each accepted state, track if it's a new best guess
        for s, lp in zip(state, lps):
            current_iter += 1
            accept_count += 1  # Only accepted states are in these lists

            # Calculate current acceptance probability
            accept_prob = accept_count / current_iter if current_iter > 0 else 0

            # Update best guesses
            if not best_guesses or lp > best_guesses[-1][1]:
                best_guesses.append((s, lp, current_iter, accept_prob))
                if len(best_guesses) > 3:  # Keep only top 3
                    best_guesses.pop(0)

        states.extend(state)
        entropies.extend(lps)
        i += 1

    print("\nBest Guesses:\n")
    for i, (guess_state, guess_entropy, iter_num, accept_prob) in enumerate(reversed(best_guesses)):
        print(f"Guess {i + 1}:")
        print(f"Found at iteration: {iter_num}")
        print(f"Acceptance probability: {accept_prob:.4f}")
        print("\nDecoded text:")
        print(pretty_state(guess_state, full=True))
        print(shutil.get_terminal_size().columns * '*')
        print("\n")


if __name__ == "__main__":
    main(sys.argv)