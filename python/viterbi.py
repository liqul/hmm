from hmm import HMM
import numpy as np

#the Viterbi algorithm
def viterbi(hmm, initial_dist, emissions):
    probs = hmm.emission_dist(emissions[0]) * initial_dist
    stack = []

    for emission in emissions[1:]:
        trans_probs = hmm.transition_probs * np.row_stack(probs)
        max_col_ixs = np.argmax(trans_probs, axis=0)
        probs = hmm.emission_dist(emission) * trans_probs[max_col_ixs, np.arange(hmm.num_states)]

        stack.append(max_col_ixs)

    state_seq = [np.argmax(probs)]

    while stack:
        max_col_ixs = stack.pop()
        state_seq.append(max_col_ixs[state_seq[-1]])

    state_seq.reverse()

    return state_seq

# The viterbi algorithm where the input emissions are distributions
# Sometimes, the observation is with uncertainty. For instance, the observation is output from a error prone classifier with estimated
# probabilities for each class. In this case, instead of using the emission probability defined in the parameter, we calculate the 
# inner product of the observed distribution and the corresponding emission distribution.
def viterbi2(hmm, initial_dist, emission_dists):
    probs = np.sum(hmm.emission_matrix * np.column_stack(initial_dist) * emission_dists[0], axis = 1)
    stack = []

    for emission in emission_dists[1:]:
        trans_probs = hmm.transition_probs * np.row_stack(probs)
        max_col_ixs = np.argmax(trans_probs, axis=0)
        probs = np.sum(hmm.emission_matrix * emission, axis = 1) * trans_probs[max_col_ixs, np.arange(hmm.num_states)]

        stack.append(max_col_ixs)

    state_seq = [np.argmax(probs)]

    while stack:
        max_col_ixs = stack.pop()
        state_seq.append(max_col_ixs[state_seq[-1]])

    state_seq.reverse()

    return state_seq

def dumy_dist(arr, num_states):
    new_arr = []
    for v in arr:
        tmp = np.zeros((1,num_states))
        tmp[v] = 1.0
        new_arr.append(tmp)
    return new_arr

#examples
#from Wikipedia
wiki_transition_probs = np.array([[0.7, 0.4], [0.3, 0.6]]) #0=Healthy, 1=Fever
wiki_emissions = [2, 1, 0]
wiki_emission_probs = np.array([[0.1, 0.4, 0.5], [0.6, 0.3, 0.1]]) #0=Dizzy, 1=Cold, 2=Normal
wiki_initial_dist = np.array([[0.6, 0.4]])
wiki_hmm = HMM(wiki_transition_probs, wiki_emission_probs)

if __name__ == "__main__":
    print(viterbi(wiki_hmm, wiki_initial_dist, wiki_emissions))
    print(viterbi2(wiki_hmm, wiki_initial_dist, dumy_dist(wiki_emissions, 3)))
