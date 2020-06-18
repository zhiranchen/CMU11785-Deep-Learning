import numpy as np


'''
SymbolSets: A list containing all the symbols (the vocabulary without blank)

y_probs: Numpy array with shape (# of symbols + 1, Seq_length, batch_size)
         Your batch size for part 1 will remain 1, but if you plan to use your
         implementation for part 2 you need to incorporate batch_size.

Return the forward probability of the greedy path (a float) and
the corresponding compressed symbol sequence i.e. without blanks
or repeated symbols (a string).
'''
def GreedySearch(SymbolSets, y_probs):
    # Follow the pseudocode from lecture to complete greedy search :-)
    num_symbols, seq_len, batch_size = y_probs.shape

    # Find the maximum probable path via greedy search
    resSymbolPaths = []
    resProbs = []
    resCompressedPaths = []
    for b in range(batch_size):
        prob = 1
        symbolPath = ["placeHolder"] * seq_len
        for t in range(seq_len):
            currMax = 0
            curr = "placeHolder"
            for i in range(num_symbols):
                if y_probs[i][t][b] > currMax:
                    currMax = y_probs[i][t][b]
                    # Take care of blank symbol
                    curr = "placeHolder" if i == 0 else SymbolSets[i-1]
            symbolPath[t] = curr
            prob *= currMax
        resSymbolPaths.append(symbolPath)
        resProbs.append(prob)

    # Build compressed paths
    for b in range(batch_size):
        compressedPath = ""
        prev = None
        for t in range(seq_len):
            # redundant symbol
            if prev != None and resSymbolPaths[b][t] == prev:
                continue
            if resSymbolPaths[b][t] == "placeHolder":
                prev = None
                continue
            compressedPath += resSymbolPaths[b][t]
            prev = resSymbolPaths[b][t]
        resCompressedPaths.append(compressedPath)

    # Return output accordingly wrt batch_size
    if batch_size == 1:
        return resCompressedPaths[0], resProbs[0]
    else:
        return resCompressedPaths, resProbs



##############################################################################


def InitializePaths(SymbolSets, y):
    InitialBlankPathScore, InitialPathScore = {}, {}
    # First push the blank into a path-ending-with-blank stack. No symbol has been invoked yet
    path = ""
    InitialBlankPathScore[path] = y[0] # Score of blank at t=1
    InitialPathsWithFinalBlank = set()
    InitialPathsWithFinalBlank.add(path)

    # Push rest of the symbols into a path-ending-with-symbol set, without the blank
    InitialPathsWithFinalSymbol = set()
    for i in range(len(SymbolSets)):
        path = SymbolSets[i]
        InitialPathScore[path] = y[i + 1]
        InitialPathsWithFinalSymbol.add(path)  # set addition
    return InitialPathsWithFinalBlank, InitialPathsWithFinalSymbol, InitialBlankPathScore, InitialPathScore

def ExtendWithBlank(PathsWithTerminalBlank, PathsWithTerminalSymbol, y, BlankPathScore, PathScore):
    UpdatedPathsWithTerminalBlank = set()
    UpdatedBlankPathScore = {}

    # First work on paths with terminal blanks, horizontal transitions
    for path in PathsWithTerminalBlank:
        # Repeating a blank does not change the symbol sequence
        UpdatedPathsWithTerminalBlank.add(path)
        UpdatedBlankPathScore[path] = BlankPathScore[path] * y[0]
    # Then extend paths with terminal symbols by blanks
    for path in PathsWithTerminalSymbol:
        # If there is already an equivalent string in UpdatedPathsWithTerminalBlank
        # simply add the score. If not create a new entry
        if path in UpdatedPathsWithTerminalBlank:
            UpdatedBlankPathScore[path] += PathScore[path] * y[0]
        else:
            UpdatedPathsWithTerminalBlank.add(path)
            UpdatedBlankPathScore[path] = PathScore[path] * y[0]
    return UpdatedPathsWithTerminalBlank, UpdatedBlankPathScore


def ExtendWithSymbol(PathsWithTerminalBlank, PathsWithTerminalSymbol, SymbolSet, y, BlankPathScore, PathScore):
    UpdatedPathsWithTerminalSymbol = set()
    UpdatedPathScore = {}

    # First extend the paths terminating in blanks. This will always create a new sequence
    for path in PathsWithTerminalBlank:
        for i in range(len(SymbolSet)): # Symbolset does not include blanks
            newpath = path + SymbolSet[i]
            UpdatedPathsWithTerminalSymbol.add(newpath)
            UpdatedPathScore[newpath] = BlankPathScore[path] * y[i+1]

    # Next work on paths with terminal symbols
    for path in PathsWithTerminalSymbol:
        for i in range(len(SymbolSet)): # Symbolset does not include blanks
            # Extend the path with every symbol other than blank
            newpath = path if (SymbolSet[i] == path[-1]) else path + SymbolSet[i] # horizontal
            if newpath in UpdatedPathsWithTerminalSymbol: # Already in list, merge paths
                UpdatedPathScore[newpath] += PathScore[path] * y[i+1]
            else: # Create new path
                UpdatedPathsWithTerminalSymbol.add(newpath)
                UpdatedPathScore[newpath] = PathScore[path] * y[i+1]
    return UpdatedPathsWithTerminalSymbol, UpdatedPathScore

def Prune(PathsWithTerminalBlank, PathsWithTerminalSymbol, BlankPathScore, PathScore, BeamWidth):
    PrunedBlankPathScore, PrunedPathScore = {}, {}
    PrunedPathsWithTerminalBlank, PrunedPathsWithTerminalSymbol = set(), set()
    scorelist = []
    # First gather all the relevant scores
    for p in PathsWithTerminalBlank:
        scorelist.append(BlankPathScore[p])
    for p in PathsWithTerminalSymbol:
        scorelist.append(PathScore[p])

    # Sort and find cutoff score that retains exactly BeamWidth paths
    scorelist.sort(reverse=True)
    cutoff = scorelist[BeamWidth] if (BeamWidth < len(scorelist)) else scorelist[-1]

    for p in PathsWithTerminalBlank:
        if BlankPathScore[p] > cutoff:
            PrunedPathsWithTerminalBlank.add(p)
            PrunedBlankPathScore[p] = BlankPathScore[p]

    for p in PathsWithTerminalSymbol:
        if PathScore[p] > cutoff:
            PrunedPathsWithTerminalSymbol.add(p)
            PrunedPathScore[p] = PathScore[p]
    return PrunedPathsWithTerminalBlank, PrunedPathsWithTerminalSymbol, PrunedBlankPathScore, PrunedPathScore

def MergeIdenticalPaths(PathsWithTerminalBlank, PathsWithTerminalSymbol, BlankPathScore, PathScore):
    # All paths with terminal symbosl will remain
    MergedPaths = PathsWithTerminalSymbol
    FinalPathScore = PathScore

    # Paths with terminal blanks will contribute scores to existing identical paths from
    # PathsWithTerminalSymbol if present, or be included in the final set, otherwise
    for p in PathsWithTerminalBlank:
        if p in MergedPaths:
            FinalPathScore[p] += BlankPathScore[p]
        else:
            MergedPaths.add(p)
            FinalPathScore[p] = BlankPathScore[p]
    return MergedPaths, FinalPathScore

'''
SymbolSets: A list containing all the symbols (the vocabulary without blank)

y_probs: Numpy array with shape (# of symbols + 1, Seq_length, batch_size)
         Your batch size for part 1 will remain 1, but if you plan to use your
         implementation for part 2 you need to incorporate batch_size.

BeamWidth: Width of the beam.

The function should return the symbol sequence with the best path score
(forward probability) and a dictionary of all the final merged paths with
their scores.
'''

def BeamSearch(SymbolSets, y_probs, BeamWidth):
    # Follow the pseudocode from lecture to complete beam search :-)
    PathScore = {} # dict of scores for paths ending with symbols
    BlankPathScore = {} # dict of scores for paths ending with blanks
    num_symbols, seq_len, batch_size = y_probs.shape

    # First time instant: initialize paths with each of the symbols, including blank, using score at t=1
    NewPathsWithTerminalBlank, NewPathsWithTerminalSymbol, NewBlankPathScore, NewPathScore = InitializePaths(SymbolSets, y_probs[:, 0, :])

    # Subsequent time steps
    for t in range(1, seq_len):
        PathsWithTerminalBlank, PathsWithTerminalSymbol, BlankPathScore, PathScore = Prune(NewPathsWithTerminalBlank,
                                                                                           NewPathsWithTerminalSymbol,
                                                                                           NewBlankPathScore, NewPathScore,
                                                                                           BeamWidth)

        NewPathsWithTerminalBlank, NewBlankPathScore =  ExtendWithBlank(PathsWithTerminalBlank, PathsWithTerminalSymbol, y_probs[:, t, :], BlankPathScore, PathScore)

        # Next extend paths by a symbol
        NewPathsWithTerminalSymbol, NewPathScore = ExtendWithSymbol(PathsWithTerminalBlank, PathsWithTerminalSymbol, SymbolSets, y_probs[:, t, :], BlankPathScore, PathScore)

    # Merge identical paths differing only by the final blank
    MergedPaths, FinalPathScore = MergeIdenticalPaths(NewPathsWithTerminalBlank, NewPathsWithTerminalSymbol, NewBlankPathScore, NewPathScore)


    # Pick the best path
    BestPath = max(FinalPathScore, key=FinalPathScore.get) # Find the path with the best score
    return BestPath, FinalPathScore




