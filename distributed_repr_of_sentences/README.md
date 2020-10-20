### [Distributed Representations of Sentences and Documents](https://cs.stanford.edu/~quocle/paragraph_vector.pdf)

Many ML models require fixed size input features. In NLP this can be achieved by representing each document as a bag of words (vector of length = vocab size with counts). This has two drawbacks. Firstly, you loose the ordering of the words ("me and the cat" has the same BOW repr as "the cat and me"). Separately, you ignore meaning.

This paper proposes a method to compress paragraphs or documents into fixed length vectors.