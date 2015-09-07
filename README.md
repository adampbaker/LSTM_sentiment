Word Level LSTM network for implementing [Hermann et al.](http://arxiv.org/pdf/1506.03340v1.pdf).

TODO:
- [DONE] Enable picking out of single word in the sequence for the cost function
- [DONE] Modify existing char-level model to use a word-encoding
- Create test sequences using the Penn Tree Bank data - predict the next word from arbitrary length sequences
- Modify cost function to use W(a) and g from the paper
- Add skip connections for inputs and outputs
- Set up word vocabulary for the pubmed dataset

This code is derived from [Wojciech Zaremba's learning to execute repo](https://github.com/wojciechz/learning_to_execute/blob/master/main.lua)

Also see [Adam's description of the LSTM code](code_description.md)
