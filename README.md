# RecurrentAttentionObjectCounting


With some small tweaks, essentially recreating (including borrowing some code) what has been done here: https://github.com/Element-Research/rnn/blob/master/examples/recurrent-visual-attention.lua

which is in turn based on the following paper:
https://papers.nips.cc/paper/5542-recurrent-models-of-visual-attention.pdf

But applying it to the attached dataset, generated using a slightly modified version of SIMCEP:
http://www.cs.tut.fi/sgn/csb/simcep/tool.html

And adding an experimental test feature where a network trained to work with N glimpses is forced to make early predictions based on M glimpses, M <= N, in order to test the network's ability to choose the appropriate salient image regions to glimpse at.

