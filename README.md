# RecurrentAttentionObjectCounting

Authors: Jack Lindsey, Steven Jiang

Written for Torch.

With some small tweaks, RA.lua recreates (including borrowing some code and relying on the same dpnn and rnn libraries) what has been done here: https://github.com/Element-Research/rnn/blob/master/examples/recurrent-visual-attention.lua

which is in turn based on the following paper:
https://papers.nips.cc/paper/5542-recurrent-models-of-visual-attention.pdf

But applying it to the task of object counting, using this dataset:
https://drive.google.com/file/d/0Bz1o6xt9cD-3ZlFlc3lpQ25CeWM/view?usp=sharing

generated using a slightly modified version of SIMCEP
http://www.cs.tut.fi/sgn/csb/simcep/tool.html

which consists of images containing between one and five artificially generated "cells."

RA.lua also has a test feature where a network trained to work with N glimpses is forced to make early predictions based on M glimpses, M <= N, in order to test the network's ability to choose the appropriate salient image regions to glimpse at.

feedforward.lua is a more standard feedforward convolutional network geared towards classifying the same dataset, based on this tutorial: https://github.com/soumith/cvpr2015/blob/master/Deep%20Learning%20with%20Torch.ipynb.  Both files borrow some code from Torch tutorials/documentation. 



