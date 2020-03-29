# deeplearning-seq2seq
This Respository is for the Final Project of COMS4995 Deep Learning Course. In this project, we are applying seq2seq deep learning techniques and models to predict the top-n ascending-in-percent stock symbols based on the previous day's top-n ascending-in-percent ones. We believe there may be an intrinsic corresponding relationship between the input sequence and target sequence. 

There're main two steps:
1. Find the best encoder-decoder model for the seq2seq problem.
2. Augument the loss function with data-dependent gaussian prior term, certify that the paper's idea works universely, not merely on their datasets. The link to the paper: https://openreview.net/pdf?id=S1efxTVYDr
