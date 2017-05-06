# pytorch-sentiment-neuron
Requires pytorch, cuda and python 3.5

Pytorch version of generating-reviews-discovering-sentiment : https://github.com/openai/generating-reviews-discovering-sentiment

Sample command :

python visualize.py -seq_length 1000 -cuda -load_model mlstm_ns.pt -temperature 0.4 -neuron 2388 -init "I couldn't figure out"

Model file mlstm_ns.pt is available in release v0.1
