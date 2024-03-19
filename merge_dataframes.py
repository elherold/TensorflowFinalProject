from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import random