from pytest import param
from sklearn.model_selection import ParameterSampler
import torch

import torch.nn as nn
import torch.nn.functional as F




class CNN_Layer(nn.Module):
	""" """
	def __init__(self, previous_layer, parameters, laspeptideSize=33, dropoutPercentage = None, filters = None, kernel_size = 10, maxPool=False, batchNorm=False):
		super().__init__()
		self.rolled_out = False

		if filters == None:
			filters = parameters["CNN_filters"] 
		
		self.outputDimensions = filters

		
		kernel_size = 10

		self.layers = nn.ModuleList()
		self.layers.append(
				nn.Conv1d(
					in_channels=previous_layer.outputDimensions,  
					out_channels=filters,   
					kernel_size = kernel_size)
					)
		if maxPool:
			self.layers.append(nn.MaxPool1d(kernel_size=2))
		self.layers.append(nn.ReLU())  
		if batchNorm:
			self.layers.append(nn.BatchNorm1d(filters))
		self.layers.append(nn.Dropout(dropoutPercentage))



	def forward(self, x):
		x = torch.permute(x, (0,2,1))
		for layer in self.layers:
			x = layer(x)
		x = torch.permute(x, (0,2,1))
		return x


class FC_Layer(nn.Module):
	""" """

	def __init__(self, previous_layer, layer_size, peptideSize=33, dropoutPercentage=0.5, useActivation=True):
		super().__init__()
		self.rolled_out = True
		self.outputDimensions = layer_size
		
		input_size = previous_layer.outputDimensions
		
		if not previous_layer.rolled_out:
			self.flatten_input = True
			input_size *= peptideSize
		else:
			self.flatten_input=False

		self.layers = nn.ModuleList()
		self.layers.append(nn.Linear(input_size, layer_size))
		if useActivation:
			self.layers.append(nn.ReLU())
			self.layers.append(nn.Dropout(dropoutPercentage))


	def forward(self, x):

		if self.flatten_input:
			x = torch.flatten(x, start_dim=1)

		for layer in self.layers:
			x = layer(x)
		return x


class LSTM_Layer(nn.Module):
	""" """
	def __init__(self, previous_layer, parameters, laspeptideSize=33):
		super().__init__()
		self.outputDimensions = parameters["LSTM_hidden_size"] * 2
		self.rolled_out = False

		dropoutPercentage = parameters["LSTM_dropout"]
		hidden_size = parameters["LSTM_hidden_size"]

		self.layers = nn.ModuleList()
		self.layers.append(torch.nn.LSTM(previous_layer.outputDimensions, hidden_size, parameters["LSTM_layers"], batch_first=True, bidirectional=True, dropout=dropoutPercentage))


	def forward(self, x):
		x, (h_n, c_n) = self.layers[-1](x)

		self.h_n = h_n.view(x.shape[0], x.size()[-1])
		self.c_n = c_n
		return x
		


class BahdanauAttention(nn.Module):
	"""
	input: from RNN module h_1, ... , h_n (batch_size, seq_len, units*num_directions),
									h_n: (num_directions, batch_size, units)
	return: (batch_size, num_task, units)
	"""
	def __init__(self,previous_layer, in_features, hidden_units,num_task, add_extra=0):
		super(BahdanauAttention,self).__init__()
		self.previous_layer = previous_layer
		self.rolled_out = True
		self.outputDimensions = previous_layer.outputDimensions + add_extra

		self.W1 = nn.Linear(in_features=in_features,out_features=hidden_units)
		self.W2 = nn.Linear(in_features=in_features,out_features=hidden_units)
		self.V = nn.Linear(in_features=hidden_units, out_features=num_task)

	def forward(self, x, task_indexes=None):
		values = x

		hidden_states = self.previous_layer.h_n
		if task_indexes:
			hidden_states = hidden_states[task_indexes[0]: task_indexes[1], :]
		hidden_with_time_axis = torch.unsqueeze(hidden_states,dim=1)

		score  = self.V(nn.Tanh()(self.W1(values)+self.W2(hidden_with_time_axis)))
		attention_weights = nn.Softmax(dim=1)(score)
		values = torch.transpose(values,1,2)   # transpose to make it suitable for matrix multiplication

		context_vector = torch.matmul(values,attention_weights)
		context_vector = torch.transpose(context_vector,1,2)

		x = torch.mean(context_vector, 1)
		return x