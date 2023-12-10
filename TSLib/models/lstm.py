import torch
import torch.nn as nn

class LSTM(nn.Module):
	def __init__(self, input_size, hidden_size, num_class, num_layers, device): 
		super(LSTM, self).__init__()
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		self.device = device

		self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
		self.linear = nn.Linear(hidden_size, num_class)
		self.activat = nn.ReLU()

	def forward(self, X):
		h_0 = torch.zeros(self.num_layers, X.size(0), self.hidden_size).to(self.device)
		c_0 = torch.zeros(self.num_layers, X.size(0), self.hidden_size).to(self.device)
		out, _ = self.lstm(X, (h_0, c_0))
		out = self.activat(out)
		out = self.linear(out[:, -1, :])
		return out