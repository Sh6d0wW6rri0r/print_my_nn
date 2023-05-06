import numpy as np
from scipy.special import softmax
import matplotlib.pyplot as plt
import os, sys, time, datetime, json, random
from PIL import Image
import torch as torch
from printPytorch import TorchPrinter


class TinyModel(torch.nn.Module):
	def __init__(self):
		super(TinyModel, self).__init__()

		self.linear1 = torch.nn.Linear(100, 200)
		self.activation = torch.nn.ReLU()
		self.linear2 = torch.nn.Linear(200, 400)
		self.activation = torch.nn.ReLU()
		self.linear3 = torch.nn.Linear(500, 200)
		self.activation = torch.nn.ReLU()
		self.linear4 = torch.nn.Linear(200, 10)
		self.softmax = torch.nn.Softmax()

	def forward(self, x):
		x = self.linear1(x)
		x = self.activation(x)
		x = self.linear2(x)
		x = self.softmax(x)
		return x


mod = TinyModel()	
tprint = TorchPrinter(mod)
tprint.printWeightsImage("test_weights.jpg",2)
tprint.printArchitecture("test_architecture.jpg")