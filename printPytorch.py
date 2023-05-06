import numpy as np
from scipy.special import softmax
import matplotlib.pyplot as plt
import os, sys, time, datetime, json, random
from PIL import Image
import torch as torch

class TorchPrinter:
	def __init__(self,model):
		self.model = model
		
	def printWeightsImage(self,outfile,zoom=1):
		model_dict=self.model.state_dict()
		nb_layers=int(len(model_dict)/2)
		print("Model with ",nb_layers," layers.")
		max_y=0
		total_x=0
		for layer in model_dict:
			if layer.endswith("weight"):
				if model_dict[layer].shape[1]>max_y:
					max_y=model_dict[layer].shape[1]
				total_x+=model_dict[layer].shape[0]
		data = []
		pos = 0
		for layer in model_dict:
			if layer.endswith("weight"):
				np_layer = np.array(model_dict[layer])
				image=np.zeros((model_dict[layer].shape[1]*zoom,model_dict[layer].shape[0]*zoom, 3), dtype=np.uint8)
				min_weight = np.min(np_layer)
				if min_weight<0:
					min_weight=-min_weight
				np_layer=np_layer+min_weight
				max_weight = np.max(np_layer)
				np_layer=np_layer/max_weight				
				for kx in range(0,model_dict[layer].shape[0]):
					for ky in range(0,model_dict[layer].shape[1]):
						for kl in range(0,zoom):
							for km in range(0,zoom):
								image[ky*zoom+kl,kx*zoom+km]=[int(np_layer[kx,ky]*255),int(np_layer[kx,ky]*255),int(np_layer[kx,ky]*255)]
				arr= image
				if model_dict[layer].shape[1]<max_y:
					add_before = int((max_y*zoom-model_dict[layer].shape[1]*zoom)/2)
					add_after = max_y*zoom-model_dict[layer].shape[1]*zoom-add_before
					arr=np.pad(arr,((add_before,add_after),(0,0),(0,0)),constant_values=0)
				if pos==0:
					data = arr
				else:
					data=np.concatenate((data,arr),axis=1)
				pos+=1
		print("Image of size: ",total_x*zoom,"x",max_y*zoom," RGB pixels")
		img = Image.fromarray(data, 'RGB')
		img.save(outfile)
	
	def drawLine(self,x,y,x2,y2,img):
		y_float = float(y)
		y_increase = (y2-y)/(x2-x)
		for i in range(0,x2-x):
			img[x+i,int(y_float),0]=255
			img[x+i,int(y_float),1]=255
			img[x+i,int(y_float),2]=255
			y_float+=y_increase						
	
	def printArchitecture(self,outfile):
		model_dict=self.model.state_dict()
		nb_layers=int(len(model_dict)/2)
		print("Model with ",nb_layers," layers.")
		max_y=0
		total_x=0
		for layer in model_dict:
			if layer.endswith("weight"):
				if model_dict[layer].shape[1]>max_y:
					max_y=model_dict[layer].shape[1]
		data = np.zeros((max_y*10,(nb_layers+1)*100,3),dtype=np.uint8)
		pos=0
		active_layer=0
		previous_layer = None
		for layer in model_dict:
			if layer.endswith("weight"):
				np_layer = np.array(model_dict[layer])
				start = int((max_y-np_layer.shape[1])/2)*10
				for i in range(0,np_layer.shape[1]):
					for l in range(0,6):
						for m in range(0,6):
							data[start+10*i+3+l,47+pos+m,0]=255	
							data[start+10*i+3+l,47+pos+m,1]=255	
							data[start+10*i+3+l,47+pos+m,2]=255				
				pos+=100
				if active_layer==nb_layers-1:
					start = int((max_y-np_layer.shape[0])/2)*10
					for i in range(0,np_layer.shape[0]):
						for l in range(0,6):
							for m in range(0,6):
								data[start+10*i+3+l,47+pos+m,0]=255	
								data[start+10*i+3+l,47+pos+m,1]=255	
								data[start+10*i+3+l,47+pos+m,2]=255					
				active_layer+=1
		img = Image.fromarray(data, 'RGB')
		img.save(outfile)
