# import the frameworks, packages and libraries
import streamlit as st
from PIL import Image
from io import BytesIO
import numpy as np
import cv2 # computer vision

import os
from network import *
import torch.backends.cudnn as cudnn
from torchvision import transforms
from os.path import *
from os import listdir
import torchvision.utils as vutils
from PIL import Image
import tensorflow as tf

# function to convert an image to a water color sketch
def convertto_watercolorsketch(inp_img):
	img_1 = cv2.edgePreservingFilter(inp_img, flags=2, sigma_s=50, sigma_r=0.8)
	img_water_color = cv2.stylization(img_1, sigma_s=100, sigma_r=0.5)
	return(img_water_color)

# function to convert an image to a pencil sketch
def pencilsketch(inp_img):
	img_pencil_sketch, pencil_color_sketch = cv2.pencilSketch(
		inp_img, sigma_s=50, sigma_r=0.07, shade_factor=0.0825)
	return(img_pencil_sketch)

# function to load an image
def load_an_image(image):
	img = Image.open(image)
	return img

def modcrop(img, modulo):
    (ih, iw) = img.size
    ih = ih - (ih % modulo)
    iw = iw - (iw % modulo)
    img = img.crop((0, 0, ih, iw))
    return img

# function to convert an image to a general sketch
def generalsketch(inp_image):
    #device = torch.device("cuda:1")
    #cudnn.benchmark = True
    P2S = P2Sv2()
    #P2S.to(device)
    
    if os.path.exists('models/P2S_v2.pth'):
        pretrained_dict = torch.load('models/P2S_v2.pth', map_location=lambda storage, loc: storage)
        model_dict = P2S.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        P2S.load_state_dict(model_dict)

    P2S.eval()

    HR = inp_image.convert('RGB')
    HR = modcrop(HR, 8)
    with torch.no_grad():
        img = transform(HR).unsqueeze(0)
        out, heat_map = P2S(img)
    #torch.cuda.empty_cache()

    out = out.repeat(1, 3, 1, 1).clamp(0, 1).cpu().data
    return out  #return an EagerTensor
    
# function to convert an image to a simple sketch
def simplesketch(inp_image):
    #device = torch.device("cuda:1")
    #cudnn.benchmark = True
    P2S = P2Sv2()
    #P2S.to(device)
    
    if os.path.exists('models/simple_P2S.pth'):
        pretrained_dict = torch.load('models/simple_P2S.pth', map_location=lambda storage, loc: storage)
        model_dict = P2S.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        P2S.load_state_dict(model_dict)

    P2S.eval()

    HR = inp_image.convert('RGB')
    HR = modcrop(HR, 8)
    with torch.no_grad():
        img = transform(HR).unsqueeze(0)
        out, heat_map = P2S(img)
    #torch.cuda.empty_cache()

    out = out.repeat(1, 3, 1, 1).clamp(0, 1).cpu().data
    return out  #return an EagerTensor 

# the main function which has the code for
# the web application
def main():
	
	# basic heading and titles
	st.title('WEB APPLICATION TO CONVERT IMAGE TO SKETCH')
	st.write("This is an application developed for converting\
	your ***image*** to a ***General Sketch*** OR ***Simple Sketch***")
	st.subheader("Please Upload your image")
	
	# image file uploader
	image_file = st.file_uploader("Upload Images", type=["png", "jpg", "jpeg"])

	# if the image is uploaded then execute these
	# lines of code
	if image_file is not None:
		
		# select box (drop down to choose)
		option = st.selectbox('How would you like to convert the image',
							('Convert to general sketch',
							'Convert to simple sketch'))
		if option == 'Convert to general sketch':
			image = Image.open(image_file)
			#final_sketch = convertto_watercolorsketch(np.array(image))
			final_sketch = generalsketch(image)
			im_pil = final_sketch.permute(2, 3, 0, 1).numpy().squeeze()  
			# remove batch size, output an ndarray

            # two columns to display the original image and the
			# image after applying water color sketching effect
			col1, col2 = st.columns(2)
			with col1:
				st.header("Original Image")
				st.image(load_an_image(image_file), width=250)

			with col2:
				st.header("General Sketch")
				st.image(im_pil, width=250)
				buf = BytesIO()
				img = im_pil
				img_normalized = (img - img.min()) * (255.0 / (img.max() - img.min()))
				img = Image.fromarray(img_normalized.astype('uint8'))
				img.save(buf, format="PNG")

				byte_im = buf.getvalue()
				st.download_button(
					label="Download image",
					data=byte_im,
					file_name="generalsketch.png",
					mime="image/png"
				)

		if option == 'Convert to simple sketch':
			image = Image.open(image_file)
			final_sketch = simplesketch(image)
			im_pil = final_sketch.permute(2, 3, 0, 1).numpy().squeeze()  
			
			# two columns to display the original image
			# and the image after applying
			# pencil sketching effect
			col1, col2 = st.columns(2)
			with col1:
				st.header("Original Image")
				st.image(load_an_image(image_file), width=250)

			with col2:
				st.header("Simple Sketch")
				st.image(im_pil, width=250)
				buf = BytesIO()
				img = im_pil
				img_normalized = (img - img.min()) * (255.0 / (img.max() - img.min()))
				img = Image.fromarray(img_normalized.astype('uint8'))
				img.save(buf, format="PNG")
				byte_im = buf.getvalue()
				st.download_button(
					label="Download image",
					data=byte_im,
					file_name="simplesketch.png",
					mime="image/png"
				)

transform = transforms.Compose([
    transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
])
transform_inv = transforms.ToPILImage()

if __name__ == '__main__':
	main()
