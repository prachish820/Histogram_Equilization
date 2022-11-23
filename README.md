# Histogram Equalization
## Welcome to Histogram Equalization software documentation
**This software does histogram equalization of images to improve contrast.**\
\
Histogram equalization is a common algorithm used in image processing to improve the contrast of an image. In low-contrast images, most of the pixel intensities are clustered around a particular pixel intensity. To check that, we plot a histogram to see frequencies at different intensity levels. In the ideal case, we get a uniform histogram, i.e. every pixel intensity has the same frequency. Hence, we try to make our frequency distribution uniform. To do so, we effectively spread out the most frequent pixel intensity values, i.e. stretching out the intensity range of the image. 

# Getting Started
## Installing and importing dependencies
### Numpy
Installing numpy as OpenCV requires numpy backend. So to install it, first run this command.\
\
<img width="151" alt="np" src="https://user-images.githubusercontent.com/96483297/203165054-143ad3f3-cee5-424b-be72-342b5c35c0f4.png">


### Matplotlib
Installing matplotlib to plot histograms\
\
<img width="223" alt="plt" src="https://user-images.githubusercontent.com/96483297/203165088-e28c2bbf-ed9b-40e0-b69f-5824638c3d2e.png">

### OpenCV
To read images, we are using cv2\
\
<img width="182" alt="openCV" src="https://user-images.githubusercontent.com/96483297/203165114-777eb987-96ef-4f2a-a95e-2275e7c57201.png">

### OS
OS provides a portable way of using operating system dependent functionality. We are using it to read image files.\
\
<img width="107" alt="os" src="https://user-images.githubusercontent.com/96483297/203165133-1e88c802-5e7a-4db4-b64e-3cf408d94926.png">

### Glob
To read files from a given directory of specified extension.\
\
<img width="145" alt="glob" src="https://user-images.githubusercontent.com/96483297/203165185-c9345690-0fc2-43e0-a41f-a93898fd43d9.png">

## Running your first code
The software allows user to apply histogram equalization on a list of image files, or a directory containing images and also allows to generate a random image using either "Gaussian", "Uniform" or "Binomial" distribution sample.
\
\
<img width="322" alt="1" src="https://user-images.githubusercontent.com/96483297/203161831-7b96dcf6-8eb5-4c45-bec7-1744c4b87dc1.png">

### Example
The software will produce contrast improved image along with original as well as equalized histograms.\
\
<img width="350" alt="Hist_ex" src="https://user-images.githubusercontent.com/96483297/203162358-6fd0e221-0f9b-40cf-9fc6-107077b7a35e.png">
\
\
<img width="600" alt="RGB_org" src="https://user-images.githubusercontent.com/96483297/203162388-728cbe46-77b4-4442-90a8-781b0dc7e3c8.png">
\
\
<img width="300" alt="new" src="https://user-images.githubusercontent.com/96483297/203162402-fceea273-203d-4b80-95e8-b1d3260fb3c9.png">

## Note
The code works for any size image (M,N) from intensity levels 0 to 255 (8-bit), 0 being black and 255 white. \
The code can be further improved to include larger bit sizes -- 12 bit, 16 bit and 24 bit.\
The repository also contains standard test images for user to try out the software. Have fun !

