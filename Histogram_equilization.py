# importing libraries
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import glob


# #### *rk_freq: original pixel intensity frequency*
# #### *sk_freq: new pixel intensity frequency*


# Histogram equalization function, [0,L-1]: pixel intensities ; 0:black, L-1: white
def hist_eql(img, L=256):
    M, N = img.shape

    # Calculate frequency of occurence

    rk_freq = {key: 0 for key in range(L)}
    for i in range(M):
        for j in range(N):
            key = img[i, j]
            rk_freq[key] += 1
    rk_freq_arr = np.array(list(rk_freq.values()))

    # Calculate new pixel intensities (k=0 to L-1)

    sk_freq = {key: 0 for key in range(L)}
    for k in range(L):
        sk_freq[k] = (L - 1) * (np.sum(rk_freq_arr[:k + 1]) / (M * N))
    sk_freq_arr = np.array(list(sk_freq.values()), dtype=np.uint8)

    # Applying new pixel intensities (transformed values) for each pixel
    img_new = np.zeros([M, N])
    for i in range(M):
        for j in range(N):
            key = img[i, j]
            img_new[i, j] = sk_freq_arr[key]
    img_new = img_new.astype(np.uint8)

    return img_new


# Function to plot original and equalized image
def plot_image(old, new):
    cv2.imshow('Original Image', old)
    cv2.waitKey(0)

    cv2.imshow('Histogram Equalized Image', new)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


#### *Histogram plot of original and new image*


def plot_hist(old,new,img_type='GS'):

    if img_type=='GS':
        old_image = old.reshape([np.size(old), ])  # Converting 2D array into a single list.
        new_image = new.reshape([np.size(new), ])

        plt.figure(figsize=(12, 4))
        plt.subplot(121)
        plt.hist(old_image, bins=256)  # Histogram for original image
        plt.title('Hist. of original image')

        plt.subplot(122)
        plt.hist(new_image, bins=256)  # Histogram after equalization
        plt.title("Hist. of new image")

        plt.show()
        plt.close()
    elif img_type=='RGB':

        b, g, r = cv2.split(old)   # Separating RGB channels

        # Converting 2D array into a single list.

        b_image = b.reshape([np.size(b), ])
        g_image = g.reshape([np.size(g), ])
        r_image = r.reshape([np.size(r), ])
        new_image = new.reshape([np.size(new), ])

        plt.figure(figsize=(12, 4))
        plt.subplot(131)
        plt.hist(r_image, bins=256)  # Histogram for R channel
        plt.title('Hist. of R channel')

        plt.subplot(132)
        plt.hist(g_image, bins=256)  # Histogram for G channel
        plt.title('Hist. of G channel')
        plt.subplot(133)
        plt.hist(b_image, bins=256)  # Histogram for B channel
        plt.title('Hist. of B channel')

        plt.show()

        plt.hist(new_image, bins=256)  # Histogram after equalization
        plt.title("Hist. of new image")

        plt.show()
        plt.close('all')


'''Function to perform histogram equalization'''

def perform_hist_eql(img_lst):

    for img in img_lst:

        # Check if image is colored or grayscale
        if len(img.shape) < 3:  # Grayscale image equalization

            new = hist_eql(img)  # New image after histogram equalization
            plot_image(img, new)  # Plot for original and new image
            plot_hist(img, new,img_type='GS')  # Plot histograms for original and equalized image

        elif len(img.shape) == 3:  # R G B image equalization

            # Segregate color streams, R G B
            b, g, r = cv2.split(img)

            # Separately equalize all three channels
            img_b = hist_eql(b)
            img_g = hist_eql(g)
            img_r = hist_eql(r)

            # Merge equalized channels
            new = cv2.merge((img_b, img_g, img_r))
            plot_image(img, new)  # Plot for original and new image
            plot_hist(img, new,img_type='RGB')  # Plot histograms for original and equalized image

'''Function to randomly generate normalized histogram
M,N: size of image
k: pixel intensity value
'''


def generate_norm(M, N, k):
    k -= 1
    np.random.seed(123)
    mat = np.random.normal(int(k / 2), int(k / 8), size=(M * N,))  # generating normal values with mean k/2
    # mat=(k)*((mat-np.min(mat))/(np.max(mat)-np.min(mat)))         #rescaling values to lie between [0,k-1]
    mat = mat.reshape((M, N)).astype(np.uint8)

    return mat


'''Function to randomly generate uniform histogram
M,N: size of image
k: pixel intensity value
'''


def generate_uni(M, N, k):
    k -= 1
    np.random.seed(100)
    mat = np.random.randint(0, k - 1, size=M * N)
    mat = mat.reshape((M, N)).astype(np.uint8)

    return mat


'''Function to randomly generate binomial histogram
M,N: size of image
k: pixel intensity value
'''


def generate_bin(M, N, k):
    k -= 1
    np.random.seed(150)
    mat = np.random.binomial(k, 0.5, size=M * N)
    mat = mat.reshape((M, N)).astype(np.uint8)

    return mat


# Driver code
print('This is a histogram equalization software.\n')

while True:
    print("If you wish to enter a path of directory, press 1.", "If you wish to enter path of images, press 2.",
          "If you wish to randomly generate histogram, press 3", "If you want to quit, press 0.\n", sep='\n')
    print('Do not put file path in quotes.\n')
    choice = int(input())

    # Hist. Equa. on directory
    if choice == 1:

        print('Enter path of your directory. \n')
        image_folder = input()

        # To load images of different formats
        ext = ['\*.jpg', '\*.png', '\*.jpeg', '\*.tif', '\*.tiff', '\*.bmp', '\*.gif', '\*.raw', '\*.eps', '\*.dng']

        if os.path.exists(image_folder):
            files = []
            imgs = []
            for e in ext:
                files.extend(glob.glob(image_folder + e))  # Appending all image files of all formats

            for image_file in files:
                img = cv2.imread(image_file,-1)  # Reading Image and Converting to M*N pixel matrix
                imgs.append(img)  # Appending all image pixel matrix in a list

            perform_hist_eql(imgs)  # Perform histogram equalization on all images


        else:
            print("Folder not found.\n")

    # Hist. Equa. on multiple file paths
    elif choice == 2:
        imgs = []
        while True:

            print('Enter path of your images. (Do not put file path in quotes.)\n')
            print("Hit 'Enter' when you're done.\n")
            image_file = input()

            if image_file == '':
                break
            if os.path.exists(image_file):
                img = cv2.imread(image_file,-1)  # Reading Image and Converting to M*N pixel matrix
                imgs.append(img)  # Appending all image pixel matrix in a list

            else:
                print("File not found.\n")
        perform_hist_eql(imgs)

        # Hist. Equa. on random sample
    elif choice == 3:

        # Enter image size
        print("Enter the size of image (M,N). \n")
        print('Enter M and N, seperated by space.')
        M, N = input().split()
        M = int(M)
        N = int(N)

        # Fix pixel intensity (uint8 or uint16)
        bits = {8: (256, np.uint8), 16: (65536, np.uint16)}

        #print("Enter pixel bit size")
        #bit_size = int(input())
        bit_size=8

        print("Which distribution you want to use? Enter choice number. \n")
        print("1. Gaussian", "2. Uniform", "3. Binomial", sep='\n')
        choice2 = int(input())

        if choice2 == 1:
            # Histogram equalization for samples drawn from gaussian
            norm = generate_norm(M, N, bits[bit_size][0])  # generate normal histogram
            new_norm = hist_eql(norm, bits[bit_size][0])  # New image after histogram equalization
            plot_image(norm, new_norm)  # Plot for original and new image
            plot_hist(norm, new_norm)  # Plot histograms for original and equalized image

        elif choice2 == 2:
            # Histogram equalization for samples drawn from uniform
            uni = generate_uni(M, N, bits[bit_size][0])  # generate uniform histogram
            new_uni = hist_eql(uni, bits[bit_size][0])  # New image after histogram equalization
            plot_image(uni, new_uni)  # Plot for original and new image
            plot_hist(uni, new_uni)  # Plot histograms for original and equalized image

        elif choice2 == 3:
            # Histogram equalization for samples drawn from binomial
            bino = generate_bin(M, N, bits[bit_size][0])  # generate binomial histogram
            new_bino = hist_eql(bino, bits[bit_size][0])  # New image after histogram equalization
            plot_image(bino, new_bino)  # Plot for original and new image
            plot_hist(bino, new_bino)  # Plot histograms for original and equalized image


        else:
            print("Invalid choice. \n")

    elif choice == 0:
        break
    else:
        print("Invalid choice. Enter your choice again. \n")