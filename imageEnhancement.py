import cv2
import numpy as np
import math
from typing import List
import seaborn as sns
import matplotlib.pyplot as plt

"""Step 1 : load image and conver into gray-scale"""

def rgb2gray(rgb):
  r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
  gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
  return gray

# Provide the image path here
img_path = "SampleImage.png"

rgbimage = cv2.imread(img_path)
# convert to grayscale
image = np.round(rgb2gray(rgbimage))

"""calculate SNR"""

def calculateSNR(image):
    float_image = [[float(pixel) for pixel in row] for row in image]

    # Calculate mean intensity and standard deviation of the noise
    total_pixels = len(image) * len(image[0])
    sum_intensity = sum(sum(row) for row in float_image)
    mean_intensity = sum_intensity / total_pixels

    sum_squared_diff = sum((pixel - mean_intensity) ** 2 for row in float_image for pixel in row)
    std_dev_noise = (sum_squared_diff / total_pixels) ** 0.5

    # Calculate Signal-to-Noise Ratio (SNR)
    snr = mean_intensity / std_dev_noise

    return snr

"""Histrogram dictionary calculate and plot viewer"""

#this function will return a dictionary, key of indesities with frequencies
def calculateHistrogram(image):
  indensity_dic = {}
  for i in range(256):
    indensity_dic[i] = 0
  for row in image:
    for column in row:
      indensity_dic[column] +=1
  return indensity_dic

#plot the histrogram by using seaborn and matplotlib
def plotHistrogram(indensity_dic):
  perc =  [i for i in indensity_dic.values()]
  plt.figure(figsize=(50,25))
  plt.title('Histrogram')
  sns.barplot(x=list(indensity_dic.keys()),y=perc)
  plt.xticks(rotation=90,horizontalalignment='right',fontweight='light')

"""Step 2 : Mean filter -> laplacian filter -> cantrast stretching | clipping

here are the required algorithms

Mean filter algorithm
"""

def meanFilter(image, maskSize):

  #get height and with of the image
  (rows, cols) = image.shape
  mask_sum = maskSize * maskSize

  # create a mask with 1's
  mask = [[1 for width in range(maskSize)] for height in range(maskSize)]

  #create a output image with zeros
  output_image = np.zeros((rows, cols), dtype=int)

  remove = math.floor(maskSize / 2)
  removed_points = [i for i in range(remove)]

  row_end_points = [rows - i - 1 for i in range(remove)]
  col_end_points = [cols - i - 1 for i in range(remove)]

  # mask has to travel range of image then create a list of those points
  mask_traval = [0]
  for i in removed_points:
    mask_traval.insert(len(mask_traval), i + 1)
    mask_traval.insert(0, -1 * (i + 1))

  r = 0
  while r < rows:
    # r should be in output image range
    if r in removed_points or r in row_end_points:
      r += 1
      continue

    c = 0
    while c < cols:
      # c should be in output image range
      if c in removed_points or c in col_end_points:
        c += 1
        continue

      sum = 0
      for k in mask_traval:
        for l in mask_traval:
          sum += image[r + k][c + l]

      output_image[r][c] = sum / mask_sum
      c += 1
    r += 1
  #output the grayscale image
  output_image = output_image[remove - 1:len(output_image)]
  return output_image

"""Median filter algorithm"""

#function to find median value from set of values
def findMedian(numbers: List):
  numbers.sort()
  if len(numbers) % 2 == 0:  # even length array
    lowerMiddle = numbers[int(len(numbers) / 2) - 1]
    upperMiddle = numbers[int(len(numbers) / 2)]
    return (lowerMiddle + upperMiddle) / 2
  return numbers[int((len(numbers) - 1) / 2)]

#median filter algorithm
def medianFilter(image, maskSize):
  (rows, cols) = image.shape
  mask_sum = maskSize * maskSize

  #create a output image with zeros
  output_image = np.zeros((rows, cols), dtype=int)

  remove = math.floor(maskSize / 2)
  removed_points = [i for i in range(remove)]

  row_end_points = [rows - i - 1 for i in range(remove)]
  col_end_points = [cols - i - 1 for i in range(remove)]

  # mask has to travel range of image then create a list of those points
  mask_traval = [0]
  for i in removed_points:
    mask_traval.insert(len(mask_traval), i + 1)
    mask_traval.insert(0, -1 * (i + 1))

  r = 0
  while r < rows:
    # r should be in output image range
    if r in removed_points or r in row_end_points:
      r += 1
      continue

    c = 0
    while c < cols:
      # c should be in output image range
      if c in removed_points or c in col_end_points:
        c += 1
        continue

      mask_value_list = []
      for k in mask_traval:
        for l in mask_traval:
          mask_value_list.append(int(image[r + k][c + l]))
      output_image[r][c] = findMedian(mask_value_list)
      c += 1
    r += 1

  return output_image

"""Normal clipping algorithm


"""

#this clipping function is cut the 1.5% from the beginning and 1% from the ending
def clipping(histrogram, image):
  height = len(image)
  width = len(image[0])
  total = 0
  r1 = 0
  r2 = 0
  for i in histrogram.keys():
    total = total + histrogram[i]
    if total >= width*height*1.5/100 and r1==0:
      r1 = i
    elif total >= width*height*99/100 and r2==0:
      r2 = i
  for i in range(len(image)):
    for j in range(len(image[0])):
      s = (image[i][j] - r1)*(255/(r2-r1))
      if s<0:
        image[i][j] = 0
      elif s>255:
        image[i][j] = 255
      else:
        image[i][j] = math.ceil(s)

  return image

"""Laplacian filter"""

#this function normalize the laplacian image because laplacian image can have negative values
def clip(image):
  c = image.min()
  d = image.max()
  for i in range(len(image)):
    for j in range(len(image[0])):
      s = (image[i][j] - c)*(255/(d-c))
      image[i][j] = round(s)
  return image


def laplacian_filter(image):
  #default kernal
  laplacian_kernel = [[0, 1, 0],
                      [1, -5, 1],
                      [0, 1, 0]]

  # Apply convolution
  m = len(image)
  n = len(image[0])
  result = np.zeros((m-2, n-2))
  for i in range(m-2):
    for j in range(n-2):
      #add the atucal image value to the laplacian value
      result[i, j] = -np.sum(image[i:i+3, j:j+3] * laplacian_kernel)+image[i][j]
  return clip(result)

"""percentile Contrast Stretching algorithm"""

# this algorithm uses three linear contrast stretching functions to stretch the contrast
# alpha,beeta,gama values athe gradients of those functions

#this function return the look up table to map old values to new values
def contrastStretchingLUT(histrogram,width,height,alpha,beeta,gama):
  total = 0
  r1 = 0 # r1 - 5% margin indesity leve
  r2 = 0 # r2 - 95% margin indensity leve

  #find r1 and r1
  for i in histrogram.keys():
    total = total + histrogram[i]
    if total >= width*height*5/100 and r1==0:
      r1 = i
    elif total >= width*height*95/100 and r2==0:
      r2 = i

  new_indensity_dic = {} #look up table as a dictionary
  for i in range(256):
    if i<=r1: # 0% - 5% range
      new_indensity_dic[i] = round(alpha*i)
    elif r1<i and i<=r2: # 5% - 95% range
      if round(beeta*(i-r1))+new_indensity_dic[r1] <255:
        new_indensity_dic[i] = round(beeta*(i-r1))+new_indensity_dic[r1]
      else:
        new_indensity_dic[i] = 255
    else: # 95% - 100% range
      if round(gama*(i-r2))+new_indensity_dic[r2]<255:
        new_indensity_dic[i] = round(gama*(i-r2))+new_indensity_dic[r2]
      else:
        new_indensity_dic[i] = 255
  return new_indensity_dic

#apply that look up table to the image
def applyContrastStretching(image):
  LUT = contrastStretchingLUT(calculateHistrogram(image),len(image),len(image[0]),0.5,1.5,0.5)
  for i in range(len(image)):
    for j in range(len(image[0])):
      image[i][j] = LUT[image[i][j]]

  return image

"""Apply step 2 :

1st apply mean filter and median filter and save the image named ***mean.png*** and ***median.png*** respectively


"""

# apply mean filter
plotHistrogram(calculateHistrogram(image))
output_img_mean = meanFilter(image, 7)
cv2.imwrite('mean.png', output_img_mean)

#apply median filter
output_img_median = medianFilter(image, 7)
cv2.imwrite('median.png', output_img_median)


# Display original and mean-filtered image
plt.figure(figsize=(8, 4))

plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Original sample Image')

plt.subplot(1, 3, 2)
plt.imshow(output_img_mean, cmap='gray')
plt.title('mean Filtered Image')

plt.subplot(1, 3, 3)
plt.imshow(output_img_median, cmap='gray')
plt.title('median Filtered Image')

plt.show()

print('SNR value of the resultant image(mean) : '+str(calculateSNR(output_img_mean)))
print('SNR value of the resultant image(median) : '+str(calculateSNR(output_img_median)))

"""Next apply Laplacian filter to mean.png and save it named ***mean-laplacian.png***"""

# Apply Laplacian filter to mean
mean_laplacian = laplacian_filter(output_img_mean)
cv2.imwrite('mean-laplacian.png', mean_laplacian)

# Apply Laplacian filter to median
median_laplacian = laplacian_filter(output_img_median)
cv2.imwrite('median-laplacian.png', median_laplacian)

# Display mean and mean-Laplacian filtered images
plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.imshow(median_laplacian, cmap='gray')
plt.title('median - Laplacian filtered Image')

plt.subplot(1, 2, 2)
plt.imshow(mean_laplacian, cmap='gray')
plt.title('Mean - Laplacian Filtered Image')
plt.show()
print('SNR value of the resultant image(mean) : '+str(calculateSNR(mean_laplacian)))
print('SNR value of the resultant image(median) : '+str(calculateSNR(median_laplacian)))

# I choose mean laplacian
plotHistrogram(calculateHistrogram(mean_laplacian))

"""Next apply contrast stretching(**percentile**) filter and save it named ***mean-laplacian-percentile.png***"""

mean_laplacian_percentile = applyContrastStretching(mean_laplacian)
cv2.imwrite('mean-laplacian-percentile.png', mean_laplacian_percentile)

# Display mean-laplacian and mean-laplacian-percentile images
plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.imshow(mean_laplacian, cmap='gray')
plt.title('mean - laplacian filtered Image')

plt.subplot(1, 2, 2)
plt.imshow(mean_laplacian_percentile, cmap='gray')
plt.title('mean - laplacian - percentile filtered Image')
print('SNR value of the resultant image : '+str(calculateSNR(mean_laplacian_percentile)))

plotHistrogram(calculateHistrogram(mean_laplacian_percentile))

"""Apply normal clipping method to contrast stretching named it as ***mean-laplacian-clip.png***"""

mean_laplacian_clip = clipping(calculateHistrogram(mean_laplacian),mean_laplacian)
cv2.imwrite('mean-laplacian-clip.png', mean_laplacian_clip)

# Display original and Laplacian-filtered images
plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.imshow(mean_laplacian, cmap='gray')
plt.title('mean - laplacian Image')

plt.subplot(1, 2, 2)
plt.imshow(mean_laplacian_clip, cmap='gray')
plt.title('mean - laplacian - clip Image')
print('SNR value of the resultant image : '+str(calculateSNR(mean_laplacian_clip)))

plotHistrogram(calculateHistrogram(mean_laplacian_clip))

"""Histrogram of the final Image: I choose the ***mean-Laplacian-clip*** image as my final image"""

plotHistrogram(calculateHistrogram(mean_laplacian_clip))

"""

*   Calculate the ***Entropy*** of the image
*   Calculate ***number of bits per pixel***

*   Calculate ***compression ratio*** of the image



"""

def entropy(histrogram,number_of_pixel):
  entropy = 0
  #entropy = Zigma(-li*P(li))
  for i in histrogram.keys():
    if histrogram[i]>0:
      entropy = entropy - histrogram[i]*(math.log(histrogram[i]/number_of_pixel,2))
  return entropy

size = len(mean_laplacian_clip)*len(mean_laplacian_clip[0])
entropy = entropy(calculateHistrogram(mean_laplacian_clip),size)
avgBitsPerPixel = entropy/size
compressionRatio = 8/avgBitsPerPixel
print('Entropy of the final image is '+str(entropy))
print('avarage number of bits per pixel is '+str(avgBitsPerPixel))
print('compression ratio of the image is '+str(compressionRatio))
