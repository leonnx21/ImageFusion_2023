# from re import I
# import numpy as np
# from scipy import signal
# from scipy import ndimage

# from PIL import Image


# def crosscorrelation(orignal, fusedimage):

#     imA = Image.open(orignal)
#     imA = np.asarray(imA)

#     imB = Image.open(orignal)
#     imB = np.asarray(imB)

#     cc = signal.correlate2d(imA, imB)

#     return cc

# def mutual_information_2d(original, fusedimage):
#     sigma = 1
#     normalized = False
#     EPS = np.finfo(float).eps
#     original_image = Image.open(original)
#     original_image = np.asarray(original_image)

#     original_shape = original_image.shape    

#     if len(original_shape) == 2:
#         original_image = np.stack((original_image,)*3, axis=-1)

#     fused_image = Image.open(fusedimage)
#     fused_image = (np.asarray(fused_image))

#     mi_total = []

#     original_image = original_image.reshape(3, original_shape[0], original_shape[1])
#     fused_image = fused_image.reshape(3, original_shape[0], original_shape[1])

#     for i in range(3):
#         x = original_image[i].flatten()
#         y = fused_image[i].flatten()

#         print(x.shape)
#         print(y.shape)
#         """
#         Computes (normalized) mutual information between two 1D variate from a
#         joint histogram.
#         Parameters
#         ----------
#         x : 1D array
#             first variable
#         y : 1D array
#             second variable
#         sigma: float
#             sigma for Gaussian smoothing of the joint histogram
#         Returns
#         -------
#         nmi: float
#             the computed similariy measure
#         """
#         bins = (256, 256)

#         jh = np.histogram2d(x, y, bins=bins)[0]

#         # smooth the jh with a gaussian filter of given sigma
#         ndimage.gaussian_filter(jh, sigma=sigma, mode='constant',output=jh)

#         # print("EPS", EPS)

#         # compute marginal histograms
#         jh = jh + EPS
#         sh = np.sum(jh)
#         jh = jh / sh
#         s1 = np.sum(jh, axis=0).reshape((-1, jh.shape[0]))
#         s2 = np.sum(jh, axis=1).reshape((jh.shape[1], -1))

#         # Normalised Mutual Information of:
#         # Studholme,  jhill & jhawkes (1998).
#         # "A normalized entropy measure of 3-D medical image alignment".
#         # in Proc. Medical Imaging 1998, vol. 3338, San Diego, CA, pp. 132-143.
#         if normalized:
#             mi = ((np.sum(s1 * np.log(s1)) + np.sum(s2 * np.log(s2)))
#                     / np.sum(jh * np.log(jh))) - 1
#         else:
#             mi = ( np.sum(jh * np.log(jh)) - np.sum(s1 * np.log(s1))
#                 - np.sum(s2 * np.log(s2)))

#         print('layer:',i,"-",mi)
#         mi_total.append(mi)

#     result = sum(mi_total)/len(mi_total)
    
#     return result


# if __name__ == "__main__":
#     cc = crosscorrelation('New_model/Original_test_image/IR/1.png', "New_model/output_final/addition/fusion('00001.jpg',)-('00001.jpg',).jpg")
#     # mi = mutual_information_2d('New_model/Original_test_image/VIS/1.png', "New_model/output_final/addition/fusion('1.png',)-('1.png',).jpg")
#     print(cc)
#     # print(mi)

#     # x = Image.open('New_model/Original_test_image/IR/1.png')
#     # x = (np.asarray(x))
#     # print(len(x.shape))

#     # original_shape = x.shape    

#     # if len(original_shape) == 2:
#     #     x = np.stack((x,)*3, axis=-1)

#     # x = x.reshape(3, original_shape[0], original_shape[1])

#     # x_shape = x.shape
#     # print(x_shape)



#     # y = Image.open("New_model/output_final/addition/fusion('1.png',)-('1.png',).jpg")
#     # y = (np.asarray(y))
#     # print(y.shape)