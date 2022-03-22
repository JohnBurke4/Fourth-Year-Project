from cmath import pi
import math
import cv2
import matplotlib.pyplot as plt
import pandas as pd

import numpy as np
from skimage.metrics import structural_similarity as compare_ssim


class DCT:

    def normal1D(N):
        a0 = 1 / math.sqrt(N)
        a1 = math.sqrt(2/N)

        result = np.zeros((N, N))

        for u in range(0, N):
            for n in range(0, N):
                coef = a1
                if u == 0:
                    coef = a0

                result[u, n] = coef * np.cos((2*n + 1) * u * pi / (2 * N))

        return result

    def getDCTBitcount(dct):
        max = np.amax(dct)
        return int(math.log2(max)) + 1

    def int1D(N, precision):
        a0 = 1 / math.sqrt(N)
        a1 = math.sqrt(2/N)

        result = np.zeros((N, N), dtype=int)

        for u in range(0, N):
            for n in range(0, N):
                coef = a1
                if u == 0:
                    coef = a0

                result[u, n] = int(
                    (2**precision) * coef * np.cos((2*n + 1) * u * pi / (2 * N)))

        return result

    def convertImage(image, dct, precision=None):
        if (precision):
            return (np.matmul(np.matmul(dct, image).astype(int) / 2**(precision), np.transpose(dct)).astype(int) / 2**(precision))
        return np.matmul(np.matmul(dct, image), np.transpose(dct))

    def remakeImageImage(image, dct, precision=None):
        if (precision):
            return (np.matmul(np.matmul(np.transpose(dct).astype(int) / 2**(precision), image), dct).astype(int) / 2**(precision))
        return np.matmul(np.matmul(np.transpose(dct), image), dct)

    def removeSmallBits(chunk):
        chunk[abs(chunk) < 10] = 0
        return chunk


class ImageReader:

    def getImage(path, N):
        # Read Images
        img = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        maxSize = tuple([x - (x % N) for x in gray.shape])
        maxSize = tuple(reversed(maxSize))
        return cv2.resize(gray, maxSize)

    def displayImage(image):
        cv2.imshow("Gray", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def chopUpImage(image, N):
        result = []
        size = image.shape

        for i in range(0, size[0], N):
            for j in range(0, size[1], N):
                result.append((image[i:i+N, j:j+N]).astype(int) - 128)

        return {
            "pieces": result,
            "size": size
        }

    def reconstructImage(pieces, N):
        length = pieces['size'][0]
        width = pieces['size'][1]

        result = np.ndarray(shape=(length, width), dtype=np.uint8)

        index = 0
        for i in range(0, length, N):
            for j in range(0, width, N):
                # print((pieces['pieces'][index] + 128))
                result[i:i+N, j:j+N] = (pieces['pieces'][index] + 128)
                # print(result[i:i+N, j:j+N], i, j)
                index += 1

        return result
    SampleChunk = [[26, -5, -5, -5, -5, -5, -5, 8],
                   [64, 52, 8, 26, 26, 26, 8, -18],
                   [126, 70, 26, 26, 52, 26, -5, -5],
                   [111, 52, 8, 52, 52, 38, -5, -5],
                   [52, 26, 8, 39, 38, 21, 8, 8],
                   [0, 8, -5, 8, 26, 52, 70, 26],
                   [-5, -23, -18, 21, 8, 8, 52, 38],
                   [-18, 8, -5, -5, -5, 8, 26, 8]]

    ImageSetPaths = [
        "Image Test Sets\\artificial.pgm",
        "Image Test Sets\\big_building.pgm",
        "Image Test Sets\\big_tree.pgm",
        "Image Test Sets\\bridge.pgm",
        "Image Test Sets\cathedral.pgm",
        "Image Test Sets\deer.pgm",
        "Image Test Sets\\fireworks.pgm",
        "Image Test Sets\\flower_foveon.pgm",
        "Image Test Sets\hdr.pgm",
        "Image Test Sets\leaves_iso_200.pgm",
        "Image Test Sets\leaves_iso_1600.pgm",
        "Image Test Sets\\nightshot_iso_100.pgm",
        "Image Test Sets\\nightshot_iso_1600.pgm",
        "Image Test Sets\spider_web.pgm",
        "Image Test Sets\zone_plate.pgm",
    ]

    TestImage = "Fourth-Year-Project\JPEG Encryptor\deers.jpg"


precision = 9
N = 8

image = ImageReader.getImage(ImageReader.TestImage, 8)
chopped = ImageReader.chopUpImage(image, N)

dctLossy = DCT.int1D(N, precision)

print(dctLossy)

# print(DCT.convertImage(
#     chopped['pieces'][0], dctLossy, precision))

# print(DCT.removeSmallBits(DCT.convertImage(
#     chopped['pieces'][0], dctLossy, precision)))
total = 0
for i in range(0, len(chopped['pieces'])):
    chopped['pieces'][i] = DCT.convertImage(
        chopped['pieces'][i], dctLossy, precision)
    chopped['pieces'][i] = DCT.removeSmallBits(chopped['pieces'][i])
    total += np.sum(chopped['pieces'][i] == 0)
    chopped['pieces'][i] = DCT.remakeImageImage(
        chopped['pieces'][i], dctLossy, precision)
print(total / (64*len(chopped['pieces'])))
reconstructed = ImageReader.reconstructImage(chopped, N)
ImageReader.displayImage(reconstructed)


def testImageSets(precision, N):
    psnrs = []
    ssims = []
    dctLossy = DCT.int1D(N, precision)
    bitsRequired = DCT.getDCTBitcount(dctLossy)
    print("\n", "Precision:", precision, ". Bits Required:", bitsRequired)
    for path in ImageReader.ImageSetPaths:
        image = ImageReader.getImage(path, N)
        chopped = ImageReader.chopUpImage(image, N)
        for i in range(0, len(chopped['pieces'])):
            chopped['pieces'][i] = DCT.convertImage(
                chopped['pieces'][i], dctLossy, precision)
            chopped['pieces'][i] = DCT.remakeImageImage(
                chopped['pieces'][i], dctLossy, precision)

        reconstructed = ImageReader.reconstructImage(chopped, N)
        psnr = cv2.PSNR(image, reconstructed)
        # print("Image: {}".format(path))
        # print("Precision: 2^{}".format(precision))
        # print("Chunk size: {}".format(N))
        # print("PSNR: {}".format(psnr))
        psnrs.append(psnr)
        (score, diff) = compare_ssim(image, reconstructed, full=True)
        # print("SSIM: {}\n".format(score))
        ssims.append(score)
    print("PSNR Mean:", np.array(psnrs).mean(),
          ". STD:", np.array(psnrs).std())
    print("SSIM Mean:", np.array(ssims).mean(),
          ". STD:", np.array(ssims).std(), "\n")


# for i in range(2, precision+1):
#     testImageSets(i, 8)


def graphResults():
    fileName = "Image Test Set 1 Values.csv"
    #fileName = "BLESSINGTON STREET.csv"
    df = pd.read_csv(fileName, header=1)
    bitsRequired = np.array(df.iloc[:, 0])
    psnrMean = np.array(df.iloc[:, 1])
    psnrSTD = np.array(df.iloc[:, 2])
    ssinMean = np.array(df.iloc[:, 3])
    ssinSTD = np.array(df.iloc[:, 4])
    # psnrMeans =
    plt.figure(1)
    plt.errorbar(bitsRequired, psnrMean, yerr=psnrSTD, linewidth=3)
    plt.xlabel('DCT Bits')
    plt.ylabel('PSNR Score')
    plt.title("Lossy DCT PSNR")
    plt.figure(2)
    plt.errorbar(bitsRequired, ssinMean, yerr=ssinSTD, linewidth=3)
    plt.xlabel('DCT Bits')
    plt.ylabel('SSIN Score')
    plt.title("Lossy DCT SSIN")
    plt.show()


# graphResults()
