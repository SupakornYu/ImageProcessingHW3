import numpy as np
import math
import cmath
import matplotlib.pyplot as plt
import threading

class ImageSegmentationEx:
    def readPGMImage(self,path):
        file = open(path, "rb")
        pgmVer = file.readline().split()
        pgmComment = []
        while True:
            pgmComment_eachline = file.readline()
            if(pgmComment_eachline[0]=="#"):
                pgmComment.append(pgmComment_eachline)
            else:
                break
        pgmSize = pgmComment_eachline.split()
        pgmGreyscale = file.readline().split()
        pgmDataList = []
        htg = np.zeros((256),dtype=np.int32)
        np.set_printoptions(suppress=True)
        for j in range(int(pgmSize[1])):
            pgmDataX = []
            for i in range(int(pgmSize[0])):
                byte = file.read(1)
                chrToInt = ord(byte)
                pgmDataX.append(chrToInt)
                htg[chrToInt] = htg[chrToInt]+1
            pgmDataList.append(pgmDataX)
        file.close()
        pgmData = np.asarray(pgmDataList,dtype=np.int32)
        return pgmVer,pgmComment,pgmSize,pgmGreyscale,pgmData,htg

    def buildPGMFile(self,fileName,width,height,greyLevel,pgmData):
        f = open(str(fileName)+".pgm","wb")
        f.write("P5\n");
        f.write("# "+str(fileName)+"\n");
        f.write(str(width)+" "+str(height)+"\n"+str(greyLevel[0])+"\n");
        for i in range(int(height)):
            for j in range(int(width)):
                if pgmData[i][j]<0:
                    pgmData[i][j] = 0
                elif pgmData[i][j]>int(greyLevel[0]):
                    pgmData[i][j] = int(greyLevel[0])
                f.write(chr(pgmData[i][j]));
        f.close()

    def plotHistogramFromArray(self,histogram_arr):
        index = np.arange(256)
        bar_width = 0.35
        opacity = 0.4
        rects1 = plt.bar(index, histogram_arr, bar_width,
                         alpha=opacity,
                         color='b',
                         label='histogram1')
        plt.xlabel('Grey level')
        plt.legend()
        plt.tight_layout()
        plt.show()

    def moveAxispgmDataBeforeFourier(self,pgmData,pgmSize):
        for j in range(int(pgmSize[0])):
            for i in range(int(pgmSize[1])):
                x = math.pow(-1.0,float(i+j))
                pgmData[i][j] = float(pgmData[i][j])*x
        return pgmData
        #function for moving axis to center before using FFT.

    def convertToFourier(self,pgmData):
        return np.fft.fft2(pgmData)
        #function for fast fourier transform converting.

    def GaussianLowPassFilter(self,filename,cutoff):
        pgmVer,pgmComment,pgmSize,pgmGreyscale,pgmData,htg = self.readPGMImage(str(filename)+".pgm")
        #print pgmSize
        pgmData = self.moveAxispgmDataBeforeFourier(pgmData,pgmSize)
        pgmDataFourier = self.convertToFourier(pgmData)
        hFilter = np.zeros((int(pgmSize[1]),int(pgmSize[0])),dtype=np.float)
        hFilter.fill(0)
        m = float(pgmSize[0])
        n = float(pgmSize[1])
        for j in range(int(pgmSize[0])):
            for i in range(int(pgmSize[1])):
                dUV = float(np.sqrt(((float(j)-(m/2.0))**2.0)+((float(i)-(n/2.0))**2.0)))
                hFilter[i][j] = np.exp(-1.0*((dUV**2.0)/(2.0*(cutoff**2.0))))
        pgmResult = pgmDataFourier*hFilter
        pgmResult = np.abs(np.fft.ifft2(pgmResult))
        pgmResult = self.moveAxispgmDataBeforeFourier(pgmResult,pgmSize)
        pgmResult = np.abs(np.round(pgmResult,0).astype(int))
        self.buildPGMFile(str(filename)+"GaussianLowPassFilter"+str(cutoff),pgmSize[0],pgmSize[1],pgmGreyscale,pgmResult)
        #function for taking gaussian filter to image.

    def segmentSetGreyScaleWithMultiThreshold(self,filename,outputfilename,greyscaleWasSet,ThresholdMin,ThresholdMax):
        pgmVer,pgmComment,pgmSize,pgmGreyscale,pgmData,htg = self.readPGMImage(str(filename)+".pgm")
        pgmDataOutput = np.zeros((int(pgmSize[1]),int(pgmSize[0])),dtype=np.int)
        pgmDataOutput.fill(0)
        countPixel = 0
        for i in range(int(pgmSize[1])):
            for j in range(int(pgmSize[0])):
                if pgmData[i][j] >= ThresholdMin and  pgmData[i][j] <= ThresholdMax:
                    pgmDataOutput[i][j] = greyscaleWasSet
                    countPixel+=1
                else:
                    pgmDataOutput[i][j] = 0
        self.buildPGMFile(str(outputfilename)+"Threshold"+str(greyscaleWasSet),pgmSize[0],pgmSize[1],pgmGreyscale,pgmDataOutput)
        return countPixel
        #pixels in grey scale range(ThresholdMin,ThresholdMax) are assigned as greyscaleWasSet parameter.

    def calculateArea(self,countPixelObject,countPixelBackGroundCircle,backGroundCircleArea):
        return (float(backGroundCircleArea)/float(countPixelBackGroundCircle))*float(countPixelObject)
        #calculate from Ratio for getting area.

    def plotThreadHistogram(self,htgData):
        self.plotHistogramFromArray(htgData)

if __name__ == "__main__":


    myLib = ImageSegmentationEx()

    #myLib.GaussianLowPassFilter("Flower_Snack",30) #Function For Blur Picture By gaussian filter


    countObj = myLib.segmentSetGreyScaleWithMultiThreshold("CrabGaussianLowPassFilter50","CrabGaussian_Crab",255,220,255)
    countCir_part =  myLib.segmentSetGreyScaleWithMultiThreshold("CrabGaussianLowPassFilter50","CrabGaussian_Crab_Circle_little_part",255,0,65)
    countCir =  myLib.segmentSetGreyScaleWithMultiThreshold("CrabGaussianLowPassFilter50","CrabGaussian_Crab_Circle",255,83,255)
    countCir = countCir + countCir_part
    print str("Crab Area : ")+str(myLib.calculateArea(countObj,countCir,2.766))+str(" obj : ")+str(countObj)+str(" cir : ")+str(countCir)



    pgmVer,pgmComment,pgmSize,pgmGreyscale,pgmData,htg = myLib.readPGMImage("Flower_SnackGaussianLowPassFilter50.pgm")
    print pgmSize
    print pgmData.shape

    countObj = myLib.segmentSetGreyScaleWithMultiThreshold("Flower_SnackGaussianLowPassFilter50","Flower_Snack_Gaussian_Flower",255,229,255)
    countCir = myLib.segmentSetGreyScaleWithMultiThreshold("Flower_SnackGaussianLowPassFilter50","Flower_Snack_Gaussian_Flower_Circle",255,55,255)
    print str("Flower Area : ")+str(myLib.calculateArea(countObj,countCir,2.6))+str(" obj : ")+str(countObj)+str(" cir : ")+str(countCir)


    pgmVer,pgmComment,pgmSize,pgmGreyscale,pgmData,htgCrab = myLib.readPGMImage("CrabGaussianLowPassFilter50.pgm")
    pgmVer,pgmComment,pgmSize,pgmGreyscale,pgmData,htgFlower = myLib.readPGMImage("Flower_SnackGaussianLowPassFilter50.pgm")
    print str("pgmsize : ")+str(pgmSize)
    print str("data size : ")+str(pgmData.shape)

    """
    print "starting threading command"
    t = threading.Thread(target=myLib.plotThreadHistogram, args=(htgCrab,))
    t.start()
    t = threading.Thread(target=myLib.plotThreadHistogram, args=(htgFlower,))
    t.start()
    print "ending threading command"
    """

    myLib.plotHistogramFromArray(htgCrab)
    myLib.plotHistogramFromArray(htgFlower)