#!/usr/bin/env python
"""
beachfront-py
https://github.com/venicegeo/beachfront-py
Copyright 2017, RadiantBlue Technologies, Inc.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
  http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""


import os
import sys

try:
    import gdal
    import osr

except:
    from osgeo import gdal,ogr, osr
import numpy as np
from scipy import ndimage
try:
    from skimage.filters import threshold_otsu
except:
    from skimage.filter import threshold_otsu

import random
import json
import sklearn
from sklearn.decomposition import  PCA
from sklearn import cluster
from sklearn import mixture

import beachfront.vectorize as bfvec
import gippy


def getRandomCoordinate(xMax,yMax):
    x = random.randint(0,xMax)
    y = random.randint(0,yMax)
    return [x,y]

def getRandomSample(img,nSamples=None, percentSample=None, coordMap=0):
    y,x,z = img.shape
    if nSamples is None:
        if percentSample is None:
            percentSample == 0.10
        nSamples = int(percentSample * (x*y))
    sampleImg = np.zeros([nSamples,z], dtype=img.dtype)
    coordTracker = np.zeros([y,x], dtype='uint8')
    coords = [int(x/2), int(y/2)]
    yCoord,xCoord = coords[1], coords[0]
    #print yCoord, xCoord
    for i in range(nSamples):
        while coordTracker[yCoord,xCoord] == 255:
            coords = getRandomCoordinate(x-1,y-1)
            yCoord,xCoord = coords[1], coords[0]
        coordTracker[yCoord,xCoord] = 255
        #print yCoord, xCoord
        sampleImg[i,:] = img[yCoord,xCoord,:]
    if coordMap==1:
        imshow(coordTracker, 'gray')
    return sampleImg


def BuildWaterMasks(img):
    if type(img) == type('str'):
        img = readImage(img)
    y,x,z = img.shape
    indices = np.zeros([y,x,4], dtype='float32')
    indices[:,:,0] = (img[:,:,1].astype('float32') - img[:,:,4].astype('float32'))/(img[:,:,1].astype('float32') + img[:,:,4].astype('float32') + 0.000001)
    indices[:,:,1] = (img[:,:,1].astype('float32')/(img[:,:,4].astype('float32') + 0.000001))
    indices[:,:,2] = (img[:,:,2].astype('float32')/(img[:,:,4].astype('float32') + 0.000001))
    indices[:,:,3] = img[:,:,1].astype('float32')/(img[:,:,0].astype('float32') + 
                                                   img[:,:,2].astype('float32') + 
                                                   img[:,:,3].astype('float32') + 
                                                   img[:,:,4].astype('float32') + 0.000001)
    if z == 6:
        mask = np.where(img[:,:,5] == 0)
    else:
        mask = np.where(img[:,:,0] <= 0)
    #mask = np.where(img[:,:,5] == 0)
    for i in range(4):
        temp = indices[:,:,i]
        temp[mask] = 0
        indices[:,:,i] = temp
    return indices


def xO_Kmeans(inImg,percentage=0.25,n_clusters=2):
    y,x,z = inImg.shape
    test = sklearn.cluster.KMeans(n_clusters=n_clusters, init='random')
    sample = getRandomSample(inImg, percentSample=percentage)
    test2 = test.fit(sample)
    data = np.reshape(inImg,[y*x,z])
    test3 = test2.predict(data)
    test4 = np.reshape(test3, [y,x])
    test4 = test4.astype('uint8')
    return test4


def xO_FA(inImg,percentage=0.25,n_clusters=2):
    y,x,z = inImg.shape
    test = sklearn.cluster.FeatureAgglomeration(n_clusters=1)
    sample = getRandomSample(inImg, percentSample=percentage)
    test2 = test.fit(sample)
    data = np.reshape(inImg,[y*x,z])
    inImg = None
    test3 = test2.transform(data)
    data = None
    outImg = np.reshape(test3,[y,x])
    thresh = threshold_otsu(outImg)
    outImg = outImg > thresh
    outImg = outImg.astype('uint8')
    return outImg


def xO_PCA_inMem(img, n_bands=3, ot='float16'):
    x,y,z = img.shape
    test = np.reshape(img,((x * y),z))
    pca_transf = PCA(n_components=n_bands)
    S_ = pca_transf.fit_transform(test)
    outImg = np.reshape(S_,(x,y,n_bands))
    rs = 0
    if ot == 'float16':
        outImg_rs = outImg.astype('float16')
        rs = 1
    if ot == 'uint8':
        outImg_rs = Atebit(outImg)
        rs = 1
    if ot == 'uint16':
        outImg_rs = resample16bit(outImg)
        rs = 1
    if rs != 1:
        print 'outType %s not recognized by program' % (ot)
        outImg_rs = outImg.astype(ot)
    return outImg_rs


def PCA_Binary_Thresh(inImg):
    temp = xO_PCA_inMem(inImg, n_bands=1)[:,:,0]
    thresh = threshold_otsu(temp)
    temp = temp >= thresh
    temp = temp.astype('uint8')
    return temp


# deprecated.  In 0.18+ its GaussianMixture
def xO_GM(inImg, percentage=0.25, n_clusters=2):
    y,x,z = inImg.shape
    test = sklearn.mixture.GMM(n_components=n_clusters)
    sample = getRandomSample(inImg, percentSample=percentage)
    test2 = test.fit(sample)
    data = np.reshape(inImg,[y*x,z])
    test3 = test2.predict(data)
    test4 = np.reshape(test3, [y,x])
    test4 = test4.astype('uint8')
    return test4


def saveArrayAsRaster(rasterfn, newRasterfn, array):
    raster = gdal.Open(rasterfn)
    # nBands = raster.count
    checksum = array.ndim
    if checksum == 3:
        temp = array.shape
        nBands = temp[2]
    else:
        nBands = 1
    geotransform = raster.GetGeoTransform()
    originX = geotransform[0]
    originY = geotransform[3]
    pixelWidth = geotransform[1]
    pixelHeight = geotransform[5]
    cols = raster.RasterXSize
    rows = raster.RasterYSize

    driver = gdal.GetDriverByName('GTiff')
    if array.dtype == 'uint8':
        outRaster = driver.Create(newRasterfn, cols, rows, nBands,
                                  gdal.GDT_Byte)
    elif array.dtype == 'int16':
        outRaster = driver.Create(newRasterfn, cols, rows, nBands,
                                  gdal.GDT_Int16)
    elif array.dtype == 'float16':
        outRaster = driver.Create(newRasterfn, cols, rows, nBands,
                                  gdal.GDT_Float32)
    elif array.dtype == 'float32':
        outRaster = driver.Create(newRasterfn, cols, rows, nBands,
                                  gdal.GDT_Float32)
    elif array.dtype == 'int32':
        outRaster = driver.Create(newRasterfn, cols, rows, nBands,
                                  gdal.GDT_Int32)
    elif array.dtype == 'uint16':
        outRaster = driver.Create(newRasterfn, cols, rows, nBands,
                                  gdal.GDT_UInt16)
    else:
        outRaster = driver.Create(newRasterfn, cols, rows, nBands,
                                  gdal.GDT_CFloat64)
    outRaster.SetGeoTransform((originX, pixelWidth, 0, originY, 0,
                               pixelHeight))
    if nBands > 1:
        for i in range(1, (nBands + 1)):
            outband = outRaster.GetRasterBand(i)
            x = i-1
            outband.WriteArray(array[:, :, x])
    else:
        outband = outRaster.GetRasterBand(1)
        outband.WriteArray(array)
    outRasterSRS = osr.SpatialReference()
    outRasterSRS.ImportFromWkt(raster.GetProjectionRef())
    outRaster.SetProjection(outRasterSRS.ExportToWkt())
    outband.FlushCache()

def WinnerTakesAll(img_path, outName=None, method=1, percentage=0.25):
    # check assertions
    
    # open image raster
    rs = gdal.Open(img_path)
    img = rs.ReadAsArray()
    if img.ndim > 2:
        img = np.rollaxis(img, 0, 3)
    rs = None
    
    # Generate water indices
    waterIndices = BuildWaterMasks(img)
    y,x,z = img.shape
    if z == 6:
        mask = np.where(img[:,:,5] == 0)
    else:
        mask = np.where(img[:,:,0] <= 0)
    img = None
    
    binary = None
    # Convert to binary
    method = int(method)
    if method == 1:
        binary = xO_GM(waterIndices, percentage=percentage, n_clusters=2)
    if method == 2:
        binary = xO_Kmeans(waterIndices, percentage=percentage, n_clusters=2)
    if method == 3:
        binary = xO_FA(waterIndices, percentage=percentage, n_clusters=2)
    if method == 4:
        binary = PCA_Binary_Thresh(waterIndices)
    if binary is None:
        print 'Error: No binary image generated.  Check arguments and retry'
        return None
    
    #binary[binary == 0] = 2    
    binary[mask]=0
    # save as tif
    if outName is not None:
        saveArrayAsRaster(img_path,
                          outName,
                          binary)
    else:
        return binary
   

def VectorizeBinary(binary,outName=None):
    # ISSUES:
    #       Has to be read by gippy from file, so intermediate binary image needs to be written to file

    # open image with gippy
    imgSrc = gippy.GeoImage(binary)
    geoimg = imgSrc[0]
    # vectorize thresholded (ie now binary) image
    #b = (geoimg[0] == 1)
    #geoimg[geoimg[0] == 0] = 3
    geoimg.set_nodata(3)
    #b.set_nodata(0)
    #coastline = bfvec.potrace(b)
    coastline = bfvec.potrace(geoimg)
    # convert coordinates to GeoJSON
    geojson = bfvec.to_geojson(coastline, source=geoimg.basename())

    # write geojson output file
    with open(outName, 'w') as f:
        f.write(json.dumps(geojson))

    return geojson

def WTA_Service(img_path, outName=None, method=1, percentage=0.25):
    s1 = outName.find('.')
    tempOut = '%s_binary.tif' % outName[:s1]
    WinnerTakesAll(img_path, outName = tempOut, method=method, percentage=percentage)
    VectorizeBinary(tempOut, outName=outName)


def usage():
    print("""
          Usage:
          bfalg_WTA -i in_raster -o out_raster
          -m method (optional, default=1)
          -p sampling percentage (optional, default=0.25)"""
          )
    sys.exit(1)


if __name__ == '__main__':

    img_path = None
    out_path = None
    method = 1
    percentage = 0.25

    for i in range(len(sys.argv)-1):
        arg = sys.argv[i]
        if arg == '-i':
            img_path = sys.argv[i+1]
        elif arg == '-o':
            out_path = sys.argv[i+1]
        elif arg == '-m':
            method = int(sys.argv[i+1])
        elif arg == '-p':
            percentage = float(sys.argv[i+1])

    if img_path is None:
        usage()
    if out_path is None:
        usage()

    WTA_Service(img_path, out_path, method=method, percentage=percentage)
    sys.exit(0)
