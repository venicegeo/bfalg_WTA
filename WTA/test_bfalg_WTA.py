import  WTA.bfalg_WTA as wta
import os

def test_getRandomCoordinate():
    x,y = wta.getRandomCoordinate(10,10)
    assert x <= 10
    assert y <= 10


def test_getRandomSample():
    img = wta.readImage('test/fixtures/image1.tif')
    sample = wta.getRandomSample(img, nSamples = 10)
    x,y = sample.shape# THIS IS DEFINITELY NOT WRITTEN CORRECTLY.  TEST and FIX
    assert x == 10
    sample = wta.getRandomSample(img, percentSample=0.01)


def test_BuildWaterMasks():
    mask = wta.BuildWaterMasks('test/fixtures/image1.tif')
    x,y,z = mask.shape
    assert z == 4


def test_xO_Kmeans():
    mask = wta.BuildWaterMasks('test/fixtures/image1.tif')
    binary = wta.xO_Kmeans(mask, percentage = 0.05, n_clusters=2)
    assert binary.max() < 2


def test_xO_FA():
    mask = wta.BuildWaterMasks('test/fixtures/image1.tif')
    binary = wta.xO_FA(mask, percentage = 0.05, n_clusters=2)
    assert binary.max() < 2


def test_xO_PCA_inMem():
    mask = wta.BuildWaterMasks('test/fixtures/image1.tif')
    binary = wta.xO_PCA_inMem(mask, n_bands=3, ot='float16')


def test_PCA_Binary_Thresh():
    mask = wta.BuildWaterMasks('test/fixtures/image1.tif')
    binary = wta.PCA_Binary_Thresh(mask)
    print 'Binary Max is %s' % binary.max()
    assert binary.max() < 2


def test_xO_GM():
    mask = wta.BuildWaterMasks('test/fixtures/image1.tif')
    binary = wta.xO_GM(mask, percentage = 0.05, n_clusters=2)
    assert binary.max() < 2


def test_saveArrayAsRaster():
    img = wta.readImage('test/fixtures/image1.tif')
    wta.saveArrayAsRaster('test/fixtures/image1.tif',
	                  'test/fixtures/image2.tif',
			  img)
    assert os.path.exists('test/fixtures/image2.tif')


def test_WinnerTakesAll():
    binary = wta.WinnerTakesAll('test/fixtures/image1.tif', outName='test/fixtures/binary1.tif', method=1, percentage=0.05)
    binary = wta.WinnerTakesAll('test/fixtures/image1.tif', outName='test/fixtures/binary2.tif', method=2, percentage=0.05)
    binary = wta.WinnerTakesAll('test/fixtures/image1.tif', outName='test/fixtures/binary3.tif', method=3, percentage=0.05)
    binary = wta.WinnerTakesAll('test/fixtures/image1.tif', outName='test/fixtures/binary4.tif', method=4, percentage=0.05)

def test_VectorizeBinary():
    result = wta.VectorizeBinary('test/fixtures/binary1.tif', outName='test/fixtures/results1.geojson')#, simple=0.00035)


def test_WTA_v1():
    result = wta.WTA_v1('test/fixtures/image1.tif', outName='test/fixtures/results_v1.geojson', percentage=0.05)#, simple=0.00035)
    assert os.path.exists('test/fixtures/results_v1_binary.tif')
    assert os.path.exists('test/fixtures/results_v1.geojson')

def test_WTA_v2():
    result = wta.WTA_v2('test/fixtures/image1.tif', outName='test/fixtures/results_v2.geojson', percentage=0.05, minsize=100.0)#, simple=0.00035)
    assert os.path.exists('test/fixtures/results_v2.geojson')

