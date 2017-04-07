# Description

bfalg_WTA is a Beachfront service that extracts shorelines from RapidEye satellite imagery.  This service utilizes a winner-takes-all approach to shoreline extraction, rather than relying on any single band ratio or normalized index.  This service calculates multiple combinations of band ratios and normalized indices, then reduces the results stack into binary classes, water and not-water.  Using the Beachfront-py (http://github.com/venicegeo/beachfront-py), the binary mask is then converted into geojson vectors.

# Operation
Bfalg_WTA can be called directly from python, or through the use of a Docker container.  

## Python Syntax:

usage: bfalg_WTA.py -i INPUT -o OUTFILE
                    [-m method] [-p sampling percentage] [-v version] [-s simplification tolerance]
					
| Keys | Description
------|-------------------
| -i | Input RapidEye image in geotif format	
| -o | Output Geojson file
| -m | Method for reducing the result stack to binary classes. 
|    | The default is 1 (Gaussian Mixtures)
|    | 2 (KMeans)
|    | 3 (Feature Agglomeration)
|    | 4 (Principle Components Transformation)
| -s | Simplification tolerance for simplifying the output geojson vectors.  Recommend 0.00035 for RapidEye images in WGS84
| -p | Percentage as a float for sampling the results stack when reducing to binary classes.  A lower percentage can greatly speed up the process, but can potentially impact results.  Default is 0.25
| -v | Specify the version of the algorithm to use.  V1 writes the binary mask to a tif as an intermediate step.  V2 does not, but has a input image size limit.

Sample Query:
~~~
 $python bfalg_WTA.py -i input.tif -o output.geojson -m 4 -s 0.00035 -p 0.15
~~~



## Docker

A Dockerfile is included for ease of development. The built docker image provides all the system dependencies needed to run the tool. The tool can also be tested locally, but all system dependencies must be installed first, and the use of a virtualenv is recommended.

To build the docker image:

~~~
$ docker build -t wta:latest .
~~~

Run this image by creating a symbolic link between the folder containing your input image and the docker containers /work/data.  Pass your arguments in the docker command statement

~~~
$docker run -v ~/LocalRapidEyeFolder:/work/data -d wta:latest /bin/bash -c "python bfalg_WTA.py -i data/Input.tif -o data/Output.geojson"
~~~


# Algorithm


Beachfront’s Winner-Takes-All shoreline extraction algorithm was designed specifically for RapidEye, though it could likely be modified for use with Landsat-8, PlanetScope, and other MSI image sources.
The WTA algorithm works by first generating a 4 band image stack of various water indices.  This consists of a standard NDWI index, a Green/NIR simple ratio, a Red/NIR simple ratio, and a ratio comparing the green band to the sum of the remaining bands.  The algorithm was designed specifically for RapidEye, though it could likely be modified for use with Landsat-8, PlanetScope, and other MSI image sources.  This algorithm is designed to reduce or eliminate false positives buy removing the reliance on a single index.

~~~
Index_1 = (Green - NIR)/(Green + NIR)
Index_2 = Green/NIR
Index_3 = Red/NIR
Index_4 = Green / (Blue + Red + Red Edge + NIR)
~~~

There are many possible ways to calculate a single winning set of classes from the results stack.  Through extensive research and testing, we found that while the four included methods were generally all suitable, each method has scenes in which is it a far superior method.  The default method, Gaussian Mixtures is generally the ideal, but the other methods were left in place for testing and user convenience.  There are 2 general approaches that this tool uses to reduce the water indice stack to a binary image.  Unsupervised classification attempts to examine the data in its 4-band configuration and cluster the pixels into similar classes. Kmeans and Gaussian Mixture Modeling are examples of unsupervised classification.  The second approach are techniques utilized to reduce high complexity data to low complexity.  After reducing the complexity to a single channel, the tool then calculats the otsu threshold in order to segregate the now single band image into a binary image.  These data dimensionality reduction techniques included feature agglomeration and principal components analysis.  It is important to note that as a default parameter, the classes are trained using a 25% sample of the image pixels, rather than the entire image.  Testing has shown this to  result in increased performance speed with limited impact on result accuracy.

The KMeans algorithm clusters data by trying to separate samples in n groups of equal variance, minimizing a criterion known as the inertia or within-cluster sum-of-squares. Inertia, or the within-cluster sum of squares criterion, can be recognized as a measure of how internally coherent clusters are. Inertia makes the assumption that clusters are convex and isotropic, which is not always the case. It responds poorly to elongated clusters, or manifolds with irregular shapes.  This algorithm requires the number of clusters to be specified. It scales well to large number of samples and has been used across a large range of application areas in many different fields.

A Gaussian mixture model is a probabilistic model that assumes all the data points are generated from a mixture of a finite number of Gaussian distributions with unknown parameters. One can think of mixture models as generalizing k-means clustering to incorporate information about the covariance structure of the data as well as the centers of the latent Gaussians.  The Scikit-learn Gaussian Mixture object implements the expectation-maximization (EM) algorithm for fitting mixture-of-Gaussian models. Gaussian mixture modeling predicts a Gaussian Mixture Model then assigns to each sample the Gaussian it mostly probably belongs to.

Agglomerative clustering is a bottom-up approach: each observation starts in its own cluster, and clusters are iteratively merged in such a way to minimize a linkage criterion. This approach is particularly interesting when the clusters of interest are made of only a few observations. When the number of clusters is large, it is much more computationally efficient than k-means.  Feature agglomeration is similar to agglomerative clustering.  This approach can be implemented by clustering in the feature direction, in other words clustering the transposed data.

Principal component analysis (PCA) is a statistical procedure that uses an orthogonal transformation to convert a set of observations of possibly correlated variables into a set of values of linearly uncorrelated variables called principal components. This transformation is defined in such a way that the first principal component has the largest possible variance(that is, accounts for as much of the variability in the data as possible), and each succeeding component in turn has the highest variance possible under the constraint that it is orthogonal to the preceding components. The resulting vectors are an uncorrelated orthogonal basis set. PCA is sensitive to the relative scaling of the original variables

