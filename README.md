The distance method takes in a set of training points and for each inout data point, computes the average of the 
10 nearest neighbor points as well as the sampling distance from all points.  The latter is defined by the algorithm for feature distance
outlined in the main text.  It is performed for correlated and feature weighted data as well as uncorrelated, non 
fearure weighted data.  The results are grouped into 10 sets of points by average distance metric value; the mean error value is then 
calculated for the points in each of ten groups.
