the dataset need to be loaded manually on the cluster. The dataset is available at:
https://api.openml.org/d/554

the code uses the mnist_784.csv file

You could also modify the code so it fetches the dataset dinamically from OpenML, but depending on your cluster and the compute node you could get problems with the network connection, so it is better to download the dataset manually and upload it to the cluster.