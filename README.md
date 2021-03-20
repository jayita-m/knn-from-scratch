# knn-from-scratch

In this project, I've built a k-NN ML algorithm from scratch, that I've studied for different values of k on sklearn datasets (uses Python 3.8 interpreter) 

## k-Nearest Neighbor
* kNN is an ML algorithm that can be used for classification or regression. It is a supervised algorithm (meaning data is labelled to tell the machine what patterns to look for). It is relatively easy to understand. 

The main tasks of a kNN can be simplified as:  
1. Consider a data point in a vast array of data - and we are supposed to find the "type" of this data. 
2. We find the k neighbors nearest to this point. k is an arbitrary natural number, it is better to have k as an odd number though. 
3. The nearest neighbors are found using a distance formula for the new point and the other points around it. 
4. We then choose the k nearest neighbors to cast their "votes". 
5. The characteristic with the most "votes" is assigned to the data point. 

## Calculating Distance
For this algorithm, I've calculated the distance using **Minkowski Distance** where p=1, which translates into **Manhattan Distance/Taxicab Distance**. 
The Manhattan Distance is calculated by:

[image](https://user-images.githubusercontent.com/67204925/111870887-36d3ae80-8955-11eb-96c4-8d7ec1245c31.png)
