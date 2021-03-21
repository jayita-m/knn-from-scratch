# knn-from-scratch

In this project, I've built a k-NN ML algorithm from scratch, that I've studied for different values of k on sklearn datasets (uses Python 3.8 interpreter) 

## k-Nearest Neighbor

The main tasks of a kNN algorithm can be simplified as:  
1. Consider a data point in a vast array of data - and we are supposed to find the "type" of this data. 
2. We find the k neighbors nearest to this point. k is an arbitrary natural number, it is better to have k as an odd number though. 
3. The nearest neighbors are found using a distance formula for the new point and the other points around it. 
4. We then choose the k nearest neighbors to cast their "votes". 
5. The characteristic with the most "votes" is assigned to the data point. 

## Calculating Distance
For this algorithm, I've calculated the distance using **Minkowski Distance** where p=1, which translates into **Manhattan Distance/Taxicab Distance**. 
The Manhattan Distance is calculated by:

![alt text](https://miro.medium.com/max/426/1*ph2xC44Zy-EHazYOom6tjg.png)

This is basically |x1 - x2| + |y1 -y2| for two points P1 and P2. 

## Results

**Accuracy:** 0.9736842105263158 
This is the accuracy of the kNN algorithm I built. I measured it using the 'accuracy_score' method of sklearn.metrics package. 

I range k from 1 to 99, and plot it against the accuracy obtained for each k. 

![alt text](https://i.imgur.com/45hefzm.jpg)


**Analysis:** Accuracy never crosses ~97%. It fluctuates around 95% until k = 40, after which it starts to decline. It reaches a low of ~55% as k closes to 99. 



