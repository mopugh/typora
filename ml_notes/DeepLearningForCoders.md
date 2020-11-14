# Deep Learning For Coders With fastai and PyTorch

## Chapter 1: Your Deep Learning Journey

* **Deep learning** is a computer technique to extract and transform data by using multiple layers of neural networks.
  
  * Each layer takes its inputs from previous layers and progressively refines them.
  
  * Use algorithm to minimize errors

* Approach
  
  * Teaching the whole game
  
  * Always teaching through examples
  
  * Simplifying as much as possible
  
  * Removing barriers

* Traits of successful deep learning practioners
  
  * playfulness
  
  * curiosity

* Typical programming:
  
  ![](/home/mopugh/Documents/typora/ml_notes/figures/2020-11-13-06-10-07-image.png)
  
  where the programmer programs all the individual steps

* Idea: show a computer examples of a problem to solveand have it learn to solve the problem

* Ideas from original AI essay from Arthur Samuel:
  
  * weight assignment
    
    ![](/home/mopugh/Documents/typora/ml_notes/figures/2020-11-13-06-15-55-image.png)
    
    * weights as another type of input
    
    * now called "model parameters"
  
  * performance evaluated on weight assignment
  
  * automatic way to test performance
  
  * mechanism to improve performance by changing the weight assignment
    
    ![](/home/mopugh/Documents/typora/ml_notes/figures/2020-11-13-06-18-54-image.png)

* Once model is done training, the weights are no longer updated and are considered part of the model:
  
  how to use a model (identical to above with program replaced by model)
  
  ![](/home/mopugh/Documents/typora/ml_notes/figures/2020-11-13-06-20-06-image.png)

* A trained model can be treated just like a regular computer program

* **Machine learning**: the training of programs developed by allowing a computer to learn from its experience, rather than through manual coding the individual steps.

* Neural networks can approximate any function by the **universal approximation theorem**

* **Stochastic gradient descent** is a general procedure to learn the function.

* Updated process with modern nomenclature
  
  ![](/home/mopugh/Documents/typora/ml_notes/figures/2020-11-13-06-27-44-image.png)

* **Overfitting is the single most important and challenging issue**
  
  * A **metric** is a function that measures the quality of the model's predictions (on the validation set).
    
    * A metric is for human consumption
    
    * A **loss** is for SGD

* Use pre-trained networks to start when possible!

* Using a pretrained model for a task different from what it was originally trained for is known as **transfer learning**. 

* **Fine tuning**: a transfer learning technique that updates the parameters of a pre-trained model by training for additional epochs using a different task from that used for pre-training.

* **Epochs**: Number of time each datapoint is seen by the algorithm. 
