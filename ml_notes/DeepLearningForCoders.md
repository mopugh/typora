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

* Be creative in how to represent data:
  
  * converting time series to images or mouse movement to images led to state-of-the-art results

  | Term | Meaning |

  |Label | The data that we're trying to predict, such as "dog" or "cat"
  |Architecture | The _template_ of the model that we're trying to fit; the actual mathematical function that we're passing the input data and parameters to
  |Model | The combination of the architecture with a particular set of parameters
  |Parameters | The values in the model that change what task it can do, and are updated through model training
  |Fit | Update the parameters of the model such that the predictions of the model using the input data match the target labels
  |Train | A synonym for _fit_
  |Pretrained model | A model that has already been trained, generally using a large dataset, and will be fine-tuned
  |Fine-tune | Update a pretrained model for a different task
  |Epoch | One complete pass through the input data
  |Loss | A measure of how good the model is, chosen to drive training via SGD
  |Metric | A measurement of how good the model is, using the validation set, chosen for human consumption
  |Validation set | A set of data held out from training, used only for measuring how good the model is
  |Training set | The data used for fitting the model; does not include any data from the validation set
  |Overfitting | Training a model in such a way that it _remembers_ specific features of the input data, rather than generalizing well to data not seen during training
  |CNN | Convolutional neural network; a type of neural network that works particularly well for computer vision tasks
  |=====

* **Tabular model**: tries to predict one column of a table based on information in other columns of the table.

* Datasets are food for models
  
  * consider starting with a smaller dataset and move to a larger dataset once one understands the problem.

* We can overfit the validation data since we (the modeler) are adjusting the hyperparameters to get better performance on the validation set.
  
  * Create a third set (besides test and validation): the **test set**
  
  * Only use the **test set** once at the end!
  
  * If using a 3rd party to model/solve problem, set aside a test set and metric to verify their work meets your needs.

* Make sure test/validation sets are representative of what you want to test
  
  * E.g. for time series, if you want to predict the future, make sure to hold out a future sequence to test
    
    * Don't select random time samples!
