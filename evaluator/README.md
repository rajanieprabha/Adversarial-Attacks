# evaluator

We've seen very different implementations for specific attacks on specific models etc. 
In order to make evaluations comparable and reproducible we've implemented our own evaluation framework 
that will allow us to configure experiments in a modular way and helps to perform different evaluations easily.

```
.
+-- attacks       
    +-- ...          # strong attack like CW l2
+-- models        
    +-- ...          # standard CNN and ResNet model
+-- dataset.py       # MNIST and CIFAR utils
+-- defense.py       # dropout uncertainty based and LID
+-- solver.py        # train / evaluate / attack a model on a dataset
+-- utils.py         # helper functions
+-- visualization.py # histograms, ROC curves, ...
```