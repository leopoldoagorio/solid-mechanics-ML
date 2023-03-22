# mamaML
First tests in automating a pipeline for generating automatically a database of FEM solutions to unidimensional compression/extension problem, and learning with a Neural Network to solve same mechanical problem.

## Meetings

Fridays 10:00 am at: [![zoom](https://img.shields.io/badge/zoom-meetings-red)](https://salavirtual-udelar.zoom.us/j/88647392899)


### 1.02.22

- We discussed different methods and algorithms that can be used to simulate structures.
- Leo and Mauricio present some minor scripting results on uniaxial compression example.
- Bruno shared a [reference](https://doi.org/10.1016/j.engappai.2018.01.006) denoting a skepticism on the ability of ML models to generalize well.
- Bruno shared a [reference](https://doi.org/10.1016/j.advengsoft.2005.03.022) of structural optimization using ML model instead of FEA analyses.
- Bruno shared a [reference](https://doi.org/10.1080/17415970600573411) of structural parameter identification using ML model instead of FEA analyses.
- We decided to generate 1000 samples changing E, Lx, p and to predict the value of $u_x$ at $x$ = [ $L_x$ , $0$ , $0$ ]  

### 8.02.22
- Data visualization seems to not be useful since we have an analytic solution for the displacements
- Mauricio shares [this reference ](https://www.sciencedirect.com/science/article/pii/S2666990022000040)about surrogate modes to predict breast displacements field. 
-  Leo shows showed scripts to generate .csv results. Features [ $L_x$ ,  $E$ ,  $p$ ] targets  [ $u_x$ , $u_y$ , $u_z$ ]
- Bruno contributes to the analytic solution of the uni-axial compression example.

### 08.03.22
- We discussed possible reasons on difference between validation and train losses. We opened an issue #33 to attack this strange behavior.  
- We set the definition for the new two materials example with ONSAS #31. 
- Bruno and Mauricio discussed experimental results obtained with a foam cube by Santiago from IIMPI as a possible example to model in ONSAS. 
- Bruno agreed on the MLP model implementation.
- Bruno shared [this book](https://www.amazon.com/Hands-Machine-Learning-Scikit-Learn-TensorFlow/dp/1492032646)
- Leo shared XGboost tool form sklearn. And we defined to try it on the first example. 

### 17.03.22
- We discussed the reasons why analytic test is lower than train/validation. We conclude that inputs in the border of the training has an important impact on the analytic test error. Finally, Leo run an acceptable range. 

- We set final goals for the project, excluding XGBoost by the moment. If we get MLP results and we add sufficient detailed documentation on the obtained results then we can try XGBoost. 

- The next example will take 6000 random samples avoiding repetition and split 80/20 for train and validation. 