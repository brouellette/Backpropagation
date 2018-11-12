% Read in the data file from MNIST
trainingImages = loadMNISTImages('train-images.idx3-ubyte');
trainingLabels = loadMNISTLabels('train-labels.idx1-ubyte'); % 60,000 labels

testImages = loadMNISTImages('t10k-images.idx3-ubyte');
testLabels = loadMNISTLabels('t10k-labels.idx1-ubyte'); % 10,000 labels

% nnd11bc is the test program from the NND book to see all the algorithm
 
% Display the data to understand how the numbers are represented
disp(trainingImages(:,1:10)); % Show the first ten images
disp(trainingLabels(1:10)); % Show the first 10 labels

% Break into subgroups of around 12 elements?


% Loop through the subgroup and pass each element to the backprop network
Backpropagation()

% Handle the results


% Create graphical representations of the inputs and outputs

