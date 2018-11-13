% Read in the data file from MNIST
rawTrainingImages = loadMNISTImages('train-images.idx3-ubyte');
trainingLabels = loadMNISTLabels('train-labels.idx1-ubyte'); % 60,000 labels

rawTestImages = loadMNISTImages('t10k-images.idx3-ubyte');
testLabels = loadMNISTLabels('t10k-labels.idx1-ubyte'); % 10,000 labels

% Break training images into columns. Consider making batches here later
column1 = rawTrainingImages(:, 1);

% Set initial values for weightMatrix and biases
weightMatrix = zeros(size(column1, 1), 1);
biases = zeros(size(column1, 1), 1);
biases(1, 1) = 1;

% Send data to algorithm
BackpropagationAlgorithm(column1, weightMatrix, biases)

% Handle the results


% Create graphical representations of the inputs and outputs

