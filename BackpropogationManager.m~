% Read in the data file from MNIST
rawTrainingImages = loadMNISTImages("train-images.idx3-ubyte");
rawTrainingLabels = loadMNISTLabels("train-labels.idx1-ubyte"); % 60,000 labels

% Don't use these until testing time comes
%rawTestImages = loadMNISTImages("t10k-images.idx3-ubyte");
%rawTestLabels = loadMNISTLabels("t10k-labels.idx1-ubyte"); % 10,000 labels

% Network Properties
numberOfLayers = 4;                                 % Network structure: R-16-16-9
layer1NeuronCount = size(rawTrainingImages, 1);     % Input layer
layer2NeuronCount = layer1NeuronCount / 49;         % Hidden layer for computation
layer3NeuronCount = layer1NeuronCount / 49;         % Hidden layer for computation
layer4NeuronCount = 10;                             % For the outputs 0-9
costFunctionSize  = layer4NeuronCount;              % The values of each cost function for each output

% add the deriatives of the transfer functions here? 

% The values of each cost function for each output: C = (a^L - y)^2
costFunction = zeros(costFunctionSize, 1);                                       

% Initialize the weight matricies and biases for each layer
% TODO: Generalize this into a for-loop here using the number of layers and
% a string representation of the index for the range 1:numberOfLayers
% "layer" + i + "NeuronCount"?
W1 = PopulateVectorRandomly(zeros(layer1NeuronCount*layer2NeuronCount, 1));
B2 = PopulateVectorRandomly(zeros(layer2NeuronCount, 1));
W2 = PopulateVectorRandomly(zeros(layer2NeuronCount*layer3NeuronCount, 1));
B3 = PopulateVectorRandomly(zeros(layer3NeuronCount, 1));
W3 = PopulateVectorRandomly(zeros(layer3NeuronCount*layer4NeuronCount, 1));
B4 = PopulateVectorRandomly(zeros(layer4NeuronCount, 1));

% Break training images into columns. Consider making batches here later
inputRound1 = rawTrainingImages(:, 1);

% Send data to algorithm. Algorithm should return a set of activate
A1 = BackpropagationAlgorithm(inputRound1, W1, B2, rawTrainingLabels(1));
A2 = BackpropagationAlgorithm(A1, W2, B3, rawTrainingLabels(2));
A3 = BackpropagationAlgorithm(A2, W3, B4, rawTrainingLabels(3));

uh = [];

% Handle the results


% Create graphical representations of the inputs and outputs

