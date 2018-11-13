% Read in the data file from MNIST
rawTrainingImages = loadMNISTImages("train-images.idx3-ubyte");
rawTrainingLabels = loadMNISTLabels("train-labels.idx1-ubyte"); % 60,000 labels

% Don't use these until testing time comes
%rawTestImages = loadMNISTImages("t10k-images.idx3-ubyte");
%rawTestLabels = loadMNISTLabels("t10k-labels.idx1-ubyte"); % 10,000 labels

% Network Properties
numberOfLayers = 4;                                 % Network structure: R-16-16-9
layer1NeuronCount = size(rawTrainingImages, 1);     % Input layer
layer2NeuronCount = 16;                             % Hidden layer for computation
layer3NeuronCount = 16;                             % Hidden layer for computation
layer4NeuronCount = 10;                             % For the outputs 0-9
costFunctionSize  = layer4NeuronCount;              % The values of each cost function for each output

% The values of each cost function for each output: C = (a^L - y)^2
costFunction = zeros(costFunctionSize, 1);                                       

% Initialize the weight matricies and biases for each layer
% TODO: Generalize this into a for-loop here using the number of layers and
% a string representation of the index for the range 1:numberOfLayers
% "layer" + i + "NeuronCount"
layer1WeightMatrix = PopulateVectorRandomly(zeros(layer1NeuronCount, 1));
layer1Biases = PopulateVectorRandomly(zeros(layer1NeuronCount, 1));
layer2WeightMatrix = PopulateVectorRandomly(zeros(layer2NeuronCount, 1));
layer2Biases = PopulateVectorRandomly(zeros(layer2NeuronCount, 1));
layer3WeightMatrix = PopulateVectorRandomly(zeros(layer3NeuronCount, 1));
layer3Biases = PopulateVectorRandomly(zeros(layer3NeuronCount, 1));
layer4WeightMatrix = PopulateVectorRandomly(zeros(layer4NeuronCount, 1));
layer4Biases = PopulateVectorRandomly(zeros(layer4NeuronCount, 1));

% Break training images into columns. Consider making batches here later
inputRound1 = rawTrainingImages(:, 1);

% Send data to algorithm. Algorithm should return a cost function for the
% label that was passed in
BackpropagationAlgorithm(inputRound1, layer1WeightMatrix, layer1Biases, rawTrainingLabels(1))

% Handle the results


% Create graphical representations of the inputs and outputs

