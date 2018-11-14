% Read in the data file from WNIST
rawTrainingImages = loadMNISTImages("train-images.idx3-ubyte");
rawTrainingLabels = loadMNISTLabels("train-labels.idx1-ubyte"); % 60,000 labels

% Don't use these until testing time comes
%rawTestImages = loadMNISTImages("t10k-images.idx3-ubyte");
%rawTestLabels = loadMNISTLabels("t10k-labels.idx1-ubyte"); % 10,000 labels

% Network Properties
numberOfLayers = 4;                                 % Network structure: R-16-16-10
layer1NeuronCount = size(rawTrainingImages, 1);     % Input layer
layer2NeuronCount = layer1NeuronCount / 49;         % Hidden layer for computation
layer3NeuronCount = layer1NeuronCount / 49;         % Hidden layer for computation
layer4NeuronCount = 10;                             % For the outputs 0-9
%costFunctionSize  = layer4NeuronCount;              % The values of each cost function for each output

% add the deriatives of the transfer functions here? 

% The values of each cost function for each output: C = (a^L - y)^2
%costFunction = zeros(costFunctionSize, 1);                                       

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

% Break training images into batches
batchSize = (size(rawTrainingImages, 2))/600;              % 100
numberOfBatches = (size(rawTrainingImages, 2))/batchSize;  % 600
costs = zeros(batchSize, numberOfBatches);                 % 100x600 = 60,000
for i = 1:numberOfBatches
    for j = 1:batchSize
        offset = j+((batchSize*i) - batchSize);            
        input = rawTrainingImages(:, offset);

        A1 = BackpropagationAlgorithm(input, W1, B2);
        A2 = BackpropagationAlgorithm(A1, W2, B3);
        A3 = BackpropagationAlgorithm(A2, W3, B4);

        labelVector = LabelToVector(rawTrainingLabels(j), zeros(layer4NeuronCount, 1));

        C = ComputeCost(A3, labelVector);

        costs(j, i) = C;
    end
end

uh = [];

% Create graphical representations of the inputs and outputs

