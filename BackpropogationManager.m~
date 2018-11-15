% Read in the data file from WNIST
rawTrainingImages = loadMNISTImages("train-images.idx3-ubyte");
rawTrainingLabels = loadMNISTLabels("train-labels.idx1-ubyte"); % 60,000 labels

% Don't use these until testing time comes
%rawTestImages = loadMNISTImages("t10k-images.idx3-ubyte");
%rawTestLabels = loadMNISTLabels("t10k-labels.idx1-ubyte"); % 10,000 labels

% Network Properties
layerCount = 4;                                 % Network structure: R-16-16-10

inputCount = size(rawTrainingImages, 1);     % Input layer
neuronCountL1 = inputCount / 49;         % Hidden layer for computation
neuronCountL2 = inputCount / 49;         % Hidden layer for computation
neuronCountL3 = 10;                             % For the outputs 0-9

% Weights and Biases
W1 = [];
B1 = [];
W2 = [];
B2 = [];
W3 = [];
B3 = [];

% Execute the intial network setup using the external inputs
for layer = 1:layerCount - 1 
    switch layer 
        case 1
            W1 = PopulateVectorRandomly(zeros(inputCount*neuronCountL1, 1));
            B1 = PopulateVectorRandomly(zeros(neuronCountL1, 1));
        case 2
            W2 = PopulateVectorRandomly(zeros(neuronCountL1*neuronCountL2, 1));
            B2 = PopulateVectorRandomly(zeros(neuronCountL2, 1));
        case 3
            W3 = PopulateVectorRandomly(zeros(neuronCountL2*neuronCountL3, 1));
            B3 = PopulateVectorRandomly(zeros(neuronCountL3, 1));
    end
end

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

        labelVector = LabelToVector(rawTrainingLabels(j), zeros(neuronCountL3, 1));

        C = ComputeCost(A3, labelVector);
        
%         S = (2*C)*(transferFuncDerivative);

        costs(j, i) = C;
    end
end

% Create graphical representations of the inputs and outputs

