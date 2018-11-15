% Read in the data file from WNIST
rawTrainingImages = loadMNISTImages("train-images.idx3-ubyte");
rawTrainingLabels = loadMNISTLabels("train-labels.idx1-ubyte"); % 60,000

% Don't use these until testing time comes
%rawTestImages = loadMNISTImages("t10k-images.idx3-ubyte");
%rawTestLabels = loadMNISTLabels("t10k-labels.idx1-ubyte"); % 10,000

% Network Properties
layerCount = 4;                             % Network structure: R-16-16-10

inputCount = size(rawTrainingImages, 1);    % Input layer
neuronCountL1 = inputCount / 49;            % Hidden layer for computation
neuronCountL2 = inputCount / 49;            % Hidden layer for computation
neuronCountL3 = 10;                         % For the outputs 0-9

% Weights and Biases
W1 = [];
B1 = [];
W2 = [];
B2 = [];
W3 = [];
B3 = [];

% Execute the intial network setup using network properties
for layer = 1:layerCount - 1 
    switch layer 
        case 1
            W1 = RandomlyPopulate(zeros(inputCount, neuronCountL1));
            B1 = RandomlyPopulate(zeros(neuronCountL1, 1));
        case 2
            W2 = RandomlyPopulate(zeros(neuronCountL1, neuronCountL2));
            B2 = RandomlyPopulate(zeros(neuronCountL2, 1));
        case 3
            W3 = RandomlyPopulate(zeros(neuronCountL2, neuronCountL3));
            B3 = RandomlyPopulate(zeros(neuronCountL3, 1));
    end
end

% Break the training images into batches
batchSize = (size(rawTrainingImages, 2))/600;              % 100
numberOfBatches = (size(rawTrainingImages, 2))/batchSize;  % 600
batchCosts = zeros(numberOfBatches, 1);

% For each of the batches, compute the network outputs for the inputs
for i = 1:numberOfBatches
    for j = 1:batchSize
        % Used to remember pointer location even when the batch set changes
        offset = j+((batchSize*i) - batchSize);   
        
        % Select the column representing the inputs according to the offset
        P = rawTrainingImages(:, offset);   

        % Compute all neuron activations for each of the layers
        A1 = ComputeLayerActivations(P, W1, B1);
        A2 = ComputeLayerActivations(A1, W2, B2);
        A3 = ComputeLayerActivations(A2, W3, B3);

        labelVector = LabelToVector(rawTrainingLabels(j), zeros(neuronCountL3, 1));

        layerCost = ComputeNetworkCost(A3, labelVector);
        
        % Store each cost to later take the average of them
        batchCosts(i) = layerCost;
    end
end

% Update the network's weight and biases with the averaged batch cost
averageBatchCost = mean(batchCosts);

for layer = 1:layerCount
   switch layer
       case 1
           
       case 2
           
       case 3
           
   end
end

% Create graphical representations of the inputs and outputs

