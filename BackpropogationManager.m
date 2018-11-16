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
costs = zeros(numberOfBatches, 1);

% For each of the batches, compute the network outputs for the inputs
for i = 1:numberOfBatches
    batchCosts = zeros(batchSize, 1);
    
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
        batchCosts(j) = layerCost;
    end
    
    costs(i) = mean(batchCosts);
    batchCosts = [];
end

% Update the network's weight and biases with the averaged batch cost
% averageBatchCost = mean(batchCosts);

% The function of the Log-sigmoid derivative 
logSigD = [];

% Need to update the weights and biases for each batch, so LOOP OVER THE
% BATCHES HERE

%  for batchCost = 1:size(costs, 1)
%     for layer = 1:layerCount
%         switch layer
%             case 3
%                 for i = 1:size(W1, 2)
%                     for j = 1:size(W1, 1)
%                         W1(j, i) = (2*batchCost)*(A2());
%                     end
%                 end
%             case 2
%                 
%             case 1
%            
%         end
%     end
%  end

% Create graphical representations of the inputs and outputs

