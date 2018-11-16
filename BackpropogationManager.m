% Read in the data file from WNIST
rawTrainingImages = loadMNISTImages("train-images.idx3-ubyte");
rawTrainingLabels = loadMNISTLabels("train-labels.idx1-ubyte"); % 60,000

% Don't use these until testing time comes
rawTestImages = loadMNISTImages("t10k-images.idx3-ubyte");
rawTestLabels = loadMNISTLabels("t10k-labels.idx1-ubyte"); % 10,000

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

% For each of the batches, compute the network outputs for the inputs
for i = 1:numberOfBatches
    batchCost = zeros(neuronCountL3, 1);
    
    for j = 1:batchSize
        % Used to remember pointer location even when the batch set changes
        offset = j+((batchSize*i) - batchSize);   
        
        % Select the column representing the inputs according to the offset
        P = rawTrainingImages(:, offset);   

        % Compute all neuron activations for each of the layers
        [A1, N1] = ComputeLayerActivations(P, W1, B1);
        [A2, N2] = ComputeLayerActivations(A1, W2, B2);
        [A3, N3] = ComputeLayerActivations(A2, W3, B3);

        labelVector = LabelToVector(rawTrainingLabels(j), zeros(neuronCountL3, 1));

        layerCost = ComputeNetworkCost(A3, labelVector);
        
        % Store each cost to later take the average of them
        batchCost = batchCost + layerCost;
    end
    
    % Get the average batch
    batchCost = batchCost / batchSize;
        
    % Update weights and biases
    % Add in N values 
    S3 = 2 * diag(dlogsig(N3, A3) * batchCost');
    S2 = diag(dlogsig(N2, A2)) * (W3 * S3);
    S1 = diag(dlogsig(N1, A1)) * (W2 * S2);
    
    % Update the weights and biases for each layer 
    W1 = W1 - (1 * (S1*P')');
    W2 = W2 - (1 * (S2*A1')');
    W3 = W3 - (1 * (S3*A2')');
    
    B1 = B1 - (1 * S1);
    B2 = B2 - (1 * S2);
    B3 = B3 - (1 * S3);
    
    batchCost = zeros(neuronCountL3, 1);
end

% Testing
matchPercentage = zeros(size(rawTestImages, 2), 1);
for imageIndex = 1:size(rawTestImages, 2)
    P = rawTrainingImages(:, imageIndex);   

    % Compute all neuron activations for each of the layers
    A1 = ComputeLayerActivations(P, W1, B1);
    A2 = ComputeLayerActivations(A1, W2, B2);
    A3 = ComputeLayerActivations(A2, W3, B3);

    [val, idx] = max(A3);
    label = rawTrainingLabels(imageIndex);
    
    if idx == label
       % It's a match
       matchPercentage(imageIndex) = 1;
    end
end

finalPercentage = mean(matchPercentage);