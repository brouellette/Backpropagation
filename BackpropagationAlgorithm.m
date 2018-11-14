 function [] = BackpropagationAlgorithm(P, W, B, t)
    % Create an array to hold the output of each neuron in this layer
    neuronActivations = zeros(size(B, 1), 1);
 
    % Compute the neuron activations for the second layer
    for neuronIndex = 1:size(neuronActivations, 1)
        subsetStartIndex = (1 + ((neuronIndex*784) - 784));
        subsetEndIndex = (neuronIndex*784);
        subset = W([subsetStartIndex:subsetEndIndex], :);
        a = LogSigmoid((P'*subset) + B(neuronIndex));
        neuronActivations(neuronIndex) = a;
        subset = [];
    end
    
    % Compute the error: e = (t - a)
    
    
    
    % Compute the sensitivities through backpropogation
    
    
    % Update the weights and biases
        
end

