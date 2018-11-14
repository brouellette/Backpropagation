 function [output] = BackpropagationAlgorithm(P, W, B)
    % Create an array to hold the output of each neuron in this layer
    neuronActivations = zeros(size(B, 1), 1);
 
    % Compute the neuron activations for the second layer
    for neuronIndex = 1:size(neuronActivations, 1)
        offset = size(P, 1);                                % Set the size of the subsets
        subsetStartIndex = (1 + ((neuronIndex*offset) - offset));
        subsetEndIndex = (neuronIndex*offset);
        subset = W([subsetStartIndex:subsetEndIndex], :);   % Grab a subset of the weightMatrix values
        a = LogSigmoid((P'*subset) + B(neuronIndex));       % Compute the output using this subset
        neuronActivations(neuronIndex) = a;                 % Store the activation
        subset = [];                                        % Reset the subset value
    end
    
    % Return the set of activated neurons
    output = neuronActivations;
    
    % Compute the neuron activations for the third layer
    
    
    % Compute the error: e = (t - a)
    
    
    
    % Compute the sensitivities through backpropogation
    
    
    % Update the weights and biases
        
end

