 function [a, n] = ComputeLayerActivations(P, W, B)
    % Create an array to hold the output of each neuron in this layer
    neuronActivations = zeros(size(B, 1), 1);
    
    % Compute the neuron activations for the second layer
    for neuronIndex = 1:size(neuronActivations, 1)
        subset = W(1:size(P, 1), neuronIndex);              % Grab a subset of the weightMatrix values
        n = dot(subset, P) + B(neuronIndex);
        a = LogSigmoid(n);                                  % Compute the activation using this subset
        neuronActivations(neuronIndex) = a;                 % Store the activation
        subset = [];                                        % Reset the subset
    end
    
    % Return the set of activated neurons
    a = neuronActivations;       
end

