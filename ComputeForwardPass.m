function [a] = ComputeForwardPass(p, W, b)
    % Compute net input n
    n = (p' * W) + b(1, 1);
    
    % Apply transfer function to n
    a = LogSigmoid(n);
    
    % 
end

