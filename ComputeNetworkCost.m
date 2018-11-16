function [output] = ComputeNetworkCost(a, t)
    % Check to make sure the arrays are equal dimensions
    if size(a) ~= size(t)
        disp("Error. The arrays's do not match")
        return
    end

    % Get the mean squared error of the actual output, and the target
    % output
    output = t - a;
%     a = a.^2;
%     output = sum(a);
end

