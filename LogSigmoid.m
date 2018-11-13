function [output] = LogSigmoid(N)
    output = 1 / (1 + exp(-N));
end

