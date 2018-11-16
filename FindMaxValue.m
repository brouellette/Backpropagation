function [output] = FindMaxValue(a)    
    [val, idx] = max(a);
    a = zeros(size(a, 1), 1);
    a(idx) = 1;
    output = a;
end

