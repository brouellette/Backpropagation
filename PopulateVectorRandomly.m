function [output] = PopulateVectorRandomly(input)
    for i = 1:size(input, 1) 
        input(i) = randi([1, 3]);
    end
    
    output = input;
end

