function [output] = RandomlyPopulate(input)
    for i = 1:size(input, 2) 
        for j = 1: size(input, 1)
            input(j, i) = randi([-1, 1]);
        end
    end
    
    output = input;
end

