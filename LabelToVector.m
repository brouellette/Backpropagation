function [output] = LabelToVector(target, costFunction)
    % Check to make sure the target is a legal value
    if target < 0
        disp("Error. Target value is less than the minimum allowed value")
        return
    else if target > 9
            disp("Error. Target value is less than the maximum allowed value")
            return
        end
    end

    % Find the target value between 0-9 and set it accordingly within the cost function
    for i = 1:size(costFunction, 1)
        if i == (target + 1)
            costFunction(i) = 1;
        end
    end
    
    output = costFunction;
end

