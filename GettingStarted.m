% Read in the data file from MNIST
images = loadMNISTImages('train-images.idx3-ubyte');
labels = loadMNISTLabels('train-labels.idx1-ubyte');
 
% Display the data to understand it
disp(images(:,1:10)); % Show the first ten images
disp(labels(1:10)); % Show the first 10 labels


% Break into subgroups of around 12 elements?


% Loop through the subgroup and pass each element to the backprop network
Backpropagation()

% Handle the results


% Create graphical representations of the inputs and outputs

