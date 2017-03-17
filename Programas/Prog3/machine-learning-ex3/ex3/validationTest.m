clear all
close all
clc

% Space allocation
X = zeros(9,400);

% Convert the image data in the format of the neural network
for i = 1:9
  A = imread([pwd sprintf('/numbers/%d.png',i)]);
  A = A(:,:,1);
  X(i,:) = A(:);
end

save ex3validation.mat X
