function [pred, knn_label, knn_ind] = knn(querys, trains, labels, k, distance, dist_func)

% function to calculate single-shot k-nearest-neighbour
% args:
% query:    the vector whose label we want to predict, an 1*n_dim matrix or a
%           vector of length n_dim
% trains:   the training data, an N*n_dim matrix with each row be a sample
% labels:   the label corresponding to the training data
% k:        the number of nearest neighbours
% distance: the string specifying the distance function, for exp.
%           'eucliedean', 'manhattan' .... please refer to matlab doc
% dist_func: a function handle where you can specify your own distance 
%            function

% outputs:
% pred:      the predicted label
% knn_label: the k nearest sample label
% knn_sample: the k nearest sample

% specify what distance you are using, default 'euclidean'
% provice your own distance function if set distance to be "user_define"
% then provide a function handle to dist_func
if (nargin == 5)
    dist = distance;
elseif (nargin >5) && (distance == "user_define") 
    dist = dist_func;
else
    dist = 'euclidean';
end

% compute pair-wise distance from 2 dataset and return M*N distance matrix
% see "doc pdist2" for more information
d = pdist2(querys, trains, dist);
[~, mink_ind] = mink(d, k, 2);

% use the following lines to assign value to the return variables.
knn_label = labels(mink_ind);
knn_ind = mink_ind;
pred = mode(knn_label, 2);

end