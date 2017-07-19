function [mask, centerInds] = GetCandidatesFromScoreMap(score_map,prob_threshold)    
% function that applies a threshold to the score map, and outputs the
% binarized score map as well as coordinates of points in the binarized
% score map;
% Inputs:
%       score_map: 3D score map coming from FCN
%       prob_threshold: threshold to apply to score map (between 0-1)
% Outputs:
%       mask: 3D matrix obtained by applying threshold to score_map
%       centerInds: row/column/depth indices of points in the thresholded
%               mask; Note these will have to be adjusted if being used in
%               python (since matlab indexing starts from 1 and not 0)
    mask = logical(zeros(size(score_map)));    
    mask(score_map > prob_threshold) = 1;    
    [x y z] = ind2sub(size(mask),find(mask==1));
    centerInds = [x y z];
end