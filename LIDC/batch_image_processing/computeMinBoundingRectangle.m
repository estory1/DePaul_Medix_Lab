% computeMinBoundingRectangle.m
%
% Original author: Evan Story (estory1@gmail.com)
%
% Purpose:
%
%% Computes the minimum bounding box around a 2D ROI.
function [mnmn, mnmx, mxmn, mxmx] = computeMinBoundingRectangle (roiVertices)

    % compute boundary rectangle vertices
    mnmn = [min(roiVertices(:,1)), min(roiVertices(:,2))];
    mnmx = [min(roiVertices(:,1)), max(roiVertices(:,2))];
    mxmn = [max(roiVertices(:,1)), min(roiVertices(:,2))];
    mxmx = [max(roiVertices(:,1)), max(roiVertices(:,2))];

end