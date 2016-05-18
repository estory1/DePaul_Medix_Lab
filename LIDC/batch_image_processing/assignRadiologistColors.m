% assignRadiologistColors.m
%
% Original author: Evan Story (estory1@gmail.com)
%
% Purpose:
%
%% Assigns colors to the ROI pixels of an image, as well as painting the inclusion (exclusion) region.
%   img             = An matrix defining an RGB image.
%   y               = The y-axis of a set of 2D ROI pixels.
%   x               = The x-axis of a set of 2D ROI pixels.
%   rsId            = The readingSession_Id. Identifies the radiologist assigning the ROI.
%   inclusion       = Boolean string indicating whether the specified ROI is included in the radiologist's ROI. "FALSE" ==> exclusion region.
%   useConvexHull   = Boolean indicating whether to apply the convex hull function to the ROI.
function img = assignRadiologistColors(img, y, x, rsId, inclusion, useConvexHull)

    % Draw a convex hull if we can. Condition here checks for state that causes error "Error computing the convex hull. Not enough unique points specified."
    if useConvexHull && length(x) >= 3 && length(y) >= 3
        
        % Compute the convex hull for the points supplied in y & x.
        hullIndices = convhull(x, y);

        % Is exclusion region?
        if strcmpi(inclusion, 'FALSE') == 1
            for i=1 : length(x)
                img(y(i), x(i), 2:3) = 1;                           % cyan
            end
        % Not an exclusion region. Therefore, this region was marked as part of the nodule...
        else
            % Set the ROI color depending on which radiologist defined the ROI.
            % color defs: http://www.mathworks.com/help/matlab/ref/colorspec.html
            if      rsId == 0;   img(y(hullIndices), x(hullIndices), 1) = 1;      % red
            elseif  rsId == 1;   img(y(hullIndices), x(hullIndices), 2) = 1;      % green
            elseif  rsId == 2;   img(y(hullIndices), x(hullIndices), 3) = 1;      % blue
            elseif  rsId == 3;   img(y(hullIndices), x(hullIndices), 1:2) = 1;    % yellow
            end

        end
    % Convex hull not requested, OR insufficient points specified, so just fill-in wherever the radiologist indicated...
    else
        % Is exclusion region?
        if strcmpi(inclusion, 'FALSE') == 1
            for i=1 : length(x)
                img(y(i), x(i), 2:3) = 1;                           % cyan
            end
        % Not an exclusion region. Therefore, this region was marked as part of the nodule...
        else
            % Set the ROI color depending on which radiologist defined the ROI.
            % color defs: http://www.mathworks.com/help/matlab/ref/colorspec.html
            for i=1 : length(x)
                if      rsId == 0;   img(y(i), x(i), 1) = 1;      % red
                elseif  rsId == 1;   img(y(i), x(i), 2) = 1;      % green
                elseif  rsId == 2;   img(y(i), x(i), 3) = 1;      % blue
                elseif  rsId == 3;   img(y(i), x(i), 1:2) = 1;    % yellow
                end                
            end

        end 
    end

end